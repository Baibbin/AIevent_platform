from flask import Flask, render_template, jsonify, send_from_directory, request
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
import threading
import time
from enum import Enum
from ultralytics import YOLO
import platform
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
# 新增：数据库相关导入
import sqlite3
from sqlite3 import Error
import json

app = Flask(__name__)

# ===================== 配置参数 =====================
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# 新增：数据库配置
DB_FILE = "ai_detection_events.db"  # 数据库文件路径
DB_LOCK = threading.Lock()  # 数据库操作锁（保证线程安全）

# 视频/摄像头来源
class SourceType(Enum):
    CAMERA = "摄像头"
    VIDEO = "本地视频"

SOURCES = {
    "camera_0": {"type": SourceType.CAMERA.value, "path": 0},
    "video_0": {"type": SourceType.VIDEO.value, "path": "./vedio/huapo.mp4"},
    "video_1": {"type": SourceType.VIDEO.value, "path": "./vedio/cheliu2.mp4"},
    "video_2": {"type": SourceType.VIDEO.value, "path": "./vedio/people.mp4"},
    "video_3": {"type": SourceType.VIDEO.value, "path": "./vedio/light_2.mp4"},
    "video_4": {"type": SourceType.VIDEO.value, "path": "./vedio/qiaoliang.mp4"},
    "video_5": {"type": SourceType.VIDEO.value, "path": "./vedio/dawu.mp4"},
    "video_6": {"type": SourceType.VIDEO.value, "path": "./vedio/lmtx.mp4"},
}

# 检测功能枚举
class DetectFunc(Enum):
    PEDESTRIAN = "行人检测（YOLOv8）"
    CHANGE_DETECT = "变化检测（帧差法+SSIM）"
    LANDSLIDE = "滑坡检测（帧差法）"
    VEHICLE = "车辆检测（YOLOv8）"
    BRIDGE_COLLAPSE = "桥梁坍塌检测（帧差法）"
    PIER = "桥墩检测（YOLOv8-ONNX）"
    NIGHT_TRAJECTORY = "夜间轨迹检测（车灯）"
    VISIBILITY = "能见度检测"
    PIT = "路面塌陷检测（YOLOv8）"

# 通用配置
ROI_POINTS = [(1, 1), (2000, 1), (2000, 1000), (1, 1000)]

# 变化检测算法配置
MIN_CONTOUR_AREA_CHANGE = 500
CHANGE_THRESHOLD = 0.01
SSIM_THRESHOLD = 0.80

# 行人检测算法配置
YOLO_MODEL_PATH = "./model/yolov8n.pt"
PERSON_CLASS_ID = 0
YOLO_CONF_THRESH = 0.5

# 车辆检测算法配置
VEHICLE_CLASS_ID = 2
VEHICLE_CONF_THRESH = 0.3  # 降低置信度阈值，提高检测灵敏度

# 车辆跟踪和违章检测配置
VEHICLE_TRACKING_CONFIG = {
    'track_threshold': 100,  # 增加匹配阈值，提高跟踪鲁棒性
    'max_disappeared': 15,   # 增加最大消失帧数，允许车辆短暂消失
    'min_tracking_frames': 3 # 最少跟踪帧数
}

# 拥堵检测配置
CONGESTION_CONFIG = {
    'low_threshold': 5,    # 低拥堵阈值（车辆数）
    'medium_threshold': 10,  # 中拥堵阈值（车辆数）
    'high_threshold': 15,    # 高拥堵阈值（车辆数）
    'time_window': 3000      # 时间窗口（毫秒），用于平滑判断
}

# 能见度检测配置
VISIBILITY_CONFIG = {
    'clear_threshold': 70,       # 清晰阈值（分数）
    'light_fog_threshold': 50,   # 轻度雾阈值（分数）
    'moderate_fog_threshold': 20, # 中度雾阈值（分数）
    # 重度雾：小于等于moderate_fog_threshold
}

# 路面塌陷检测配置
PIT_MODEL_PATH = "./model/best_pit.pt"
PIT_CLASS_ID = 0
PIT_CONF_THRESH = 0.5

# 逆行检测配置
WRONG_WAY_CONFIG = {
    'direction_threshold': 5,  # 方向判断阈值（像素）
    'min_wrong_way_frames': 2  # 最少逆行帧数
}

# 违法停车检测配置
ILLEGAL_PARKING_CONFIG = {
    'parking_threshold': 1,  # 停车判断阈值（秒）
    'min_parking_area': 5,  # 最小停车区域（像素）
    'parking_zones': [[(0, 0), (1500, 0), (1500, 1000), (0, 1000)]]  # 停车区域
}

# 桥墩检测ONNX配置
PIER_ONNX_MODEL_PATH = "./model/best_pier.onnx"
PIER_CONF_THRESH = 0.4
PIER_IOU_THRESH = 0.45
PIER_INPUT_SHAPE = (640, 640)
PIER_CLASS_NAMES = ["桥墩"]
pier_onnx_session = None
try:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    pier_onnx_session = ort.InferenceSession(
        PIER_ONNX_MODEL_PATH,
        providers=providers,
        sess_options=session_options
    )
    print(f"成功用onnxruntime加载桥墩模型：{PIER_ONNX_MODEL_PATH}")
except Exception as e:
    print(f"桥墩ONNX模型加载失败：{e}")
    pier_onnx_session = None

# 滑坡检测算法配置
MIN_CONTOUR_AREA_LANDSLIDE = 2000
LANDSLIDE_THRESHOLD = 0.05
LANDSLIDE_KERNEL = (9, 9)

# 桥梁坍塌检测算法配置
MIN_CONTOUR_AREA_BRIDGE = 2000
BRIDGE_COLLAPSE_THRESHOLD = 0.05
BRIDGE_COLLAPSE_KERNEL = (9, 9)

# 夜间轨迹检测配置
NIGHT_TRAJECTORY_CONFIG = {
    'history_length': 50,        # 轨迹历史长度
    'brightness_threshold': 200,  # 高亮区域提取阈值
    'motion_threshold': 30,       # 运动检测阈值
    'trajectory_alpha': 0.95,     # 轨迹衰减系数
    'gaussian_kernel': (5, 5),    # 高斯模糊核大小
    'morph_open_kernel': (3, 3),  # 形态学开操作核大小
    'morph_close_kernel': (5, 5)  # 形态学闭操作核大小
}

# 全局状态管理
global_state = {
    "current_source": "video_0",
    "current_func": DetectFunc.PEDESTRIAN.value,
    "prev_frame": None,
    "curr_frame": None,
    "marked_frame": None,
    "diff_frame": None,
    "events": [],
    "max_events": 8,
    "last_update": None,
    # 新增：车辆跟踪状态
    "vehicle_tracks": {},  # 车辆跟踪信息 {track_id: {centroid: (x,y), frames: [], disappeared: 0}}
    "next_track_id": 0,     # 下一个车辆ID
    "tracked_frames": 0,    # 跟踪帧数
    # 新增：违章检测状态
    "wrong_way_vehicles": {},  # 逆行车辆 {track_id: frames_count}
    "illegal_parking_vehicles": {},  # 违法停车车辆 {track_id: start_time}
    "last_process_time": None,  # 上次处理时间
    # 新增：夜间轨迹检测状态
    "trajectory_canvas": None,  # 轨迹累积画布
    "night_trajectory_prev_frame": None,  # 夜间轨迹检测的前一帧
    "trajectory_history": None  # 轨迹历史记录
}
state_lock = threading.Lock()

# 加载YOLO模型
yolo_model = YOLO(YOLO_MODEL_PATH)

# 加载路面塌陷检测模型
try:
    pit_model = YOLO(PIT_MODEL_PATH)
    print(f"路面塌陷检测模型加载成功: {PIT_MODEL_PATH}")
except Exception as e:
    pit_model = None
    print(f"路面塌陷检测模型加载失败: {e}")

# ===================== 新增：numpy 数据类型转换函数（关键修复） =====================
def convert_numpy_to_python(obj):
    """递归将 numpy 数据类型转换为 Python 原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    else:
        return obj

# ===================== 数据库核心功能（新增） =====================
def init_database():
    """初始化数据库：创建事件表"""
    conn = None
    try:
        # 连接数据库（不存在则自动创建）
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        cursor = conn.cursor()
        
        # 创建事件表：包含事件ID、类型、描述、时间、监控源、检测功能、额外信息
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS detection_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            event_desc TEXT NOT NULL,
            event_time TEXT NOT NULL,
            source_id TEXT NOT NULL,
            detect_func TEXT NOT NULL,
            extra_info TEXT,  -- 存储额外信息（JSON格式）
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_sql)
        conn.commit()
        print(f"数据库初始化成功：{DB_FILE}")
        
    except Error as e:
        print(f"数据库初始化失败：{e}")
    finally:
        if conn:
            conn.close()

# ===================== 数据库核心功能（修改 save_event_to_db 函数） =====================
def save_event_to_db(event_data):
    """保存事件到数据库"""
    conn = None
    try:
        with DB_LOCK:  # 线程安全锁
            conn = sqlite3.connect(DB_FILE, check_same_thread=False)
            cursor = conn.cursor()
            
            # 插入事件数据
            insert_sql = """
            INSERT INTO detection_events 
            (event_type, event_desc, event_time, source_id, detect_func, extra_info)
            VALUES (?, ?, ?, ?, ?, ?);
            """
            
            # 关键修复：先转换 numpy 类型为 Python 原生类型，再序列化 JSON
            extra_info = event_data.get("extra_info", {})
            converted_extra_info = convert_numpy_to_python(extra_info)  # 转换 numpy 类型
            extra_info_json = json.dumps(converted_extra_info) if converted_extra_info else "{}"
            
            cursor.execute(insert_sql, (
                event_data["type"],
                event_data.get("desc", event_data["type"]),  # 描述默认使用类型
                event_data["time"],
                event_data["source_id"],
                event_data["detect_func"],
                extra_info_json
            ))
            conn.commit()
            print(f"事件保存到数据库：ID={cursor.lastrowid}, 类型={event_data['type']}")
            
    except Error as e:
        print(f"保存事件到数据库失败：{e}")
    finally:
        if conn:
            conn.close()

def query_events_from_db(limit=50, offset=0, func_filter=None, source_filter=None):
    """从数据库查询事件（支持过滤和分页）"""
    conn = None
    events = []
    try:
        with DB_LOCK:
            conn = sqlite3.connect(DB_FILE, check_same_thread=False)
            conn.row_factory = sqlite3.Row  # 支持按列名访问
            cursor = conn.cursor()
            
            # 构建查询SQL（支持过滤）
            query_sql = "SELECT * FROM detection_events WHERE 1=1"
            params = []
            
            if func_filter:
                query_sql += " AND detect_func = ?"
                params.append(func_filter)
            if source_filter:
                query_sql += " AND source_id = ?"
                params.append(source_filter)
            
            # 按时间倒序（最新的在前）+ 分页
            query_sql += " ORDER BY event_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query_sql, params)
            rows = cursor.fetchall()
            
            # 转换为字典列表
            for row in rows:
                events.append({
                    "id": row["id"],
                    "event_type": row["event_type"],
                    "event_desc": row["event_desc"],
                    "event_time": row["event_time"],
                    "source_id": row["source_id"],
                    "detect_func": row["detect_func"],
                    "extra_info": json.loads(row["extra_info"]) if row["extra_info"] else {},
                    "created_at": row["created_at"]
                })
                
    except Error as e:
        print(f"查询数据库事件失败：{e}")
    finally:
        if conn:
            conn.close()
    return events

def clear_events_from_db():
    """清空所有事件（谨慎使用）"""
    conn = None
    try:
        with DB_LOCK:
            conn = sqlite3.connect(DB_FILE, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM detection_events;")
            cursor.execute("VACUUM;")  # 优化数据库
            conn.commit()
            print("所有事件已从数据库清空")
    except Error as e:
        print(f"清空数据库事件失败：{e}")
    finally:
        if conn:
            conn.close()

# ===================== 中文显示修复 =====================
def get_chinese_font_pillow(font_size=20):
    system = platform.system()
    font_path = None
    
    if system == "Windows":
        font_path = "C:/Windows/Fonts/simhei.ttf"
        if not os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/simsun.ttc"
    elif system == "Linux":
        font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
        if not os.path.exists(font_path):
            font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    elif system == "Darwin":
        font_path = "/System/Library/Fonts/PingFang.ttc"
        if not os.path.exists(font_path):
            font_path = "/Library/Fonts/Songti.ttc"
    
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
        else:
            print(f"未找到中文字体文件，路径：{font_path}，中文可能显示方框")
            return ImageFont.load_default(size=font_size)
    except Exception as e:
        print(f"加载字体失败：{e}，使用默认字体")
        return ImageFont.load_default(size=font_size)

def draw_chinese_text(img, text, org, font_scale=0.8, color=(255, 255, 255), thickness=2):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    base_font_size = 20
    if font_scale == 0.6:
        font_size = 15
    elif font_scale == 0.8:
        font_size = 20
    else:
        font_size = int(base_font_size * font_scale)
    
    font = get_chinese_font_pillow(font_size=font_size)
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color[2], color[1], color[0])
    draw.text(org, text, font=font, fill=color_rgb)
    
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_bgr

# ===================== 核心算法实现 =====================
def generate_event(has_change, func_type, extra_info=None, source_id=None):
    if not has_change:
        return {
            "type": "无事件",
            "desc": "未检测到异常",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "has_change": False,
            "source_id": source_id or global_state["current_source"],
            "detect_func": func_type,
            "extra_info": extra_info
        }
    
    if func_type == DetectFunc.PEDESTRIAN.value:
        person_count = extra_info.get("person_count", 0)
        event_type = f"行人入侵"
        event_desc = f"行人入侵（共{person_count}人）"
    elif func_type == DetectFunc.CHANGE_DETECT.value:
        event_type = "车辆移动" if np.random.random() > 0.5 else "人员入侵"
        event_desc = event_type
    elif func_type == DetectFunc.LANDSLIDE.value:
        change_ratio = extra_info.get("change_ratio", 0)
        event_type = "滑坡异常"
        event_desc = f"滑坡异常（变化比例：{change_ratio:.2%}）"
    elif func_type == DetectFunc.VISIBILITY.value:
        visibility_level = extra_info.get("visibility_level", 0)
        visibility_score = extra_info.get("visibility_score", 0)
        visibility_labels = ['清晰', '轻度雾', '中度雾', '重度雾']
        visibility_text = visibility_labels[visibility_level]
        event_type = "能见度异常"
        event_desc = f"当前能见度：{visibility_text}（等级：{visibility_level}，分数：{visibility_score}）"
    elif func_type == DetectFunc.VEHICLE.value:
        # 检查是否有违章事件
        wrong_way_events = extra_info.get("wrong_way_events", [])
        illegal_parking_events = extra_info.get("illegal_parking_events", [])
        vehicle_count = extra_info.get("vehicle_count", 0)
        
        # 拥堵判断
        congestion_level = "畅通"
        if vehicle_count >= CONGESTION_CONFIG['high_threshold']:
            congestion_level = "高拥堵"
        elif vehicle_count >= CONGESTION_CONFIG['medium_threshold']:
            congestion_level = "中拥堵"
        elif vehicle_count >= CONGESTION_CONFIG['low_threshold']:
            congestion_level = "低拥堵"
        
        # 添加拥堵信息到extra_info
        extra_info['congestion_level'] = congestion_level
        extra_info['congestion_vehicle_count'] = vehicle_count
        
        if wrong_way_events:
            event_type = "车辆逆行"
            event_desc = f"检测到{len(wrong_way_events)}辆逆行车辆（当前{congestion_level}，共{vehicle_count}辆）"
        elif illegal_parking_events:
            event_type = "违法停车"
            event_desc = f"检测到{len(illegal_parking_events)}辆违法停车车辆（当前{congestion_level}，共{vehicle_count}辆）"
        else:
            event_type = "车辆检测"
            event_desc = f"当前{congestion_level}（共{vehicle_count}辆）"
    elif func_type == DetectFunc.PIER.value:
        pier_count = extra_info.get("pier_count", 0)
        event_type = "桥墩检测"
        event_desc = f"桥墩检测（共{pier_count}个）"
    elif func_type == DetectFunc.BRIDGE_COLLAPSE.value:
        change_ratio = extra_info.get("change_ratio", 0)
        event_type = "桥梁坍塌异常"
        event_desc = f"桥梁坍塌异常（变化比例：{change_ratio:.2%}）"
    elif func_type == DetectFunc.PIT.value:
        pit_count = extra_info.get("pit_count", 0)
        event_type = "路面塌陷异常"
        event_desc = f"路面塌陷检测（共{pit_count}处）"
    elif func_type == DetectFunc.NIGHT_TRAJECTORY.value:
        highlight_area = extra_info.get("highlight_area", 0)
        is_interrupted = extra_info.get("is_trajectory_interrupted", False)
        interrupted_ratio = extra_info.get("interrupted_ratio", 0.0)
        visibility_level = extra_info.get("visibility_level", 0)
        visibility_score = extra_info.get("visibility_score", 100)
        visibility_labels = ['清晰', '轻度雾', '中度雾', '重度雾']
        
        # 优先级：能见度异常 > 轨迹中断 > 正常轨迹
        if visibility_level >= 2:  # 中度雾及以上
            event_type = "能见度异常"
            event_desc = f"能见度异常：{visibility_labels[visibility_level]}({visibility_score})，请注意安全驾驶"
        elif is_interrupted:
            event_type = "灯光轨迹中断"
            event_desc = f"灯光轨迹中断（中断比例：{interrupted_ratio:.1%}，当前高亮面积：{highlight_area}）"
        else:
            event_type = "车灯轨迹检测"
            event_desc = f"车灯轨迹检测（当前高亮面积：{highlight_area}）"
    else:
        event_type = f"{func_type.split('（')[0]}异常"
        event_desc = event_type
    
    event_data = {
        "type": event_type,
        "desc": event_desc,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "has_change": True,
        "source_id": source_id or global_state["current_source"],
        "detect_func": func_type,
        "extra_info": extra_info
    }
    
    # 新增：保存事件到数据库（仅保存有异常的事件）
    if has_change:
        save_event_to_db(event_data)
    
    return event_data

# 其他算法函数（process_frame_change_detect、process_frame_yolo_pedestrian等）保持不变
def process_frame_change_detect(prev_frame, curr_frame):
    if prev_frame.shape != curr_frame.shape:
        curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    roi_mask = np.zeros_like(prev_gray)
    roi_np = np.array(ROI_POINTS, np.int32)
    cv2.fillPoly(roi_mask, [roi_np], 255)
    prev_roi = cv2.bitwise_and(prev_gray, prev_gray, mask=roi_mask)
    curr_roi = cv2.bitwise_and(curr_gray, curr_gray, mask=roi_mask)

    frame_diff = cv2.absdiff(prev_roi, curr_roi)
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    change_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA_CHANGE)
    roi_area = cv2.countNonZero(roi_mask) or 1
    change_ratio = change_area / roi_area
    ssim_score = ssim(prev_roi, curr_roi, data_range=255, full=False)
    has_change = change_ratio > CHANGE_THRESHOLD or ssim_score < SSIM_THRESHOLD

    marked_frame = curr_frame.copy()
    overlay = marked_frame.copy()
    roi_color = (0, 0, 255) if has_change else (0, 255, 0)
    cv2.fillPoly(overlay, [roi_np], roi_color)
    cv2.addWeighted(overlay, 0.3, marked_frame, 0.7, 0, marked_frame)
    cv2.polylines(marked_frame, [roi_np], True, (0, 255, 255), 2)

    if has_change:
        for c in contours:
            if cv2.contourArea(c) > MIN_CONTOUR_AREA_CHANGE:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(marked_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    text = f"变化检测 | 变化比例: {change_ratio:.4f} | SSIM: {ssim_score:.4f}"
    marked_frame = draw_chinese_text(marked_frame, text, (10, 30), font_scale=0.8, color=(0, 255, 255), thickness=2)

    diff_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    diff_frame[thresh > 0] = [0, 0, 255]

    return {
        "has_change": has_change,
        "marked_frame": marked_frame,
        "diff_frame": diff_frame,
        "extra_info": {"change_ratio": change_ratio, "ssim_score": ssim_score}
    }

def process_frame_yolo_pedestrian(curr_frame):
    roi_np = np.array(ROI_POINTS, np.int32)
    x1, y1 = roi_np[0]
    x2, y2 = roi_np[2]
    roi_frame = curr_frame[y1:y2, x1:x2]

    results = yolo_model(roi_frame, verbose=False)
    detections = results[0].boxes.data if results[0].boxes is not None else []

    valid_dets = []
    person_count = 0
    for det in detections:
        xyxy = det[:4].cpu().numpy().astype(int)
        conf = det[4].cpu().numpy()
        cls = det[5].cpu().numpy()

        if cls == PERSON_CLASS_ID and conf > YOLO_CONF_THRESH:
            person_count += 1
            orig_xyxy = [xyxy[0]+x1, xyxy[1]+y1, xyxy[2]+x1, xyxy[3]+y1]
            valid_dets.append((orig_xyxy, conf))

    marked_frame = curr_frame.copy()
    overlay = marked_frame.copy()
    cv2.fillPoly(overlay, [roi_np], (0, 255, 255))
    cv2.addWeighted(overlay, 0.2, marked_frame, 0.8, 0, marked_frame)
    cv2.polylines(marked_frame, [roi_np], True, (0, 255, 255), 2)

    for (xyxy, conf) in valid_dets:
        cv2.rectangle(marked_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
        label = f"行人 {conf:.2f}"
        marked_frame = draw_chinese_text(marked_frame, label, (xyxy[0], xyxy[1]-10), font_scale=0.6, color=(0, 0, 255), thickness=2)

    text = f"行人检测 | 置信度阈值: {YOLO_CONF_THRESH} | 检测到: {person_count}人"
    marked_frame = draw_chinese_text(marked_frame, text, (10, 30), font_scale=0.8, color=(0, 255, 255), thickness=2)

    has_change = person_count > 0

    return {
        "has_change": has_change,
        "marked_frame": marked_frame,
        "diff_frame": None,
        "extra_info": {"person_count": person_count, "detections": [{"xyxy": det[0], "confidence": det[1]} for det in valid_dets]}
    }

def process_frame_yolo_pit(curr_frame):
    roi_np = np.array(ROI_POINTS, np.int32)
    x1, y1 = roi_np[0]
    x2, y2 = roi_np[2]
    roi_frame = curr_frame[y1:y2, x1:x2]

    if pit_model is None:
        marked_frame = curr_frame.copy()
        overlay = marked_frame.copy()
        cv2.fillPoly(overlay, [roi_np], (0, 255, 255))
        cv2.addWeighted(overlay, 0.2, marked_frame, 0.8, 0, marked_frame)
        cv2.polylines(marked_frame, [roi_np], True, (0, 255, 255), 2)
        text = "路面塌陷检测模型加载失败！"
        marked_frame = draw_chinese_text(marked_frame, text, (10, 30), font_scale=0.8, color=(0, 0, 255), thickness=2)
        return {
            "has_change": False,
            "marked_frame": marked_frame,
            "diff_frame": None,
            "extra_info": None
        }

    results = pit_model(roi_frame, verbose=False)
    detections = results[0].boxes.data if results[0].boxes is not None else []

    valid_dets = []
    pit_count = 0
    for det in detections:
        xyxy = det[:4].cpu().numpy().astype(int)
        conf = det[4].cpu().numpy()
        cls = det[5].cpu().numpy()

        if cls == PIT_CLASS_ID and conf > PIT_CONF_THRESH:
            pit_count += 1
            orig_xyxy = [xyxy[0]+x1, xyxy[1]+y1, xyxy[2]+x1, xyxy[3]+y1]
            valid_dets.append((orig_xyxy, conf))

    marked_frame = curr_frame.copy()
    overlay = marked_frame.copy()
    cv2.fillPoly(overlay, [roi_np], (0, 255, 255))
    cv2.addWeighted(overlay, 0.2, marked_frame, 0.8, 0, marked_frame)
    cv2.polylines(marked_frame, [roi_np], True, (0, 255, 255), 2)

    for (xyxy, conf) in valid_dets:
        cv2.rectangle(marked_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
        label = f"路面塌陷 {conf:.2f}"
        marked_frame = draw_chinese_text(marked_frame, label, (xyxy[0], xyxy[1]-10), font_scale=0.6, color=(0, 0, 255), thickness=2)

    text = f"路面塌陷检测 | 置信度阈值: {PIT_CONF_THRESH} | 检测到: {pit_count}处"
    marked_frame = draw_chinese_text(marked_frame, text, (10, 30), font_scale=0.8, color=(0, 255, 255), thickness=2)

    has_change = pit_count > 0

    return {
        "has_change": has_change,
        "marked_frame": marked_frame,
        "diff_frame": None,
        "extra_info": {"pit_count": pit_count, "detections": [{"xyxy": det[0], "confidence": det[1]} for det in valid_dets]}
    }

def process_frame_yolo_vehicle(curr_frame):
    roi_np = np.array(ROI_POINTS, np.int32)
    x1, y1 = roi_np[0]
    x2, y2 = roi_np[2]
    roi_frame = curr_frame[y1:y2, x1:x2]

    results = yolo_model(roi_frame, verbose=False)
    detections = results[0].boxes.data if results[0].boxes is not None else []

    valid_dets = []
    vehicle_count = 0
    current_centroids = []
    
    for det in detections:
        xyxy = det[:4].cpu().numpy().astype(int)
        conf = det[4].cpu().numpy()
        cls = det[5].cpu().numpy()

        if cls == VEHICLE_CLASS_ID and conf > VEHICLE_CONF_THRESH:
            vehicle_count += 1
            orig_xyxy = [xyxy[0]+x1, xyxy[1]+y1, xyxy[2]+x1, xyxy[3]+y1]
            # 计算中心点
            cx = int((orig_xyxy[0] + orig_xyxy[2]) / 2)
            cy = int((orig_xyxy[1] + orig_xyxy[3]) / 2)
            valid_dets.append((orig_xyxy, conf, (cx, cy)))
            current_centroids.append((cx, cy, orig_xyxy))

    marked_frame = curr_frame.copy()
    overlay = marked_frame.copy()
    cv2.fillPoly(overlay, [roi_np], (255, 165, 0))
    cv2.addWeighted(overlay, 0.2, marked_frame, 0.8, 0, marked_frame)
    cv2.polylines(marked_frame, [roi_np], True, (255, 165, 0), 2)
    
    

    # 车辆跟踪和违章检测
    wrong_way_events = []
    illegal_parking_events = []
    
    with state_lock:
        # 更新跟踪帧数
        global_state['tracked_frames'] += 1
        
        # 获取当前时间
        current_time = datetime.now()
        if global_state['last_process_time'] is None:
            global_state['last_process_time'] = current_time
        
        # 计算时间差（秒）
        time_diff = (current_time - global_state['last_process_time']).total_seconds()
        global_state['last_process_time'] = current_time
        
        # 车辆跟踪
        vehicle_tracks = global_state['vehicle_tracks']
        next_track_id = global_state['next_track_id']
        
        # 匹配当前检测与现有跟踪
        unmatched_detections = list(range(len(current_centroids)))
        updated_track_ids = []
        
        # 遍历现有跟踪
        for track_id in list(vehicle_tracks.keys()):
            if vehicle_tracks[track_id]['disappeared'] > VEHICLE_TRACKING_CONFIG['max_disappeared']:
                # 删除消失的跟踪
                del vehicle_tracks[track_id]
                if track_id in global_state['wrong_way_vehicles']:
                    del global_state['wrong_way_vehicles'][track_id]
                if track_id in global_state['illegal_parking_vehicles']:
                    del global_state['illegal_parking_vehicles'][track_id]
                continue
            
            min_dist = float('inf')
            matched_det_idx = -1
            
            # 寻找最近的检测
            for det_idx in unmatched_detections:
                centroid = current_centroids[det_idx][0:2]
                dist = np.sqrt((centroid[0] - vehicle_tracks[track_id]['centroid'][0])**2 + 
                              (centroid[1] - vehicle_tracks[track_id]['centroid'][1])**2)
                
                if dist < min_dist and dist < VEHICLE_TRACKING_CONFIG['track_threshold']:
                    min_dist = dist
                    matched_det_idx = det_idx
            
            if matched_det_idx != -1:
                # 更新跟踪
                centroid, xyxy = current_centroids[matched_det_idx][0:2], current_centroids[matched_det_idx][2]
                vehicle_tracks[track_id]['centroid'] = centroid
                vehicle_tracks[track_id]['frames'].append(centroid)
                vehicle_tracks[track_id]['disappeared'] = 0
                vehicle_tracks[track_id]['last_xyxy'] = xyxy
                vehicle_tracks[track_id]['last_seen'] = current_time
                
                updated_track_ids.append(track_id)
                unmatched_detections.remove(matched_det_idx)
            else:
                # 未匹配，增加消失计数
                vehicle_tracks[track_id]['disappeared'] += 1
        
        # 处理未匹配的检测（新车辆）
        for det_idx in unmatched_detections:
            centroid, xyxy = current_centroids[det_idx][0:2], current_centroids[det_idx][2]
            vehicle_tracks[next_track_id] = {
                'centroid': centroid,
                'frames': [centroid],
                'disappeared': 0,
                'last_xyxy': xyxy,
                'last_seen': current_time
            }
            updated_track_ids.append(next_track_id)
            next_track_id += 1
        
        # 更新全局状态
        global_state['vehicle_tracks'] = vehicle_tracks
        global_state['next_track_id'] = next_track_id
        
        # 检测逆行
        for track_id in updated_track_ids:
            if len(vehicle_tracks[track_id]['frames']) < VEHICLE_TRACKING_CONFIG['min_tracking_frames']:
                continue
            
            frames = vehicle_tracks[track_id]['frames']
            centroid = vehicle_tracks[track_id]['centroid']
            
            # 计算移动方向
            if len(frames) >= 2:
                # 计算车辆移动向量：(dx, dy) = 当前帧 - 前一帧
                prev_centroid = frames[-2]
                current_centroid = frames[-1]
                vehicle_vector = (current_centroid[0] - prev_centroid[0], current_centroid[1] - prev_centroid[1])
                
                # 规定的正常方向向量：由下到上（y轴负方向）
                # (0, -1) 表示垂直向上，可根据实际需求调整
                normal_vector = (0, -1)
                #- 水平向右： (1, 0)
                # - 垂直向下： (0, 1)
                # - 水平向左： (-1, 0)
                
                # 计算向量长度（模）
                vehicle_mag = np.sqrt(vehicle_vector[0]**2 + vehicle_vector[1]**2)
                normal_mag = np.sqrt(normal_vector[0]**2 + normal_vector[1]**2)
                
                # 避免除以零
                if vehicle_mag == 0 or normal_mag == 0:
                    continue
                
                # 计算向量点积
                dot_product = vehicle_vector[0] * normal_vector[0] + vehicle_vector[1] * normal_vector[1]
                
                # 计算夹角余弦值
                cos_angle = dot_product / (vehicle_mag * normal_mag)
                
                # 计算夹角（弧度）并转换为角度
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * (180 / np.pi)
                
                # 计算移动距离，确保车辆实际在移动
                move_distance = vehicle_mag
                
                # 逆行判断：
                # 1. 移动距离大于阈值（避免微小移动误判）
                # 2. 夹角大于90度（方向相反）
                if move_distance > WRONG_WAY_CONFIG['direction_threshold'] and angle > 90:
                    if track_id in global_state['wrong_way_vehicles']:
                        global_state['wrong_way_vehicles'][track_id] += 1
                        # 判断是否持续逆行
                        if global_state['wrong_way_vehicles'][track_id] >= WRONG_WAY_CONFIG['min_wrong_way_frames']:
                            wrong_way_events.append({
                                'track_id': track_id,
                                'centroid': centroid,
                                'xyxy': vehicle_tracks[track_id]['last_xyxy']
                            })
                    else:
                        global_state['wrong_way_vehicles'][track_id] = 1
                else:
                    if track_id in global_state['wrong_way_vehicles']:
                        del global_state['wrong_way_vehicles'][track_id]
        
        # 检测违法停车
        for track_id in updated_track_ids:
            if len(vehicle_tracks[track_id]['frames']) < VEHICLE_TRACKING_CONFIG['min_tracking_frames']:
                continue
            
            centroid = vehicle_tracks[track_id]['centroid']
            
            # 计算移动距离
            frames = vehicle_tracks[track_id]['frames']
            if len(frames) >= 2:
                # 计算最近几帧的移动距离（只取最近10帧）
                recent_frames = frames[-10:]  # 只使用最近10帧
                total_dist = 0
                for i in range(1, len(recent_frames)):
                    dist = np.sqrt((recent_frames[i][0] - recent_frames[i-1][0])**2 + (recent_frames[i][1] - recent_frames[i-1][1])**2)
                    total_dist += dist
                
                # 改进：降低移动距离阈值，适应静止车辆的微小波动
                # 从10像素降低到5像素，减少微小波动的影响
                is_parking = total_dist < 5  # 移动距离小于5像素视为停车
                
                if is_parking:
                    # 重置非停车帧计数
                    if 'non_parking_frames' in vehicle_tracks[track_id]:
                        del vehicle_tracks[track_id]['non_parking_frames']
                    
                    if track_id in global_state['illegal_parking_vehicles']:
                        # 计算停车时长
                        parking_duration = (current_time - global_state['illegal_parking_vehicles'][track_id]).total_seconds()
                        if parking_duration >= ILLEGAL_PARKING_CONFIG['parking_threshold']:
                            illegal_parking_events.append({
                                'track_id': track_id,
                                'centroid': centroid,
                                'xyxy': vehicle_tracks[track_id]['last_xyxy'],
                                'duration': parking_duration
                            })
                    else:
                        global_state['illegal_parking_vehicles'][track_id] = current_time
                else:
                    # 改进：增加停车判定的稳定性，不是立即删除记录
                    # 只有当车辆连续多帧不处于停车状态时，才删除记录
                    if track_id in global_state['illegal_parking_vehicles']:
                        # 检查车辆是否有non_parking_frames计数
                        if 'non_parking_frames' not in vehicle_tracks[track_id]:
                            vehicle_tracks[track_id]['non_parking_frames'] = 1
                        else:
                            vehicle_tracks[track_id]['non_parking_frames'] += 1
                        
                        # 如果连续3帧不停车，才删除记录
                        if vehicle_tracks[track_id]['non_parking_frames'] >= 3:
                            del global_state['illegal_parking_vehicles'][track_id]
                            # 重置计数
                            del vehicle_tracks[track_id]['non_parking_frames']
        
        # 清除超时的停车记录
        for track_id in list(global_state['illegal_parking_vehicles'].keys()):
            if track_id not in updated_track_ids:
                del global_state['illegal_parking_vehicles'][track_id]

    # 绘制检测结果和违章信息
    for i, (xyxy, conf, centroid) in enumerate(valid_dets):
        cv2.rectangle(marked_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
        label = f"车辆 {conf:.2f}"
        marked_frame = draw_chinese_text(marked_frame, label, (xyxy[0], xyxy[1]-10), font_scale=0.6, color=(255, 0, 0), thickness=2)
    
    # 绘制违章车辆
    with state_lock:
        for event in wrong_way_events:
            xyxy = event['xyxy']
            cv2.rectangle(marked_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 3)
            label = f"逆行车辆"
            marked_frame = draw_chinese_text(marked_frame, label, (xyxy[0], xyxy[1]-30), font_scale=0.7, color=(0, 0, 255), thickness=2)
        
        for event in illegal_parking_events:
            xyxy = event['xyxy']
            cv2.rectangle(marked_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 3)
            label = f"违法停车 {event['duration']:.1f}s"
            marked_frame = draw_chinese_text(marked_frame, label, (xyxy[0], xyxy[1]-30), font_scale=0.7, color=(0, 255, 0), thickness=2)

    text = f"车辆检测 | 置信度阈值: {VEHICLE_CONF_THRESH} | 检测到: {vehicle_count}辆"
    marked_frame = draw_chinese_text(marked_frame, text, (10, 30), font_scale=0.8, color=(255, 165, 0), thickness=2)
    
    # 添加违章统计
    with state_lock:
        wrong_way_count = len([t for t in global_state['wrong_way_vehicles'].values() if t >= WRONG_WAY_CONFIG['min_wrong_way_frames']])
        parking_count = len(global_state['illegal_parking_vehicles'])
    
    违章_text = f"违章统计 | 逆行: {wrong_way_count}辆 | 违法停车: {parking_count}辆"
    marked_frame = draw_chinese_text(marked_frame, 违章_text, (10, 60), font_scale=0.8, color=(0, 255, 255), thickness=2)

    # 生成事件
    has_change = vehicle_count > 0 or len(wrong_way_events) > 0 or len(illegal_parking_events) > 0
    extra_info = {
        "vehicle_count": vehicle_count,
        "detections": [{"xyxy": det[0], "confidence": det[1]} for det in valid_dets],
        "wrong_way_events": wrong_way_events,
        "illegal_parking_events": illegal_parking_events
    }

    return {
        "has_change": has_change,
        "marked_frame": marked_frame,
        "diff_frame": None,
        "extra_info": extra_info
    }

def process_frame_yolo_pier(curr_frame):
    if pier_onnx_session is None:
        marked_frame = curr_frame.copy()
        roi_np = np.array(ROI_POINTS, np.int32)
        overlay = marked_frame.copy()
        cv2.fillPoly(overlay, [roi_np], (128, 0, 128))
        cv2.addWeighted(overlay, 0.2, marked_frame, 0.8, 0, marked_frame)
        cv2.polylines(marked_frame, [roi_np], True, (128, 0, 128), 2)
        text1 = "桥墩检测（YOLOv8-ONNX）"
        text2 = "模型加载失败！请检查：1. best.onnx路径 2. onnxruntime依赖"
        marked_frame = draw_chinese_text(marked_frame, text1, (10, 30), font_scale=0.8, color=(128, 0, 128), thickness=2)
        marked_frame = draw_chinese_text(marked_frame, text2, (10, 70), font_scale=0.7, color=(0, 0, 255), thickness=2)
        return {
            "has_change": False,
            "marked_frame": marked_frame,
            "diff_frame": None,
            "extra_info": None
        }

    roi_np = np.array(ROI_POINTS, np.int32)
    x1, y1 = roi_np[0]
    x2, y2 = roi_np[2]
    roi_frame = curr_frame[y1:y2, x1:x2].copy()

    original_height, original_width = roi_frame.shape[:2]
    input_height, input_width = PIER_INPUT_SHAPE
    
    scale = min(input_width / original_width, input_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized_image = cv2.resize(roi_frame, (new_width, new_height))
    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    canvas[:new_height, :new_width] = resized_image
    
    input_image = canvas.astype(np.float32) / 255.0
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)

    input_name = pier_onnx_session.get_inputs()[0].name
    output_name = pier_onnx_session.get_outputs()[0].name
    outputs = pier_onnx_session.run([output_name], {input_name: input_image})

    predictions = np.squeeze(outputs[0]).T
    boxes = []
    scores = []
    class_ids = []

    for i in range(predictions.shape[0]):
        classes_scores = predictions[i][4:]
        max_score = np.max(classes_scores)
        if max_score >= PIER_CONF_THRESH:
            class_id = np.argmax(classes_scores)
            x, y, w, h = predictions[i][0], predictions[i][1], predictions[i][2], predictions[i][3]
            left = int((x - w / 2) / scale)
            top = int((y - h / 2) / scale)
            width = int(w / scale)
            height = int(h / scale)
            if left >=0 and top >=0 and left + width <= original_width and top + height <= original_height:
                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, PIER_CONF_THRESH, PIER_IOU_THRESH)
    pier_count = len(indices) if len(indices) > 0 else 0

    marked_frame = curr_frame.copy()
    overlay = marked_frame.copy()
    cv2.fillPoly(overlay, [roi_np], (128, 0, 128))
    cv2.addWeighted(overlay, 0.2, marked_frame, 0.8, 0, marked_frame)
    cv2.polylines(marked_frame, [roi_np], True, (128, 0, 128), 2)

    detections_info = []
    if pier_count > 0:
        for i in indices.flatten():
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            x = box[0] + x1
            y = box[1] + y1
            w = box[2]
            h = box[3]
            cv2.rectangle(marked_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{PIER_CLASS_NAMES[class_id]} {score:.2f}"
            marked_frame = draw_chinese_text(marked_frame, label, (x, y - 10), font_scale=0.6, color=(0, 255, 0), thickness=2)
            detections_info.append({"xyxy": [x, y, x+w, y+h], "confidence": float(score)})

    text = f"桥墩检测 | 置信度阈值: {PIER_CONF_THRESH} | 检测到: {pier_count}个"
    marked_frame = draw_chinese_text(marked_frame, text, (10, 30), font_scale=0.8, color=(128, 0, 128), thickness=2)

    return {
        "has_change": pier_count > 0,
        "marked_frame": marked_frame,
        "diff_frame": None,
        "extra_info": {"pier_count": pier_count, "detections": detections_info}
    }

def process_frame_landslide(prev_frame, curr_frame):
    if prev_frame.shape != curr_frame.shape:
        curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    roi_mask = np.zeros_like(prev_gray)
    roi_np = np.array(ROI_POINTS, np.int32)
    cv2.fillPoly(roi_mask, [roi_np], 255)
    prev_roi = cv2.bitwise_and(prev_gray, prev_gray, mask=roi_mask)
    curr_roi = cv2.bitwise_and(curr_gray, curr_gray, mask=roi_mask)

    frame_diff = cv2.absdiff(prev_roi, curr_roi)
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones(LANDSLIDE_KERNEL, np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    change_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA_LANDSLIDE)
    roi_area = cv2.countNonZero(roi_mask) or 1
    change_ratio = change_area / roi_area
    has_change = change_ratio > LANDSLIDE_THRESHOLD

    marked_frame = curr_frame.copy()
    overlay = marked_frame.copy()
    roi_color = (0, 0, 255) if has_change else (255, 165, 0)
    cv2.fillPoly(overlay, [roi_np], roi_color)
    cv2.addWeighted(overlay, 0.3, marked_frame, 0.7, 0, marked_frame)
    cv2.polylines(marked_frame, [roi_np], True, (255, 165, 0), 2)

    if has_change:
        for c in contours:
            if cv2.contourArea(c) > MIN_CONTOUR_AREA_LANDSLIDE:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(marked_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    text = f"滑坡检测 | 变化比例: {change_ratio:.4f} | 阈值: {LANDSLIDE_THRESHOLD}"
    marked_frame = draw_chinese_text(marked_frame, text, (10, 30), font_scale=0.8, color=(255, 165, 0), thickness=2)

    diff_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    diff_frame[thresh > 0] = [0, 0, 255]

    return {
        "has_change": has_change,
        "marked_frame": marked_frame,
        "diff_frame": diff_frame,
        "extra_info": {"change_ratio": change_ratio, "change_area": change_area}
    }

def process_frame_bridge_collapse(prev_frame, curr_frame):
    if prev_frame.shape != curr_frame.shape:
        curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    roi_mask = np.zeros_like(prev_gray)
    roi_np = np.array(ROI_POINTS, np.int32)
    cv2.fillPoly(roi_mask, [roi_np], 255)
    prev_roi = cv2.bitwise_and(prev_gray, prev_gray, mask=roi_mask)
    curr_roi = cv2.bitwise_and(curr_gray, curr_gray, mask=roi_mask)

    frame_diff = cv2.absdiff(prev_roi, curr_roi)
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones(BRIDGE_COLLAPSE_KERNEL, np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    change_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA_BRIDGE)
    roi_area = cv2.countNonZero(roi_mask) or 1
    change_ratio = change_area / roi_area
    has_change = change_ratio > BRIDGE_COLLAPSE_THRESHOLD

    marked_frame = curr_frame.copy()
    overlay = marked_frame.copy()
    roi_color = (0, 0, 255) if has_change else (255, 0, 0)
    cv2.fillPoly(overlay, [roi_np], roi_color)
    cv2.addWeighted(overlay, 0.3, marked_frame, 0.7, 0, marked_frame)
    cv2.polylines(marked_frame, [roi_np], True, (255, 0, 0), 2)

    if has_change:
        for c in contours:
            if cv2.contourArea(c) > MIN_CONTOUR_AREA_BRIDGE:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(marked_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    text = f"桥梁坍塌检测 | 变化比例: {change_ratio:.4f} | 阈值: {BRIDGE_COLLAPSE_THRESHOLD}"
    marked_frame = draw_chinese_text(marked_frame, text, (10, 30), font_scale=0.8, color=(255, 0, 0), thickness=2)

    diff_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    diff_frame[thresh > 0] = [0, 0, 255]

    return {
        "has_change": has_change,
        "marked_frame": marked_frame,
        "diff_frame": diff_frame,
        "extra_info": {"change_ratio": change_ratio, "change_area": change_area}
    }

def preprocess_frame(frame):
    """帧预处理（灰度转换、高斯模糊）"""
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, NIGHT_TRAJECTORY_CONFIG['gaussian_kernel'], 0)
    return blurred


def detect_moving_lights(current_frame, prev_frame):
    """检测运动车灯区域"""
    if prev_frame is None:
        return np.zeros_like(current_frame), current_frame
    
    # 计算帧间绝对差
    frame_diff = cv2.absdiff(current_frame, prev_frame)
    # 二值化差异图像
    _, diff_thresh = cv2.threshold(frame_diff, NIGHT_TRAJECTORY_CONFIG['motion_threshold'], 255, cv2.THRESH_BINARY)
    # 形态学操作增强车灯区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, NIGHT_TRAJECTORY_CONFIG['morph_open_kernel'])
    diff_processed = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
    return diff_processed, current_frame


def extract_bright_regions(frame, motion_mask):
    """提取高亮区域（车灯）"""
    # 在原始帧上应用亮度阈值
    _, bright_mask = cv2.threshold(frame, NIGHT_TRAJECTORY_CONFIG['brightness_threshold'], 255, cv2.THRESH_BINARY)
    # 结合运动掩码：只保留运动中的高亮区域
    final_mask = cv2.bitwise_and(bright_mask, motion_mask)
    # 进一步清理噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, NIGHT_TRAJECTORY_CONFIG['morph_close_kernel'])
    cleaned_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    return cleaned_mask


def calculate_visibility(frame):
    """
    计算图像能见度等级
    基于对比度、边缘强度和暗通道先验的综合判断
    返回：能见度等级（0:清晰, 1:轻度雾, 2:中度雾, 3:重度雾）和能见度分数
    """
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. 计算对比度（标准差）
    contrast = gray.std()
    
    # 2. 计算边缘强度（Sobel算子）
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
    
    # 3. 计算暗通道（取每个像素RGB通道最小值，然后取局部最小值）
    min_channel = np.min(frame, axis=2)
    dark_channel = cv2.erode(min_channel, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    dark_level = np.mean(dark_channel)
    
    # 4. 计算能见度分数（0-100，分数越高越清晰）
    # 对比度权重0.4，边缘强度权重0.4，暗通道权重0.2
    contrast_score = min(100, max(0, (contrast - 10) / 50 * 100))  # 对比度10-60映射到0-100
    edge_score = min(100, max(0, edge_strength * 10))  # 边缘强度0-10映射到0-100
    dark_score = min(100, max(0, (100 - dark_level) / 80 * 100))  # 暗通道0-80映射到100-0
    
    visibility_score = contrast_score * 0.4 + edge_score * 0.4 + dark_score * 0.2
    
    # 5. 确定能见度等级（使用配置参数）
    if visibility_score > VISIBILITY_CONFIG['clear_threshold']:
        visibility_level = 0  # 清晰
    elif visibility_score > VISIBILITY_CONFIG['light_fog_threshold']:
        visibility_level = 1  # 轻度雾
    elif visibility_score > VISIBILITY_CONFIG['moderate_fog_threshold']:
        visibility_level = 2  # 中度雾
    else:
        visibility_level = 3  # 重度雾
    
    return visibility_level, int(visibility_score)


def process_frame_visibility(curr_frame):
    """能见度检测处理函数"""
    # 计算能见度
    visibility_level, visibility_score = calculate_visibility(curr_frame)
    visibility_labels = ['清晰', '轻度雾', '中度雾', '重度雾']
    visibility_text = visibility_labels[visibility_level]
    
    # 绘制结果
    marked_frame = curr_frame.copy()
    roi_np = np.array(ROI_POINTS, np.int32)
    overlay = marked_frame.copy()
    cv2.fillPoly(overlay, [roi_np], (255, 165, 0))
    cv2.addWeighted(overlay, 0.2, marked_frame, 0.8, 0, marked_frame)
    cv2.polylines(marked_frame, [roi_np], True, (255, 165, 0), 2)
    
    # 绘制文本信息
    text1 = f"能见度检测 | 等级: {visibility_text} | 分数: {visibility_score}"
    marked_frame = draw_chinese_text(marked_frame, text1, (10, 30), font_scale=0.8, color=(255, 165, 0), thickness=2)
    
    # 根据能见度等级设置不同的边框颜色
    border_color = (0, 255, 0)  # 清晰 - 绿色
    if visibility_level == 1:
        border_color = (0, 255, 255)  # 轻度雾 - 黄色
    elif visibility_level == 2:
        border_color = (0, 165, 255)  # 中度雾 - 橙色
    elif visibility_level == 3:
        border_color = (0, 0, 255)  # 重度雾 - 红色
    
    # 绘制能见度等级指示框
    cv2.rectangle(marked_frame, (10, 50), (300, 100), border_color, 2)
    level_text = f"当前状态: {visibility_text}"
    marked_frame = draw_chinese_text(marked_frame, level_text, (20, 70), font_scale=0.7, color=border_color, thickness=2)
    
    # 能见度异常判断（中度雾及以上）
    has_change = visibility_level >= 2
    
    extra_info = {
        "visibility_level": visibility_level,
        "visibility_score": visibility_score,
        "visibility_text": visibility_text
    }
    
    return {
        "has_change": has_change,
        "marked_frame": marked_frame,
        "diff_frame": None,
        "extra_info": extra_info
    }


def process_frame_night_trajectory(curr_frame):
    """夜间轨迹检测处理函数"""
    with state_lock:
        # 获取当前状态
        trajectory_canvas = global_state["trajectory_canvas"]
        night_trajectory_prev_frame = global_state["night_trajectory_prev_frame"]
        trajectory_history = global_state["trajectory_history"]
    
    # 初始化轨迹画布和历史记录（如果尚未初始化）
    if trajectory_canvas is None:
        height, width = curr_frame.shape[:2]
        trajectory_canvas = np.zeros((height, width), dtype=np.uint8)
    
    if trajectory_history is None:
        from collections import deque
        trajectory_history = deque(maxlen=NIGHT_TRAJECTORY_CONFIG['history_length'])
    
    # 预处理帧
    processed_frame = preprocess_frame(curr_frame)
    
    # 检测运动车灯区域
    motion_mask, night_trajectory_prev_frame = detect_moving_lights(processed_frame, night_trajectory_prev_frame)
    
    # 提取高亮区域（车灯）
    light_mask = extract_bright_regions(processed_frame, motion_mask)
    
    # 累积轨迹到画布
    trajectory_canvas = cv2.addWeighted(trajectory_canvas, NIGHT_TRAJECTORY_CONFIG['trajectory_alpha'], light_mask, 1, 0)
    trajectory_canvas = np.clip(trajectory_canvas, 0, 255)
    trajectory_history.append(trajectory_canvas.copy())
    
    # 将轨迹叠加到视频画面
    marked_frame = curr_frame.copy()
    
    # 计算能见度
    visibility_level, visibility_score = calculate_visibility(curr_frame)
    visibility_labels = ['清晰', '轻度雾', '中度雾', '重度雾']
    visibility_text = f"能见度: {visibility_labels[visibility_level]}({visibility_score})"
    
    # 将轨迹画布叠加到结果帧的红色通道
    marked_frame[:, :, 2] = cv2.add(marked_frame[:, :, 2], trajectory_canvas)
    
    # 绘制ROI区域
    roi_np = np.array(ROI_POINTS, np.int32)
    overlay = marked_frame.copy()
    cv2.fillPoly(overlay, [roi_np], (0, 255, 255))
    cv2.addWeighted(overlay, 0.2, marked_frame, 0.8, 0, marked_frame)
    cv2.polylines(marked_frame, [roi_np], True, (0, 255, 255), 2)
    
    # 绘制文本信息
    text = f"夜间轨迹检测 | 轨迹长度: {len(trajectory_history)}/{NIGHT_TRAJECTORY_CONFIG['history_length']}"
    marked_frame = draw_chinese_text(marked_frame, text, (10, 30), font_scale=0.8, color=(0, 255, 255), thickness=2)
    marked_frame = draw_chinese_text(marked_frame, visibility_text, (10, 60), font_scale=0.8, color=(0, 255, 255), thickness=2)
    
    # 计算当前高亮区域面积
    current_area = np.count_nonzero(trajectory_canvas)
    
    # 更新状态
    with state_lock:
        global_state["trajectory_canvas"] = trajectory_canvas
        global_state["night_trajectory_prev_frame"] = night_trajectory_prev_frame
        global_state["trajectory_history"] = trajectory_history
    
    # 新增：轨迹中断检测
    is_trajectory_interrupted = False
    interrupted_area = 0
    interrupted_ratio = 0.0
    
    # 只有当历史记录足够长时才进行中断检测
    if len(trajectory_history) > 5:
        # 获取最近6帧的高亮区域面积
        recent_areas = []
        for i in range(min(6, len(trajectory_history))):
            hist_canvas = trajectory_history[-(i+1)]
            area = np.count_nonzero(hist_canvas)
            recent_areas.append(area)
        
        # 计算前5帧的平均面积
        avg_area = sum(recent_areas[1:]) / len(recent_areas[1:]) if len(recent_areas) > 1 else 0
        
        # 如果平均面积较大（说明之前有明显轨迹），且当前面积骤降超过80%
        if avg_area > 1000 and current_area < avg_area * 0.9:
            is_trajectory_interrupted = True
            interrupted_area = avg_area - current_area
            interrupted_ratio = (avg_area - current_area) / avg_area if avg_area > 0 else 0.0
    
    extra_info = {
        "trajectory_length": len(trajectory_history),
        "highlight_area": current_area,
        "is_trajectory_interrupted": is_trajectory_interrupted,
        "interrupted_area": int(interrupted_area),
        "interrupted_ratio": round(interrupted_ratio, 2),
        "visibility_level": visibility_level,
        "visibility_score": visibility_score
    }
    
    # 能见度异常判断（中度雾及以上）
    is_visibility_abnormal = visibility_level >= 2
    
    # 新增：轨迹检测判断（当高亮区域面积超过阈值时，判断为检测到轨迹）
    has_trajectory = current_area > 500  # 阈值可根据实际情况调整
    
    return {
        "has_change": has_trajectory or is_trajectory_interrupted or is_visibility_abnormal,  # 检测到轨迹、轨迹中断或能见度异常时生成事件
        "marked_frame": marked_frame,
        "diff_frame": cv2.cvtColor(trajectory_canvas, cv2.COLOR_GRAY2BGR),
        "extra_info": extra_info
    }


def process_frame_placeholder(curr_frame, func_name):
    marked_frame = curr_frame.copy()
    roi_np = np.array(ROI_POINTS, np.int32)
    
    overlay = marked_frame.copy()
    cv2.fillPoly(overlay, [roi_np], (0, 255, 255))
    cv2.addWeighted(overlay, 0.2, marked_frame, 0.8, 0, marked_frame)
    cv2.polylines(marked_frame, [roi_np], True, (0, 255, 255), 2)

    text1 = func_name
    text2 = "功能暂未实现，敬请期待！"
    marked_frame = draw_chinese_text(marked_frame, text1, (10, 30), font_scale=0.8, color=(0, 255, 255), thickness=2)
    marked_frame = draw_chinese_text(marked_frame, text2, (10, 70), font_scale=0.8, color=(255, 255, 0), thickness=2)

    return {
        "has_change": False,
        "marked_frame": marked_frame,
        "diff_frame": None,
        "extra_info": None
    }

# ===================== 统一帧处理入口 =====================
def process_frame_unified(prev_frame, curr_frame, current_func):
    if current_func == DetectFunc.PEDESTRIAN.value:
        return process_frame_yolo_pedestrian(curr_frame)
    elif current_func == DetectFunc.CHANGE_DETECT.value:
        return process_frame_change_detect(prev_frame, curr_frame)
    elif current_func == DetectFunc.LANDSLIDE.value:
        return process_frame_landslide(prev_frame, curr_frame)
    elif current_func == DetectFunc.VEHICLE.value:
        return process_frame_yolo_vehicle(curr_frame)
    elif current_func == DetectFunc.PIER.value:
        return process_frame_yolo_pier(curr_frame)
    elif current_func == DetectFunc.BRIDGE_COLLAPSE.value:
        return process_frame_bridge_collapse(prev_frame, curr_frame)
    elif current_func == DetectFunc.NIGHT_TRAJECTORY.value:
        return process_frame_night_trajectory(curr_frame)
    elif current_func == DetectFunc.VISIBILITY.value:
        return process_frame_visibility(curr_frame)
    elif current_func == DetectFunc.PIT.value:
        return process_frame_yolo_pit(curr_frame)
    else:
        return process_frame_change_detect(prev_frame, curr_frame)

# ===================== 视频捕获线程 =====================
def video_capture_thread():
    cap = None
    last_source = None

    while True:
        with state_lock:
            current_source = global_state["current_source"]
            current_func = global_state["current_func"]

        if current_source != last_source:
            if cap is not None:
                cap.release()
            source_info = SOURCES[current_source]
            cap = cv2.VideoCapture(source_info["path"])
            if not cap.isOpened():
                print(f"无法打开监控源: {current_source}")
                time.sleep(0.1)
                continue
            last_source = current_source
            print(f"切换到监控源: {current_source}（类型：{source_info['type']}）")

        ret, frame = cap.read()
        if not ret:
            print(f"监控源 {current_source} 读取失败，尝试重启...")
            cap.release()
            cap = cv2.VideoCapture(SOURCES[current_source]["path"])
            # 重置车辆跟踪状态和夜间轨迹状态，解决视频循环播放时检测失效问题
            with state_lock:
                global_state['vehicle_tracks'] = {}
                global_state['next_track_id'] = 0
                global_state['wrong_way_vehicles'] = {}
                global_state['illegal_parking_vehicles'] = {}
                global_state['tracked_frames'] = 0
                # 重置夜间轨迹检测状态
                global_state['trajectory_canvas'] = None
                global_state['night_trajectory_prev_frame'] = None
                global_state['trajectory_history'] = None
            time.sleep(0.1)
            continue

        with state_lock:
            prev_frame = global_state["prev_frame"]
            global_state["curr_frame"] = frame.copy()

        need_prev_frame = [DetectFunc.CHANGE_DETECT.value, DetectFunc.LANDSLIDE.value, DetectFunc.BRIDGE_COLLAPSE.value]
        if current_func in need_prev_frame and prev_frame is None:
            with state_lock:
                global_state["prev_frame"] = frame.copy()
            time.sleep(0.1)
            continue

        result = process_frame_unified(prev_frame, frame, current_func)

        with state_lock:
            global_state["marked_frame"] = result["marked_frame"]
            global_state["diff_frame"] = result["diff_frame"]
            global_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # 生成事件时传入监控源ID
            event = generate_event(result["has_change"], current_func, result["extra_info"], current_source)
            if result["has_change"]:
                global_state["events"].insert(0, event)
                if len(global_state["events"]) > global_state["max_events"]:
                    global_state["events"] = global_state["events"][:global_state["max_events"]]

        if current_func in need_prev_frame:
            with state_lock:
                global_state["prev_frame"] = frame.copy()

        time.sleep(0.1)

# ===================== Flask API接口 =====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    with state_lock:
        marked_img_url = None
        if global_state["marked_frame"] is not None:
            marked_path = os.path.join(STATIC_DIR, "marked.jpg")
            cv2.imwrite(marked_path, global_state["marked_frame"])
            marked_img_url = f"/static/marked.jpg?t={time.time()}"

        return jsonify({
            "current_source": global_state["current_source"],
            "current_func": global_state["current_func"],
            "sources": SOURCES,
            "marked_img_url": marked_img_url,
            "events": global_state["events"],
            "last_update": global_state["last_update"]
        })

@app.route('/api/switch_source/<source_id>')
def switch_source(source_id):
    if source_id in SOURCES:
        with state_lock:
            global_state["current_source"] = source_id
            global_state["prev_frame"] = None
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "监控源不存在"}), 400

@app.route('/api/process')
def process_data():
    source_id = request.args.get('sourceId', global_state["current_source"])
    func_param = request.args.get('func', DetectFunc.PEDESTRIAN.value)

    if source_id not in SOURCES:
        return jsonify({
            "status": "error",
            "message": "监控源不存在",
            "processed_img_url": None,
            "events": []
        }), 400

    func_map = {
        "pedestrian": DetectFunc.PEDESTRIAN.value,
        "change_detect": DetectFunc.CHANGE_DETECT.value,
        "landslide": DetectFunc.LANDSLIDE.value,
        "vehicle": DetectFunc.VEHICLE.value,
        "pier": DetectFunc.PIER.value,
        "bridge_collapse": DetectFunc.BRIDGE_COLLAPSE.value,
        "night_trajectory": DetectFunc.NIGHT_TRAJECTORY.value,
        "visibility": DetectFunc.VISIBILITY.value,
        "pit": DetectFunc.PIT.value
    }
    current_func = func_map.get(func_param, DetectFunc.PEDESTRIAN.value)

    with state_lock:
        if global_state["current_source"] != source_id:
            global_state["current_source"] = source_id
            global_state["prev_frame"] = None
        if global_state["current_func"] != current_func:
            global_state["current_func"] = current_func
            global_state["prev_frame"] = None

    with state_lock:
        processed_img_url = None
        if global_state["marked_frame"] is not None:
            processed_path = os.path.join(STATIC_DIR, "processed.jpg")
            cv2.imwrite(processed_path, global_state["marked_frame"])
            processed_img_url = f"/static/processed.jpg?t={time.time()}"

        frontend_events = [
            {"type": event["type"], "time": event["time"], "desc": event.get("desc", event["type"])}
            for event in global_state["events"]
        ]

    return jsonify({
        "status": "success",
        "processed_img_url": processed_img_url,
        "events": frontend_events
    })

@app.route('/api/db/events')
def get_db_events():
    """获取数据库中的历史事件（支持过滤和分页）"""
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    func_filter = request.args.get('func')  # 可能为None
    source_filter = request.args.get('source')  # 可能为None
    
    events = query_events_from_db(
        limit=limit, 
        offset=offset, 
        func_filter=func_filter if func_filter else None,  # 无参数则不过滤
        source_filter=source_filter if source_filter else None  # 无参数则不过滤
    )
    
    # 修复：计算真实总条数（之前是简化处理，导致分页错误）
    # 新增：查询总条数
    total_count = query_events_total(func_filter, source_filter)
    
    # 转换为前端友好格式
    frontend_events = []
    for event in events:
        frontend_events.append({
            "id": event["id"],
            "event_type": event["event_type"],
            "event_desc": event["event_desc"],
            "event_time": event["event_time"],
            "source_id": event["source_id"],
            "detect_func": event["detect_func"],
            "extra_info": event["extra_info"]
        })
    
    return jsonify({
        "status": "success",
        "events": frontend_events,
        "total": total_count  # 返回真实总条数
    })

# 新增：查询事件总条数的函数
def query_events_total(func_filter=None, source_filter=None):
    """查询符合条件的事件总条数"""
    conn = None
    total = 0
    try:
        with DB_LOCK:
            conn = sqlite3.connect(DB_FILE, check_same_thread=False)
            cursor = conn.cursor()
            
            query_sql = "SELECT COUNT(*) FROM detection_events WHERE 1=1"
            params = []
            
            if func_filter:
                query_sql += " AND detect_func = ?"
                params.append(func_filter)
            if source_filter:
                query_sql += " AND source_id = ?"
                params.append(source_filter)
            
            cursor.execute(query_sql, params)
            total = cursor.fetchone()[0]
    except Error as e:
        print(f"查询事件总条数失败：{e}")
    finally:
        if conn:
            conn.close()
    return total

@app.route('/api/db/clear')
def clear_db_events():
    """清空数据库中的所有事件"""
    clear_events_from_db()
    return jsonify({
        "status": "success",
        "message": "所有历史事件已清空"
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

# ===================== 启动程序 =====================
if __name__ == "__main__":
    os.makedirs("./model", exist_ok=True)
    
    # 新增：初始化数据库
    init_database()
    
    # 启动视频捕获线程
    capture_thread = threading.Thread(target=video_capture_thread, daemon=True)
    capture_thread.start()
    
    # 启动Flask服务
    app.run(host='0.0.0.0', port=8888, debug=False)