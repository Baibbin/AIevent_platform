# ========== 先解决OpenMP冲突 ==========
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ========== 核心库 ==========
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ===================== 核心配置 =====================
VIDEO_INPUT_PATH = "light_2.mp4"   # 你的视频路径
VIDEO_OUTPUT_PATH = "output_reverse_light2.mp4"# 输出带逆行标注的视频
VEHICLE_CLASS_ID = 2                    # 只检测小车（可加5=巴士/7=货车）
CONF_THRESH = 0.5                       # 检测置信度阈值
TRACK_PERSIST = True                    # 保持跟踪ID连续
TRACK_HISTORY_LEN = 30                  # 轨迹长度（帧数）

# ---------------- 逆行判断核心配置 ----------------
# 1. 定义道路合法行驶方向（关键！需根据你的视频场景修改）
# 可选方向参考：
# - 从左到右：np.array([1, 0])   （x轴正方向）
# - 从上到下：np.array([0, 1])   （y轴正方向，高速常见）
# - 从右到左：np.array([-1, 0])  （x轴负方向）
# - 从下到上：np.array([0, -1])  （y轴负方向）
ROAD_FORWARD_DIR = np.array([0, 1])  # 示例：道路正向为「从上到下」

# 2. 逆行判定阈值：位移向量与正向夹角 > 90° 则为逆行
REVERSE_ANGLE_THRESH = 90  # 角度阈值（°）

# ===================== 初始化 =====================
model = YOLO("./model/yolov8n.pt")
track_history = defaultdict(lambda: [])  # 存储轨迹点 (x,y)
reverse_vehicle_ids = set()             # 记录逆行的车辆ID

# 视频读取/写入初始化
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (width, height))

# ===================== 工具函数：判断车辆是否逆行 =====================
def is_reverse_driving(track_points, road_forward, angle_thresh):
    """
    判断车辆是否逆行
    :param track_points: 车辆轨迹点列表 [(x1,y1), (x2,y2), ...]
    :param road_forward: 道路正向向量（如np.array([0,1])）
    :param angle_thresh: 逆行角度阈值（°）
    :return: (是否逆行, 实际夹角°)
    """
    # 至少需要2个轨迹点才能计算方向
    if len(track_points) < 2:
        return False, 0.0
    
    # 取最近两帧的轨迹点，计算位移向量（像素单位）
    prev_point = np.array(track_points[-2])  # 上一帧坐标
    curr_point = np.array(track_points[-1])  # 当前帧坐标
    move_vector = curr_point - prev_point    # 位移向量（x差, y差）
    
    # 过滤静止车辆（位移太小，无方向）
    if np.linalg.norm(move_vector) < 5:  # 位移<5像素视为静止
        return False, 0.0
    
    # 归一化向量（消除位移距离影响，只保留方向）
    move_vector_norm = move_vector / np.linalg.norm(move_vector)
    road_forward_norm = road_forward / np.linalg.norm(road_forward)
    
    # 计算向量夹角（弧度转角度）
    dot_product = np.clip(np.dot(move_vector_norm, road_forward_norm), -1.0, 1.0)
    angle = np.arccos(dot_product) * 180 / np.pi
    
    # 夹角>阈值 → 逆行
    is_reverse = angle > angle_thresh
    return is_reverse, angle

# ===================== 逐帧处理 =====================
print("开始处理视频（含逆行判断）...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. YOLO跟踪车辆
    results = model.track(
        source=frame,
        persist=TRACK_PERSIST,
        classes=VEHICLE_CLASS_ID,
        conf=CONF_THRESH,
        verbose=False
    )

    # 2. 提取跟踪结果并判断逆行
    if results[0].boxes is not None and results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            # 计算车辆中心坐标，更新轨迹
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            track_history[track_id].append((center_x, center_y))
            if len(track_history[track_id]) > TRACK_HISTORY_LEN:
                track_history[track_id].pop(0)
            
            # 3. 判断当前车辆是否逆行
            is_reverse, angle = is_reverse_driving(
                track_history[track_id],
                ROAD_FORWARD_DIR,
                REVERSE_ANGLE_THRESH
            )
            
            # 4. 可视化标注（核心：区分逆行/正常车辆）
            if is_reverse:
                reverse_vehicle_ids.add(track_id)
                # 逆行车辆：红色框 + 逆行文字 + 角度
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 粗红框
                label = f"ID:{track_id} 逆行({angle:.0f}°)"
                cv2.putText(frame, label, (x1, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # 正常车辆：蓝色框 + 正常文字
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"ID:{track_id} 正常({angle:.0f}°)"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 5. 绘制轨迹（逆行轨迹红色，正常轨迹绿色）
            points = np.array(track_history[track_id], np.int32)
            track_color = (0, 0, 255) if is_reverse else (0, 255, 0)
            cv2.polylines(frame, [points], isClosed=False, color=track_color, thickness=2)
    
    # 6. 绘制全局统计（逆行车辆数）
    reverse_count = len(reverse_vehicle_ids)
    stat_text = f"逆行车辆数：{reverse_count} | 道路正向：{ROAD_FORWARD_DIR}"
    cv2.putText(frame, stat_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # 7. 保存+显示
    video_writer.write(frame)
    cv2.imshow("Vehicle Reverse Detection (Press 'q' to exit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===================== 释放资源 =====================
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"处理完成！输出视频：{VIDEO_OUTPUT_PATH}")
print(f"检测到逆行车辆ID：{reverse_vehicle_ids}")