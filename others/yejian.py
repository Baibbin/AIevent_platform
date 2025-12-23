# import cv2
# import numpy as np
# from collections import defaultdict

# # 全局变量：用于多边形ROI绘制
# drawing = False
# points = []
# roi_selected = False

# def draw_roi(event, x, y, flags, param):
#     """鼠标回调函数：用于手动绘制多边形ROI"""
#     global drawing, points, roi_selected
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         points.append((x, y))
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             # 实时绘制临时线条
#             temp_frame = param.copy()
#             for i in range(1, len(points)):
#                 cv2.line(temp_frame, points[i-1], points[i], (0, 255, 0), 2)
#             if len(points) > 0:
#                 cv2.line(temp_frame, points[-1], (x, y), (0, 255, 0), 2)  # 最后一点到鼠标当前位置
#             cv2.imshow("Select ROI (Right-click to finish)", temp_frame)
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         # 右键点击结束绘制，闭合多边形
#         drawing = False
#         if len(points) >= 3:  # 至少3个点构成多边形
#             points.append(points[0])  # 闭合多边形
#             roi_selected = True
#             print("ROI选择完成，开始处理视频...")


# def detect_white_yellow_lights(video_path, output_path="light_tracks.png"):
#     global points, roi_selected
#     # 1. 打开视频文件
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("无法打开视频文件")
#         return

#     # 2. 读取第一帧用于手动选择ROI
#     ret, first_frame = cap.read()
#     if not ret:
#         print("无法读取视频帧")
#         return
#     temp_frame = first_frame.copy()
#     cv2.namedWindow("Select ROI (Right-click to finish)")
#     cv2.setMouseCallback("Select ROI (Right-click to finish)", draw_roi, temp_frame)
#     cv2.imshow("Select ROI (Right-click to finish)", first_frame)
#     cv2.waitKey(0)
#     cv2.destroyWindow("Select ROI (Right-click to finish)")

#     if not roi_selected or len(points) < 3:
#         print("未选择有效的ROI区域，程序退出")
#         return

#     # 3. 定义白黄色灯光的HSV阈值（可根据实际场景调整）
#     lower_white = np.array([0, 0, 200])
#     upper_white = np.array([180, 30, 255])
#     lower_yellow = np.array([20, 100, 100])
#     upper_yellow = np.array([30, 255, 255])

#     # 4. 轨迹记录与跟踪参数
#     tracks = defaultdict(list)
#     next_id = 0
#     max_distance = 30  # 帧间匹配最大距离（像素）
#     min_track_length = 5  # 最小轨迹长度（过滤噪声）
#     last_detections = []  # 上一帧的灯光中心坐标和ID

#     # 5. 重新读取视频（从第一帧开始处理）
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     cv2.namedWindow("Light Detection & Tracking")

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break  # 视频结束

#         # 创建ROI掩码（只处理多边形内的区域）
#         roi_mask = np.zeros(frame.shape[:2], np.uint8)
#         pts = np.array(points[:-1], np.int32)  # 去除闭合的最后一点
#         cv2.fillPoly(roi_mask, [pts], 255)  # 多边形内填充为白色（255）

#         # 转换为HSV并提取白黄色区域（仅在ROI内）
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         mask_white = cv2.inRange(hsv, lower_white, upper_white)
#         mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
#         mask_light = cv2.bitwise_or(mask_white, mask_yellow)
#         mask_light = cv2.bitwise_and(mask_light, roi_mask)  # 仅保留ROI内的灯光

#         # 形态学去噪
#         kernel = np.ones((3, 3), np.uint8)
#         mask_light = cv2.morphologyEx(mask_light, cv2.MORPH_OPEN, kernel)
#         mask_light = cv2.morphologyEx(mask_light, cv2.MORPH_CLOSE, kernel)

#         # 查找连通域（灯光候选区域）
#         contours, _ = cv2.findContours(mask_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         current_centers = []

#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if 1000 < area < 10000:  # 筛选灯光尺寸
#                 M = cv2.moments(cnt)
#                 if M["m00"] == 0:
#                     continue
#                 cX = int(M["m10"] / M["m00"])
#                 cY = int(M["m01"] / M["m00"])
#                 current_centers.append((cX, cY))
#                 # 在当前帧标记灯光中心（黄色点）
#                 cv2.circle(frame, (cX, cY), 3, (0, 255, 255), -1)

#         # 6. 帧间匹配与轨迹更新
#         current_ids = []
#         if last_detections:
#             for (cX, cY) in current_centers:
#                 min_dist = float('inf')
#                 matched_id = -1
#                 for (last_x, last_y, last_id) in last_detections:
#                     dist = np.sqrt((cX - last_x)**2 + (cY - last_y)** 2)
#                     if dist < min_dist and dist < max_distance:
#                         min_dist = dist
#                         matched_id = last_id
#                 if matched_id != -1:
#                     tracks[matched_id].append((cX, cY))
#                     current_ids.append((cX, cY, matched_id))
#                 else:
#                     tracks[next_id].append((cX, cY))
#                     current_ids.append((cX, cY, next_id))
#                     next_id += 1
#         else:
#             for (cX, cY) in current_centers:
#                 tracks[next_id].append((cX, cY))
#                 current_ids.append((cX, cY, next_id))
#                 next_id += 1

#         last_detections = current_ids

#         # 7. 实时绘制已生成的轨迹
#         for track_id, points in tracks.items():
#             if len(points) >= min_track_length:
#                 for i in range(1, len(points)):
#                     cv2.line(frame, points[i-1], points[i], (0, 255, 0), 2)  # 绿色轨迹
#                 cv2.circle(frame, points[0], 5, (0, 0, 255), -1)  # 起点红色
#                 cv2.circle(frame, points[-1], 5, (255, 0, 0), -1)  # 终点蓝色

#         # 绘制ROI边界（红色）
#         for i in range(1, len(points)):
#             cv2.line(frame, points[i-1], points[i], (0, 0, 255), 2)

#         # 显示实时处理结果
#         cv2.imshow("Light Detection & Tracking", frame)
#         if cv2.waitKey(25) & 0xFF == ord('q'):  # 按q退出播放
#             break

#     # 保存最终轨迹图像（最后一帧）
#     if 'frame' in locals():
#         cv2.imwrite(output_path, frame)
#         print(f"最终轨迹已保存至 {output_path}")

#     # 释放资源
#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     video_path = "light_2.mp4"  # 替换为你的视频路径
#     detect_white_yellow_lights(video_path, "light_tracks.png")


# import cv2
# import numpy as np

# def detect_lights_in_image(image_path, output_result="detected_lights.png", output_contours="light_contours.png"):
#     """对单张图片进行白黄色灯光检测，并可视化每一步处理结果"""
#     # 1. 读取图片
#     image = cv2.imread(image_path)
#     if image is None:
#         print("无法读取图片，请检查路径")
#         return
#     original = image.copy()  # 保存原图

#     # 显示步骤1：原图
#     cv2.imshow("1. Original Image", original)

#     # 2. 颜色空间转换（BGR -> HSV）
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     # 显示步骤2：HSV色彩空间图像（为了可视化，转换回BGR显示）
#     hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # HSV转BGR用于显示
#     cv2.imshow("2. HSV Color Space", hsv_vis)

#     # 3. 白黄色灯光阈值筛选
#     # 调整后的阈值
#     lower_white = np.array([0, 0, 150])
#     upper_white = np.array([180, 50, 255])
#     lower_yellow = np.array([15, 80, 120])
#     upper_yellow = np.array([35, 255, 255])
    
#     # 提取白色区域
#     mask_white = cv2.inRange(hsv, lower_white, upper_white)
#     # 提取黄色区域
#     mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
#     # 合并白黄色掩码
#     mask_light = cv2.bitwise_or(mask_white, mask_yellow)

#     # 显示步骤3：各颜色掩码
#     cv2.imshow("3.1 White Mask", mask_white)  # 白色掩码（单通道）
#     cv2.imshow("3.2 Yellow Mask", mask_yellow)  # 黄色掩码（单通道）
#     cv2.imshow("3.3 Combined Mask (White+Yellow)", mask_light)  # 合并掩码

#     # 4. 形态学操作去噪（可选，这里保留注释便于对比）
#     # kernel = np.ones((2, 2), np.uint8)
#     # mask_light = cv2.morphologyEx(mask_light, cv2.MORPH_OPEN, kernel)
#     # mask_light = cv2.morphologyEx(mask_light, cv2.MORPH_CLOSE, kernel)
#     # cv2.imshow("4. Denoised Mask (Morphology)", mask_light)  # 去噪后的掩码

#     # 5. 掩码应用：只保留原图中掩码区域的内容
#     masked_image = cv2.bitwise_and(original, original, mask=mask_light)
#     cv2.imshow("5. Masked Image (Only Light Regions)", masked_image)

#     # 6. 连通域分析与筛选
#     contours, _ = cv2.findContours(mask_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     light_centers = []
#     contour_img = np.zeros_like(original)  # 用于绘制连通域的空白图

#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if 2000 < area < 10000:  # 尺寸筛选
#             M = cv2.moments(cnt)
#             if M["m00"] == 0:
#                 continue
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#             light_centers.append((cX, cY))
#             cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), -1)  # 绿色填充连通域

#     # 显示步骤6：筛选后的连通域
#     cv2.imshow("6. Filtered Contours (Green)", contour_img)

#     # 7. 最终结果：原图标记灯光中心
#     result = original.copy()
#     for (x, y) in light_centers:
#         cv2.circle(result, (x, y), 5, (0, 255, 255), -1)  # 黄色圆点标记中心
#     cv2.imshow("7. Final Result (Centers Marked)", result)

#     # 保存结果
#     cv2.imwrite(output_result, result)
#     cv2.imwrite(output_contours, contour_img)
#     print(f"带中心标记的结果已保存至 {output_result}")
#     print(f"连通域可视化结果已保存至 {output_contours}")

#     # 等待用户关闭所有窗口
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     image_path = "test_img.jpg"  # 替换为你的图片路径
#     detect_lights_in_image(image_path)




import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os

class VehicleTrajectoryDetector:
    def __init__(self, video_path, history_length=50):
        self.cap = cv2.VideoCapture(video_path)
        self.trajectory_canvas = None  # 轨迹累积画布
        self.prev_frame = None  # 前一帧图像
        self.trajectory_history = deque(maxlen=history_length)  # 轨迹点历史记录
        self.frame_count = 0
        
        # 获取视频基本信息
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # 初始化轨迹画布
        self.reset_trajectory_canvas()
        
        # 创建输出目录
        self.output_dir = "output_videos_6"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def reset_trajectory_canvas(self):
        """创建新的轨迹画布"""
        self.trajectory_canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        
    def preprocess_frame(self, frame):
        """预处理帧图像"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def detect_moving_lights(self, current_frame):
        """检测运动车灯区域"""
        if self.prev_frame is None:
            self.prev_frame = current_frame
            return np.zeros_like(current_frame)
        
        # 计算帧间绝对差
        frame_diff = cv2.absdiff(current_frame, self.prev_frame)
        
        # 二值化差异图像
        _, diff_thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # 形态学操作增强车灯区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        diff_processed = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
        
        # 更新前一帧
        self.prev_frame = current_frame
        
        return diff_processed
    
    def extract_bright_regions(self, frame, motion_mask):
        """提取高亮区域（车灯）"""
        # 在原始帧上应用亮度阈值
        _, bright_mask = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
        
        # 结合运动掩码：只保留运动中的高亮区域
        final_mask = cv2.bitwise_and(bright_mask, motion_mask)
        
        # 进一步清理噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        return cleaned_mask
    
    def accumulate_trajectory(self, light_mask):
        """累积车灯轨迹"""
        # 将当前检测到的车灯区域添加到轨迹画布
        # self.trajectory_canvas = cv2.add(self.trajectory_canvas, light_mask)
        self.trajectory_canvas = cv2.addWeighted(self.trajectory_canvas, 0.95, light_mask, 1, 0)
        
        # 限制最大值防止溢出
        self.trajectory_canvas = np.clip(self.trajectory_canvas, 0, 255)
        
        # 保存当前帧的轨迹点
        self.trajectory_history.append(self.trajectory_canvas.copy())
        
    def analyze_trajectories(self):
        """分析累积的轨迹"""
        # 使用连通组件分析找到轨迹线
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.trajectory_canvas, connectivity=8
        )
        
        vehicles = []
        for i in range(1, num_labels):  # 跳过背景
            # 提取轨迹特征
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            centroid = centroids[i]
            
            # 过滤掉太小的区域（噪声）
            if area < 100:
                continue
                
            # 计算轨迹方向
            direction = self.calculate_trajectory_direction(i, labels)
            
            vehicles.append({
                'centroid': centroid,
                'area': area,
                'direction': direction,
                'bounding_box': (
                    stats[i, cv2.CC_STAT_LEFT],
                    stats[i, cv2.CC_STAT_TOP],
                    width,
                    height
                )
            })
        
        return vehicles
    
    def calculate_trajectory_direction(self, label, labels_matrix):
        """计算轨迹方向"""
        # 获取该轨迹的所有点
        points = np.argwhere(labels_matrix == label)
        
        if len(points) < 2:
            return 0  # 无法确定方向
        
        # 使用主成分分析(PCA)确定主要方向
        mean = np.mean(points, axis=0)
        points_centered = points - mean
        cov = np.cov(points_centered, rowvar=False)
        
        # 计算特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # 最大特征值对应的特征向量即为主要方向
        primary_vector = eigenvectors[:, np.argmax(eigenvalues)]
        
        # 计算角度（弧度转角度）
        angle = np.degrees(np.arctan2(primary_vector[1], primary_vector[0]))
        
        return angle
    
    def visualize_results(self, current_frame, light_mask, vehicles):
        """可视化处理结果"""
        # 创建可视化图像
        vis_frame = current_frame.copy()
        
        # 叠加轨迹画布（红色）
        vis_frame[:, :, 2] = cv2.add(vis_frame[:, :, 2], self.trajectory_canvas)
        
        # 标记检测到的车辆轨迹
        for vehicle in vehicles:
            x, y, w, h = vehicle['bounding_box']
            cx, cy = vehicle['centroid']
            
            # 绘制边界框
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 绘制方向箭头
            angle_rad = np.radians(vehicle['direction'])
            end_x = int(cx + 50 * np.cos(angle_rad))
            end_y = int(cy + 50 * np.sin(angle_rad))
            cv2.arrowedLine(vis_frame, (int(cx), int(cy)), (end_x, end_y), (0, 255, 255), 2)
            
            # 显示方向角度
            cv2.putText(vis_frame, f"{vehicle['direction']:.1f}°", 
                        (int(cx)+10, int(cy)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 显示当前帧的车灯检测结果（蓝色通道）
        vis_frame[:, :, 0] = cv2.add(vis_frame[:, :, 0], light_mask)
        
        return vis_frame
    
    def process_video(self, display=True):
        """处理整个视频序列"""
        # 创建视频写入器
        detection_writer = None
        canvas_writer = None
        
        if display:
            # 创建轨迹检测视频写入器
            detection_path = os.path.join(self.output_dir, "trajectory_detection.mp4")
            detection_writer = cv2.VideoWriter(
                detection_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                self.fps, 
                (self.width, self.height)
            )
            
            # 创建轨迹画布视频写入器
            canvas_path = os.path.join(self.output_dir, "trajectory_canvas.mp4")
            canvas_writer = cv2.VideoWriter(
                canvas_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                self.fps, 
                (self.width, self.height)
            )
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # 预处理帧
            processed_frame = self.preprocess_frame(frame)
            
            # 检测运动车灯区域
            motion_mask = self.detect_moving_lights(processed_frame)
            
            # 提取高亮区域（车灯）
            light_mask = self.extract_bright_regions(processed_frame, motion_mask)
            
            # 累积轨迹
            self.accumulate_trajectory(light_mask)
            
            # 每5帧分析一次轨迹
            if self.frame_count % 1 == 0:
                vehicles = self.analyze_trajectories()
                
                # 可视化结果
                vis_frame = self.visualize_results(frame, light_mask, vehicles)
                
                if display:
                    # 显示窗口
                    cv2.imshow('Trajectory Detection', vis_frame)
                    cv2.imshow('Trajectory Canvas', self.trajectory_canvas)
                    
                    # 写入视频
                    detection_writer.write(vis_frame)
                    
                    # 将轨迹画布转换为三通道以便写入视频
                    canvas_color = cv2.cvtColor(self.trajectory_canvas, cv2.COLOR_GRAY2BGR)
                    canvas_writer.write(canvas_color)
                    
                    # 在第100帧暂停
                    if self.frame_count == 100:
                        print("已到达第100帧，按任意键继续...")
                        cv2.waitKey(0)  # 无限期等待按键
                        print("继续处理视频...")
                    
                    # 检查是否按下q键退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            self.frame_count += 1
            
        self.cap.release()
        
        # 释放视频写入器
        if detection_writer:
            detection_writer.release()
            print(f"轨迹检测视频已保存至: {os.path.join(self.output_dir, 'trajectory_detection.mp4')}")
        if canvas_writer:
            canvas_writer.release()
            print(f"轨迹画布视频已保存至: {os.path.join(self.output_dir, 'trajectory_canvas.mp4')}")
        
        cv2.destroyAllWindows()
        
        return self.trajectory_canvas, self.trajectory_history

    def plot_trajectories(self):
        """绘制累积的轨迹"""
        plt.figure(figsize=(12, 8))
        
        # 创建RGB图像用于显示
        trajectory_rgb = cv2.cvtColor(self.trajectory_canvas, cv2.COLOR_GRAY2RGB)
        
        # 标记检测到的车辆轨迹
        vehicles = self.analyze_trajectories()
        for vehicle in vehicles:
            x, y, w, h = vehicle['bounding_box']
            cx, cy = vehicle['centroid']
            
            # 绘制边界框
            cv2.rectangle(trajectory_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 绘制方向箭头
            angle_rad = np.radians(vehicle['direction'])
            end_x = int(cx + 50 * np.cos(angle_rad))
            end_y = int(cy + 50 * np.sin(angle_rad))
            cv2.arrowedLine(trajectory_rgb, (int(cx), int(cy)), (end_x, end_y), (255, 255, 0), 2)
            
            # 显示方向角度
            cv2.putText(trajectory_rgb, f"{vehicle['direction']:.3f}°", 
                        (int(cx)+10, int(cy)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        plt.imshow(trajectory_rgb)
        plt.title('Accumulated Vehicle Trajectories')
        plt.axis('off')
        plt.show()

# 使用示例
if __name__ == "__main__":
    detector = VehicleTrajectoryDetector("light_2.mp4")
    trajectory_canvas, history = detector.process_video(display=True)
    detector.plot_trajectories()