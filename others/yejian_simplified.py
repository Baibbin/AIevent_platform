import cv2
import numpy as np
from collections import deque
import os

def process_video_trajectory(video_path, output_path="output_trajectory.mp4", history_length=50):
    """
    处理视频并将检测到的轨迹累计到视频画面中
    
    参数:
        video_path: 输入视频路径
        output_path: 输出视频路径
        history_length: 轨迹历史长度，默认为50
    
    返回:
        None，直接保存处理后的视频
    """
    # 初始化视频捕获
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 初始化轨迹画布
    trajectory_canvas = np.zeros((height, width), dtype=np.uint8)
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 轨迹历史记录
    trajectory_history = deque(maxlen=history_length)
    prev_frame = None
    
    def preprocess_frame(frame):
        """帧预处理（灰度转换、高斯模糊）"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred
    
    def detect_moving_lights(current_frame, prev_frame):
        """检测运动车灯区域"""
        if prev_frame is None:
            return np.zeros_like(current_frame), current_frame
        
        # 计算帧间绝对差
        frame_diff = cv2.absdiff(current_frame, prev_frame)
        # 二值化差异图像
        _, diff_thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        # 形态学操作增强车灯区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        diff_processed = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
        return diff_processed, current_frame
    
    def extract_bright_regions(frame, motion_mask):
        """提取高亮区域（车灯）"""
        # 在原始帧上应用亮度阈值
        _, bright_mask = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
        # 结合运动掩码：只保留运动中的高亮区域
        final_mask = cv2.bitwise_and(bright_mask, motion_mask)
        # 进一步清理噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        return cleaned_mask
    
    # 处理视频帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 预处理帧
        processed_frame = preprocess_frame(frame)
        
        # 检测运动车灯区域
        motion_mask, prev_frame = detect_moving_lights(processed_frame, prev_frame)
        
        # 提取高亮区域（车灯）
        light_mask = extract_bright_regions(processed_frame, motion_mask)
        
        # 累积轨迹到画布
        trajectory_canvas = cv2.addWeighted(trajectory_canvas, 0.95, light_mask, 1, 0)
        trajectory_canvas = np.clip(trajectory_canvas, 0, 255)
        trajectory_history.append(trajectory_canvas.copy())
        
        # 将轨迹叠加到视频画面
        result_frame = frame.copy()
        # 将轨迹画布叠加到结果帧的红色通道
        result_frame[:, :, 2] = cv2.add(result_frame[:, :, 2], trajectory_canvas)
        
        # 写入输出视频
        out.write(result_frame)
    
    # 释放资源
    cap.release()
    out.release()
    print(f"处理完成，轨迹视频已保存至: {output_path}")

if __name__ == "__main__":
    # 示例用法
    video_path = "light_2.mp4"  # 替换为你的视频路径
    process_video_trajectory(video_path, "output_trajectory.mp4")