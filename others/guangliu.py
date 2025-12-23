#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光流估计演示脚本
功能：读取视频文件，计算密集光流，并可视化展示
"""

import cv2
import numpy as np
import argparse


def visualize_optical_flow(prev_gray, curr_gray, frame):
    """
    可视化光流
    :param prev_gray: 前一帧灰度图
    :param curr_gray: 当前帧灰度图
    :param frame: 当前彩色帧（用于叠加光流可视化）
    :return: 带有光流可视化的帧
    """
    # 使用Farneback算法计算密集光流
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,  # 输入输出
        0.5, 3, 15, 3, 5, 1.2, 0     # 算法参数
    )
    
    # 计算光流大小和方向
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 创建光流可视化图像
    # 1. 使用颜色编码方向
    hsv = np.zeros_like(frame)
    hsv[..., 0] = angle * 180 / np.pi / 2  # 色相通道表示方向
    hsv[..., 1] = 255                       # 饱和度设为最大值
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # 亮度表示大小
    
    # 将HSV转换为BGR以便显示
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 2. 绘制光流箭头（稀疏绘制，提高可视化效果）
    flow_arrow = frame.copy()
    step = 16  # 每隔step个像素绘制一个箭头
    h, w = curr_gray.shape
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = lines.astype(np.int32)
    
    # 绘制箭头
    for line in lines:
        if magnitude[y[0], x[0]] > 2:  # 只绘制光流较大的箭头
            cv2.arrowedLine(
                flow_arrow, tuple(line[0]), tuple(line[1]),
                (0, 255, 0), 1, cv2.LINE_AA, 0, 0.3
            )
    
    # 3. 计算统计信息
    avg_magnitude = np.mean(magnitude)
    max_magnitude = np.max(magnitude)
    
    # 在图像上显示统计信息
    cv2.putText(flow_color, f'Avg Flow: {avg_magnitude:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(flow_color, f'Max Flow: {max_magnitude:.2f}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 合并显示
    combined = np.hstack((flow_arrow, flow_color))
    
    return combined


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='光流估计演示脚本')
    parser.add_argument('--video', type=str, default='che.mp4', 
                        help='视频文件路径，默认使用light_2.mp4')
    parser.add_argument('--scale', type=float, default=0.8,
                        help='显示缩放比例，默认0.8')
    args = parser.parse_args()
    
    # 打开视频文件
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f'无法打开视频文件: {args.video}')
        return
    
    print(f'正在处理视频: {args.video}')
    print('按键说明:')
    print('  SPACE: 暂停/继续')
    print('  ESC: 退出')
    
    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print('无法读取视频帧')
        cap.release()
        return
    
    # 转换为灰度图
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # 创建窗口
    cv2.namedWindow('Optical Flow Demo', cv2.WINDOW_NORMAL)
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            # 读取下一帧
            ret, curr_frame = cap.read()
            if not ret:
                print('视频播放结束')
                break
            
            frame_count += 1
            
            # 转换为灰度图
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # 计算并可视化光流
            combined = visualize_optical_flow(prev_gray, curr_gray, curr_frame)
            
            # 缩放显示
            height, width = combined.shape[:2]
            new_width = int(width * args.scale)
            new_height = int(height * args.scale)
            resized = cv2.resize(combined, (new_width, new_height))
            
            # 显示结果
            cv2.imshow('Optical Flow Demo', resized)
            
            # 更新前一帧
            prev_gray = curr_gray.copy()
        
        # 处理按键
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC键退出
            break
        elif key == ord(' '):  # 空格键暂停/继续
            paused = not paused
            if paused:
                print('视频已暂停')
            else:
                print('视频继续播放')
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print(f'处理结束，共处理 {frame_count} 帧')


if __name__ == '__main__':
    main()
