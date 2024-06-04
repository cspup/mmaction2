import os

import cv2
import numpy as np

# 读取视频文件  
cap = cv2.VideoCapture('E:/Dataset/RS_v1_50_EgoGesture/videos/Subject01/Scene1/Color/rgb2.avi')

# 检查视频是否成功打开  
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 初始化前一帧  
prev_frame = None

# 设置阈值，用于确定像素变化是否显著  
threshold = 25


# 创建保存差分图像的文件夹
diff_folder = 'E:/test/frame_diffs'
if not os.path.exists(diff_folder):
    os.makedirs(diff_folder)

# 原始图像
origin_frame_folder = 'E:/test/origin_frame'
if not os.path.exists(origin_frame_folder):
    os.makedirs(origin_frame_folder)

frame_count = 0

while True:
    # 读取当前帧  
    ret, frame = cap.read()

    # 如果读取帧失败（例如，已经到达视频末尾）  
    if not ret:
        break

        # 将帧转换为灰度图像，因为我们对颜色不感兴趣，只关心亮度变化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 如果是第一帧或者prev_frame还未初始化，则直接跳过  
    if prev_frame is None:
        prev_frame = gray
        continue

        # 计算当前帧和前一帧之间的差分
    diff = cv2.absdiff(prev_frame, gray)

    # 应用阈值来二值化差分图像  
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # 显示当前帧和阈值化后的差分图像  
    cv2.imshow('Frame', frame)
    cv2.imshow('Difference', thresholded)

    # 保存差分图像
    cv2.imwrite(os.path.join(diff_folder, f'frame_diff_{frame_count:04d}.png'), thresholded)
    # 保存原始图像
    cv2.imwrite(os.path.join(origin_frame_folder, f'frame_diff_{frame_count:04d}.png'), frame)
    frame_count += 1

    # 更新前一帧为当前帧  
    prev_frame = gray

    # 按'q'键退出循环  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 释放视频文件和窗口
cap.release()
cv2.destroyAllWindows()