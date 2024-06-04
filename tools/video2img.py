import cv2
# 打开视频文件
cap = cv2.VideoCapture("E:/Download/lmnet_cam (1).mp4")
# 设置帧率和提取范围
fps = cap.get(cv2.CAP_PROP_FPS)
start_time = 0
end_time = 10
# 开始提取帧
frame_count = 0
# root_path = 
while cap.isOpened():
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break
    # 判断当前帧是否在提取范围内
    current_time = frame_count / fps
    if start_time <= current_time <= end_time:
        # 保存当前帧为图片
        cv2.imwrite('E:/test/frame_%d.jpg'%frame_count, frame)
        print('frame_%d.jpg'%frame_count)
        frame_count += 1
# 关闭视频文件
cap.release()