import os

import cv2
import numpy as np


def draw_flow(im, flow, step=10):  # 画出光流方向线
    h, w = im.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def OpticalFlow():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('E:/Dataset/RS_v1_50_EgoGesture/videos/Subject01/Scene1/Color/rgb2.avi')

    HS_params = dict(pyr_scale=0.5,  # Horn_Schunck参数
                     levels=3,
                     winsize=15,
                     iterations=3,
                     poly_n=5,
                     poly_sigma=1.1,
                     flags=0)

    _, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0

    while True:
        _, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, **HS_params)

        vis = draw_flow(frame_gray, flow)

        cv2.imshow('flow', vis)

        # 保存光流图像
        cv2.imwrite(os.path.join(diff_folder, f'frame_diff_{frame_count:04d}.png'), vis)
        # 保存原始图像
        cv2.imwrite(os.path.join(origin_frame_folder, f'frame_diff_{frame_count:04d}.png'), frame)
        frame_count += 1

        _, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        if (cv2.waitKey(1) & 0xff) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    # 创建保存光流图像的文件夹
    diff_folder = 'E:/test/frame_optical_flow'
    if not os.path.exists(diff_folder):
        os.makedirs(diff_folder)

    # 原始图像
    origin_frame_folder = 'E:/test/origin_frame'
    if not os.path.exists(origin_frame_folder):
        os.makedirs(origin_frame_folder)

    OpticalFlow()
