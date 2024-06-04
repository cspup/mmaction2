# Convert the dataset to a format similar to sthv1
# 将数据集转换为RawFrameDataset
# Folder Name be like: SubjectXX_SceneXX_rgbXX_class_startFrame_endFrame
# 文件夹格式名为：原一级目录_原二级目录_原三级目录_手势类别_起始帧_结束帧
# if you want get label.txt you need run gen_label_egogesture.py after run gen_dataseet_egogesture.py
# 之后要获取标注文件需要再运行gen_label_egogesture.py
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
import csv

# # label files stored path Linux
# label_path = 'E:/Dataset/RS_v1_50_EgoGesture/labels-final-revised1'
# # frame files stored path
# frame_path = 'E:/Dataset/RS_v1_50_EgoGesture/images/'
# target_path = 'E:/Dataset/RS_v1_50_EgoGesture/img'
# csv_path = 'E:/Dataset/RS_v1_50_EgoGesture/csv'

# label files stored path Linux
label_path = '/root/autodl-tmp/RS_v1_50/labels-final-revised1'
# frame files stored path
frame_path = '/root/autodl-tmp/RS_v1_50/images/'
target_path = '/root/autodl-tmp/RS_v1_50/img'
csv_path = '/root/autodl-tmp/RS_v1_50/csv'

if not os.path.exists(target_path):
    os.mkdir(target_path)

if not os.path.exists(csv_path):
    os.mkdir(csv_path)

sub_ids = range(1, 50 + 1)
csv_list = []

for sub_i in tqdm(sub_ids):
    frame_path_sub = os.path.join(frame_path, 'Subject{:02}'.format(sub_i))
    label_path_sub = os.path.join(label_path, 'subject{:02}'.format(sub_i))
    assert len([name for name in os.listdir(label_path_sub) if name != '.DS_Store']) == len(
        [name for name in os.listdir(frame_path_sub)])
    for scene_i in range(1, len([name for name in os.listdir(frame_path_sub)]) + 1):
        rgb_path = os.path.join(frame_path_sub, 'Scene{:01}'.format(scene_i), 'Color')
        label_path_iter = os.path.join(label_path_sub, 'Scene{:01}'.format(scene_i))
        assert len([name for name in os.listdir(label_path_iter) if 'csv' == name[-3::]]) == len(
            [name for name in os.listdir(rgb_path)])
        for group_i in range(1, len([name for name in os.listdir(rgb_path)]) + 1):
            rgb_path_group = os.path.join(rgb_path, 'rgb{:01}'.format(group_i))
            if os.path.isfile(os.path.join(label_path_iter, 'Group{:01}.csv'.format(group_i))):
                label_path_group = os.path.join(label_path_iter, 'Group{:01}.csv'.format(group_i))
            else:
                label_path_group = os.path.join(label_path_iter, 'group{:01}.csv'.format(group_i))
            data_note = pd.read_csv(label_path_group, names=['class', 'start', 'end'])
            data_note = data_note[np.isnan(data_note['start']) == False]
            for data_i in range(data_note.values.shape[0]):
                label = data_note.values[data_i, 0]
                start = int(data_note.values[data_i, 1])
                end = int(data_note.values[data_i, 2])

                list = ['Subject{:02}'.format(sub_i), 'Scene{:01}'.format(scene_i), 'rgb{:01}'.format(group_i),
                        str(label), str(start), str(end)]
                target = os.path.join(target_path, '-'.join(list))
                if not os.path.exists(target):
                    os.mkdir(target)
                for i in range(start, end + 1):
                    src = os.path.join(rgb_path_group, '{:06}.jpg'.format(i))
                    shutil.copy(src, target)
                    old_name = os.path.join(target, '{:06}.jpg'.format(i))
                    new_name = os.path.join(target, '{:06}.jpg'.format(i - start + 1))
                    os.rename(old_name, new_name)
                csv_list.append(('-'.join(list), end + 1 - start, label))

with open(os.path.join(csv_path, 'all.csv'), 'w', encoding='utf8', newline='') as f:
    writer = csv.writer(f)
    # all.csv文件内容：文件夹 帧数 类别
    writer.writerows(csv_list)

# path = 'E:/Dataset/RS_v1_50_EgoGesture/img'
# if not os.path.exists(path):
#     os.mkdir(path)
# for i in range(178,218+1):
#     src = os.path.join('E:/Dataset/RS_v1_50_EgoGesture/images/Subject01/Scene1/Color/rgb1/','{:06}.jpg'.format(i))
#     shutil.copy(src, path)

with open(os.path.join(csv_path, 'label.csv'), 'w', encoding='utf8', newline='') as f:
    writer = csv.writer(f)
    label_list= [range(1, 84)]
    writer.writerows(label_list)
