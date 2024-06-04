# 在gen_dataset_egogesture.py后生成csv文件
# 解析SubjectXX_SceneXX_rgbXX_class_startFrame_endFrame得到标注csv文件
import csv
import os

# dataset_path = 'E:/Dataset/RS_v1_50_EgoGesture/img'
# csv_path = 'E:/Dataset/RS_v1_50_EgoGesture/csv'

dataset_path = '/root/autodl-tmp/RS_v1_50/img'
csv_path = '/root/autodl-tmp/RS_v1_50/csv'


if __name__ == '__main__':
    train_list_path = os.path.join(csv_path,'train_list.txt')
    val_list_path = os.path.join(csv_path,'val_list.txt')
    test_list_path = os.path.join(csv_path,'test_list.txt')
    train_plus_val_list_path = os.path.join(csv_path,'train_plus_val_list.txt')
    err_list_path = os.path.join(csv_path,'err_list.txt')

    train_ids = [3, 4, 5, 6, 8, 10, 15, 16, 17, 20, 21, 22, 23, 25, 26, 27, 30, 32, 36, 38, 39, 40, 42, 43, 44, 45,
               46, 48, 49, 50]
    val_ids = [1, 7, 12, 13, 24, 29, 33, 34, 35, 37]
    test_ids = [2, 9, 11, 14, 18, 19, 28, 31, 41, 47]

    train_list = []
    val_list = []
    test_list = []
    train_plus_val_list = []
    err_list = []

    with open(os.path.join(csv_path, 'all.csv'),'r') as f:
        input = csv.reader(f)
        for row in input:
            sub = row[0][7:9]
            # 下标从0开始
            row[2] = str(int(row[2])-1)
            if int(sub) in train_ids:
                # 格式：文件夹 总帧数 类别号
                train_list.append(row)
            elif int(sub) in val_ids:
                val_list.append(row)
            elif int(sub) in test_ids:
                test_list.append(row)
            else:
                err_list.append(row)
    train_plus_val_list = train_list + val_list
    print('train:{}'.format(len(train_list)))
    print('val:{}'.format(len(val_list)))
    print('test:{}'.format(len(test_list)))
    print('train_plus_val:{}'.format(len(train_plus_val_list)))
    with open(train_list_path, 'w', encoding='utf8', newline='') as f:
        # delimiter分隔符
        writer = csv.writer(f,delimiter=' ')
        writer.writerows(train_list)
    with open(val_list_path, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f,delimiter=' ')
        writer.writerows(val_list)
    with open(test_list_path, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f,delimiter=' ')
        writer.writerows(test_list)
    with open(train_plus_val_list_path, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(train_plus_val_list)
    with open(err_list_path, 'w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(err_list)



