_base_ = [
    '../../_base_/models/lmnet_r50.py', '../../_base_/default_runtime.py',
    '../../_base_/schedules/sgd_tsm_25e.py'
]

env_cfg = dict(cudnn_benchmark=True)

# model settings
model = dict(
    backbone=dict(num_segments=16,pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth'),
    cls_head=dict(num_classes=83, dropout_ratio=0.5, num_segments=16),

)

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/root/autodl-tmp/RS_v1_50/img'
data_root_val = '/root/autodl-tmp/RS_v1_50/img'
ann_file_train = '/root/autodl-tmp/RS_v1_50/csv/train_plus_val_list.txt'
ann_file_val = '/root/autodl-tmp/RS_v1_50/csv/test_list.txt'
ann_file_test = '/root/autodl-tmp/RS_v1_50/csv/test_list.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1,256)), # 最短边缩放到256
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),  # 多尺度裁剪
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),  # 13个位置
    dict(type='Resize', scale=(224, 224), keep_ratio=False), #缩放到网络输入
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='RawFrameDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=(224, 224)),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True,num_sample=1),# 使用test_mode=False以达到随机取样的效果，num_sample随机取样次数
    dict(type='RawFrameDecode'),
    # dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='Resize', scale=(-1, 232)),
    dict(type='CenterCrop', crop_size=(224, 224)),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
# test_pipeline = val_pipeline
train_dataloader = dict(
    batch_size=16,
    num_workers=15,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        filename_tmpl='{:06}.jpg',
        pipeline=train_pipeline,
        start_index=1))
val_dataloader = dict(
    batch_size=16,
    num_workers=15,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:06}.jpg',
        pipeline=val_pipeline,
        test_mode=True,
        start_index=1))
test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:06}.jpg',
        pipeline=test_pipeline,
        test_mode=True,
        start_index=1))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=0.01, weight_decay=5e-4))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=25,
        by_epoch=True,
        milestones=[10, 15, 20],
        gamma=0.1)
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=25, val_begin=1, val_interval=1)

default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (4 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)
