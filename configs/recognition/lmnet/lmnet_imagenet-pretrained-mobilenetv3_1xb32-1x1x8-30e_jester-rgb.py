_base_ = [
    '../../_base_/models/lmnet_mobilenet_v3.py', '../../_base_/default_runtime.py'
]

env_cfg = dict(cudnn_benchmark=True)

# model settings
model = dict(
    cls_head=dict(num_classes=83, dropout_ratio=0.5)
)

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/root/autodl-tmp/jester/rawframes'
data_root_val = '/root/autodl-tmp/jester/rawframes'
ann_file_train = '/root/autodl-tmp/mmaction2/data/jester/jester_train_list_rawframes.txt'
ann_file_val = '/root/autodl-tmp/mmaction2/data/jester/jester_val_list_rawframes.txt'
ann_file_test = '/root/autodl-tmp/mmaction2/data/jester/jester_val_list_rawframes.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),  # 最短边缩放为256
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),  # 多尺度裁剪 256*[1,0.875,0.75,0.66]=[256 224 192 168]尺度的裁剪
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),  # 13个位置随机裁剪
    dict(type='Resize', scale=(224, 224), keep_ratio=False),  # resize到(224,224)
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
# test_pipeline = val_pipeline
train_dataloader = dict(
    batch_size=32,
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        filename_tmpl='{:05}.jpg',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=32,
    num_workers=16,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:05}.jpg',
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:05}.jpg',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

randomness = dict(deterministic=False, diff_rank_seed=False, seed=2024)

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    # dict(type='CosineAnnealingLR', by_epoch=True, begin=5)
    dict(
        type='MultiStepLR',
        begin=5,
        end=30,
        by_epoch=True,
        milestones=[5, 15, 20],
        gamma=0.1)
]

optim_wrapper = dict(
    constructor='TSMOptimWrapperConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=20, norm_type=2))

default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (4 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=32)
