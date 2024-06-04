_base_ = [
    '../../_base_/models/swin_tiny.py', '../../_base_/default_runtime.py'
]
# 88.41 97.99
model = dict(
    backbone=dict(
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth'  # noqa: E501
    ))

dataset_type = 'RawframeDataset'
data_root = '/root/autodl-tmp/RS_v1_50/img'
data_root_val = '/root/autodl-tmp/RS_v1_50/img'
ann_file_train = '/root/autodl-tmp/RS_v1_50/csv/train_plus_val_list.txt'
ann_file_val = '/root/autodl-tmp/RS_v1_50/csv/test_list.txt'
ann_file_test = '/root/autodl-tmp/RS_v1_50/csv/test_list.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),# 最短边缩放为256
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),# 多尺度裁剪 256*[1,0.875,0.75,0.66]=[256 224 192 168]尺度的裁剪
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),# 13个位置
    dict(type='Resize', scale=(224, 224), keep_ratio=False),# resize到(224,224)
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=1,
        num_clips=4,
        test_mode=True),
    dict(type='RawFrameDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='ThreeCrop', crop_size=256),
dict(type='CenterCrop', crop_size=224),# 87.66 97.89
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
# test_pipeline = val_pipeline
train_dataloader = dict(
    batch_size=32,
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        filename_tmpl='{:06}.jpg',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=32,
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:06}.jpg',
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=8,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        filename_tmpl='{:06}.jpg',
        pipeline=test_pipeline,
        test_mode=True))




val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=1, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.1)))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2.5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=30,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=30)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (32 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=256)
