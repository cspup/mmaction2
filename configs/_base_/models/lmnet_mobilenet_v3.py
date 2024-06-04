# model settings
preprocess_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_large-3ea3c186.pth'

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='MobileNetV3LAMME',
        num_segments=8,
        arch='large',
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')
    ),
    cls_head=dict(
        type='TSMHead',
        num_segments=8,
        num_classes=400,
        in_channels=960,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        average_clips='prob'),
    # model training and testing settings
    data_preprocessor=dict(type='ActionDataPreprocessor', **preprocess_cfg),
    train_cfg=None,
    test_cfg=None)
