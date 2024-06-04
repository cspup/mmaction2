# model settings
model = dict(
    type='Recognizer2D',
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.5],
        std=[58.395, 57.12, 57.375],
        format_shape='NCHW'),
    backbone=dict(
        type='LMNet',
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        # pretrained='torchvision://resnet50',
        norm_eval=False,
        depth=50,
        num_segments=8
        # torchvision_pretrain = False,
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint=  # noqa: E251
        #     'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth',
        #     prefix='backbone.'),
    ),

    cls_head=dict(
        type='TSMHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        average_clips='prob'))
