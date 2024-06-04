# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model.weight_init import normal_init
from torch import Tensor, nn

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, get_str_type
from .base import AvgConsensus, BaseHead


@MODELS.register_module()
class MobileNetV3Head(BaseHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 num_segments: int = 8,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 spatial_type: str = 'avg',
                 consensus: ConfigType = dict(type='AvgConsensus', dim=1),
                 dropout_ratio: float = 0.8,
                 init_std: float = 0.001,
                 temporal_pool: bool = False,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.num_segments = num_segments
        self.init_std = init_std
        self.temporal_pool = temporal_pool

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if get_str_type(consensus_type) == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc_cls = nn.Sequential(
            nn.Linear(self.in_channels, 1280),
            nn.Hardswish(inplace=True),
            self.dropout,
            nn.Linear(1280, self.num_classes),
        )

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = None

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x: Tensor, num_segs: int, **kwargs) -> Tensor:
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        if self.dropout is not None:
            x = self.dropout(x)
        cls_score = self.fc_cls(x)

        if self.temporal_pool:
            # [2 * N, num_segs // 2, num_classes]
            cls_score = cls_score.view((-1, self.num_segments // 2) +
                                       cls_score.size()[1:])
        else:
            # [N, num_segs, num_classes]
            cls_score = cls_score.view((-1, self.num_segments) +
                                       cls_score.size()[1:])
        # [N, 1, num_classes]
        cls_score = self.consensus(cls_score)
        # [N, num_classes]
        return cls_score.squeeze(1)
