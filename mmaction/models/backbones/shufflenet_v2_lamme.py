import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_with_prefix)

from mmaction.registry import MODELS
from .shufflenetv2 import InvertedResidual, ShuffleNetV2
class LAM(nn.Module):
    #    Long-term Time Sequence Aggregation Module
    def __init__(self,
                 in_channels,
                 n_segment,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 init_std=0.001):
        super(LAM, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.GlobalPool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0)
        self.MLP = nn.Sequential(
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.BatchNorm1d(n_segment * 2),
            nn.ReLU(inplace=True),
            nn.Linear(n_segment * 2, kernel_size, bias=False),
            nn.Softmax(-1))

        print("Using LAM==>")

    def forward(self, x):
        # x.size = N*C*T*(H*W)
        nt, c, h, w = x.size()
        t = self.n_segment
        n_batch = nt // t
        new_x = x.view(n_batch, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        x_g = self.GlobalPool(new_x).view(n_batch, c, 1)
        x_g = self.conv(x_g)
        x_g = x_g.view(n_batch, c, 1, 1, 1)
        new_x = new_x + x_g

        out = F.adaptive_avg_pool2d(new_x.view(n_batch * c, t, h, w), (1, 1))
        out = out.view(-1, t).contiguous()
        weight = self.MLP(out)
        weight = weight.view(n_batch * c, 1, -1, 1)

        out = F.conv2d(new_x.view(1, n_batch * c, t, h * w),
                       weight=weight,
                       bias=None,
                       stride=(self.stride, 1),
                       padding=(self.padding, 0),
                       groups=n_batch * c)

        out = out.view(n_batch, c, t, h, w).contiguous()
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)

        return out

class ME(nn.Module):
    """
    Motion exexcitation
    Constructs a ME module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3, n_segment=8):
        super(ME, self).__init__()
        self.n_segment = n_segment
        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        print("Using ME===> kernel_size={}".format(k_size))

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x3_plus0, _ = x.view(n_batch, self.n_segment, c, h, w).split([self.n_segment - 1, 1], dim=1)
        x3_plus1 = x

        _, x3_plus1 = x3_plus1.view(n_batch, self.n_segment, c, h, w).split([1, self.n_segment - 1], dim=1)
        diff = x3_plus1 - x3_plus0
        diff = F.pad(diff, self.pad, mode="constant", value=0)
        diff = diff.view(nt, c, h, w)

        y = self.avg_pool(diff)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)

        # y = self.conv1(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return y.expand_as(x) * x

@MODELS.register_module()
class ShuffleNetV2LAMME(ShuffleNetV2):
    def __init__(self,
                 num_segments=8,
                 pretrained2d=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_segments = num_segments
        self.pretrained2d = pretrained2d
        self.init_structure()

    def init_structure(self):
        print("init structure")
        for m in self.modules():
            if isinstance(m,InvertedResidual):
                m.branch2.add_module('LAM',LAM(m.branch2[2].out_channels, n_segment=self.num_segments))
                m.branch2.add_module('ME', ME(k_size=3, n_segment=self.num_segments))

    def load_original_weights(self, logger):
        assert self.init_cfg.get('type') == 'Pretrained', (
            'Please specify '
            'init_cfg to use pretrained 2d checkpoint')
        self.pretrained = self.init_cfg.get('checkpoint')
        prefix = self.init_cfg.get('prefix')
        if prefix is not None:
            original_state_dict = _load_checkpoint_with_prefix(
                prefix, self.pretrained, map_location='cpu')
        else:
            original_state_dict = _load_checkpoint(
                self.pretrained, map_location='cpu')
        if 'state_dict' in original_state_dict:
            original_state_dict = original_state_dict['state_dict']

        wrapped_layers_map = dict()
        for name, module in self.named_modules():
            ori_name = name
            for wrap_prefix in ['.net']:
                if wrap_prefix in ori_name:
                    ori_name = ori_name.replace(wrap_prefix, '')
                    wrapped_layers_map[ori_name] = name

        # convert wrapped keys
        for param_name in list(original_state_dict.keys()):
            layer_name = '.'.join(param_name.split('.')[:-1])
            if layer_name in wrapped_layers_map:
                wrapped_name = param_name.replace(
                    layer_name, wrapped_layers_map[layer_name])
                original_state_dict[wrapped_name] = original_state_dict.pop(
                    param_name)

        msg = self.load_state_dict(original_state_dict, strict=False)
        logger.info(msg)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if self.pretrained2d:
            logger = MMLogger.get_current_instance()
            self.load_original_weights(logger)
        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()

    def forward(self, x):
        """unpack tuple result."""
        x = super().forward(x)
        if isinstance(x, tuple):
            assert len(x) == 1
            x = x[0]
        return x