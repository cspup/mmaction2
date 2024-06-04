import torch
import torch.nn as nn
from torch.utils import checkpoint as cp

from mmaction.registry import MODELS
from .resnet import Bottleneck, ResNet
from ..common.lamme import LAM, ME, SELayer, CBAM, eca_layer


class LAMMEBlock(nn.Module):
    """
    Args:
        block (nn.Module): Residual blocks to be substituted.
        num_segments (int): Number of frame segments.
    """

    def __init__(self, block: nn.Module, num_segments: int) -> None:
        super().__init__()
        self.block = block
        self.num_segments = num_segments

        self.lam = LAM(in_channels=block.conv2.out_channels, n_segment=num_segments)
        self.ME = ME(k_size=3,n_segment=num_segments)

        # self.attention = SELayer(block.conv3.out_channels)
        # self.attention = CBAM(block.conv3.out_channels)
        # self.attention = eca_layer(block.conv3.out_channels)


        if not isinstance(self.block, Bottleneck):
            raise NotImplementedError('LAMME-Block have not been fully '
                                      'implemented except the pattern based '
                                      'on Bottleneck block.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        assert isinstance(self.block, Bottleneck)

        def _inner_forward(x1):
            """Forward wrapper for utilizing checkpoint."""
            identity = x1

            x1 = self.ME(x1) * x1 + x1

            out1 = self.block.conv1(x1)

            out1 = self.block.conv2(out1)
            out1 = self.lam(out1)
            out1 = self.block.conv3(out1)
            # out1 = self.attention(out1)
            # out1 = self.ME(out1) * out1 + ou1

            if self.block.downsample is not None:
                identity = self.block.downsample(x1)

            return out1 + identity

        if self.block.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.block.relu(out)

        return out


@MODELS.register_module()
class LMNet(ResNet):
    """
    Args:
        depth (int): Depth of resnet, from ``{18, 34, 50, 101, 152}``.
        num_segments (int): Number of frame segments.
    """

    def __init__(self,
                 depth: int,
                 num_segments: int,
                 **kwargs) -> None:
        super().__init__(depth, **kwargs)
        assert num_segments >= 3
        self.num_segments = num_segments
        super().init_weights()
        self.make_lamme_modeling()

    def init_weights(self):
        """Initialize weights."""
        pass

    def make_lamme_modeling(self):
        """Replace ResNet-Block with LAMME-Block."""

        def make_lamme_block(stage, num_segments):
            blocks = list(stage.children())
            for index, block in enumerate(blocks):
                blocks[index] = LAMMEBlock(block, num_segments)
            return nn.Sequential(*blocks)


        for i in range(self.num_stages):
            layer_name = f'layer{i + 1}'
            res_layer = getattr(self, layer_name)
            setattr(self, layer_name,
                        make_lamme_block(res_layer, self.num_segments))

