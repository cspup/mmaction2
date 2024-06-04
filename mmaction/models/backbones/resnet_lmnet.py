# Copyright (c) OpenMMLab. All rights reserved.
# for resnet 18 34
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import _load_checkpoint

from mmaction.registry import MODELS
from .resnet import ResNet
from ..common.lamme import LAM, ME


class TemporalBlock(nn.Module):

    def __init__(self, net, num_segments=8):
        super().__init__()
        self.net = net
        self.num_segments = num_segments
        self.lam = LAM(self.net.out_features, self.num_segments)
        self.me = ME()

    def forward(self, x):
        x = self.net(x)
        x = self.lam(x)
        x = self.me(x) * x
        return x


@MODELS.register_module()
class ResNetLMNet(ResNet):

    def __init__(self,
                 depth,
                 num_segments=8,
                 pretrained2d=True,
                 **kwargs):
        super().__init__(depth, **kwargs)
        self.num_segments = num_segments
        self.pretrained2d = pretrained2d
        self.init_structure()

    def init_structure(self):
        self.make_temporal_shift()

    def make_temporal_shift(self):
        """Make temporal shift for some layers."""
        num_segment_list = [self.num_segments] * 4
        if num_segment_list[-1] <= 0:
            raise ValueError('num_segment_list[-1] must be positive')

        # if self.shift_place == 'block':
        #
            def make_block_temporal(stage, num_segments):
                blocks = list(stage.children())
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalBlock(b, num_segments=num_segments)
                return nn.Sequential(*blocks)

            self.layer1 = make_block_temporal(self.layer1, num_segment_list[0])
            self.layer2 = make_block_temporal(self.layer2, num_segment_list[1])
            self.layer3 = make_block_temporal(self.layer3, num_segment_list[2])
            self.layer4 = make_block_temporal(self.layer4, num_segment_list[3])
        #
        # elif 'blockres' in self.shift_place:
        #     n_round = 1
        #     if len(list(self.layer3.children())) >= 23:
        #         n_round = 2
        #
        #     def make_block_temporal(stage, num_segments):
        #         blocks = list(stage.children())
        #         for i, b in enumerate(blocks):
        #             if i % n_round == 0:
        #                 blocks[i].conv1.conv = TemporalBlock(
        #                     b.conv1.conv,
        #                     num_segments=num_segments)
        #         return nn.Sequential(*blocks)
        #
        #     self.layer1 = make_block_temporal(self.layer1, num_segment_list[0])
        #     self.layer2 = make_block_temporal(self.layer2, num_segment_list[1])
        #     self.layer3 = make_block_temporal(self.layer3, num_segment_list[2])
        #     self.layer4 = make_block_temporal(self.layer4, num_segment_list[3])
        #
        # else:
        #     raise NotImplementedError


    def _get_wrap_prefix(self):
        return ['.net', '.block']

    def load_original_weights(self, logger):
        """Load weights from original checkpoint, which required converting
        keys."""
        state_dict_torchvision = _load_checkpoint(
            self.pretrained, map_location='cpu')
        if 'state_dict' in state_dict_torchvision:
            state_dict_torchvision = state_dict_torchvision['state_dict']

        wrapped_layers_map = dict()
        for name, module in self.named_modules():
            # convert torchvision keys
            ori_name = name
            for wrap_prefix in self._get_wrap_prefix():
                if wrap_prefix in ori_name:
                    ori_name = ori_name.replace(wrap_prefix, '')
                    wrapped_layers_map[ori_name] = name

            if isinstance(module, ConvModule):
                if 'downsample' in ori_name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    tv_conv_name = ori_name + '.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    tv_bn_name = ori_name + '.1'
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    tv_conv_name = ori_name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    tv_bn_name = ori_name.replace('conv', 'bn')

                for conv_param in ['.weight', '.bias']:
                    if tv_conv_name + conv_param in state_dict_torchvision:
                        state_dict_torchvision[ori_name + '.conv' + conv_param] = \
                            state_dict_torchvision.pop(tv_conv_name + conv_param)

                for bn_param in [
                    '.weight', '.bias', '.running_mean', '.running_var'
                ]:
                    if tv_bn_name + bn_param in state_dict_torchvision:
                        state_dict_torchvision[ori_name + '.bn' + bn_param] = \
                            state_dict_torchvision.pop(tv_bn_name + bn_param)

        # convert wrapped keys
        for param_name in list(state_dict_torchvision.keys()):
            layer_name = '.'.join(param_name.split('.')[:-1])
            if layer_name in wrapped_layers_map:
                wrapped_name = param_name.replace(
                    layer_name, wrapped_layers_map[layer_name])
                print(f'wrapped_name {wrapped_name}')
                state_dict_torchvision[
                    wrapped_name] = state_dict_torchvision.pop(param_name)

        msg = self.load_state_dict(state_dict_torchvision, strict=False)
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
