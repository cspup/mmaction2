from mmengine import MMLogger

from mmaction.registry import MODELS
from .efficientnetv2 import EfficientNetV2
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_with_prefix)

@MODELS.register_module()
class EfficientNetV2LAMME(EfficientNetV2):
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
            print(m)


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