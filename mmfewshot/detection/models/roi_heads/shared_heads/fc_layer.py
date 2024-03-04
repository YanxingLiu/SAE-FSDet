import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.builder import SHARED_HEADS


@SHARED_HEADS.register_module()
class FCLayer(BaseModule):

    def __init__(self,
                 input_size_coarse,
                 input_size_fine,
                 output_size_coarse,
                 output_size_fine,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.pretrained = pretrained
        self.input_size_coarse = input_size_coarse
        self.input_size_fine = input_size_fine
        self.layer_coarse = nn.Linear(
            input_size_coarse[0] * input_size_coarse[1] * input_size_coarse[2],
            output_size_coarse)
        self.layer_fine = nn.Linear(
            input_size_fine[0] * input_size_fine[1] * input_size_fine[2],
            output_size_fine)

    def forward(self, features):
        features_coarse = features[0]
        features_fine = features[1]
        out_coarse = features_coarse.view(
            -1, self.input_size_coarse[0] * self.input_size_coarse[1] *
            self.input_size_coarse[2])
        out_coarse = self.layer_coarse(out_coarse)
        out_fine = features_fine.view(
            -1, self.input_size_fine[0] * self.input_size_fine[1] *
            self.input_size_fine[2])
        out_fine = self.layer_fine(out_fine)
        return [out_coarse, out_fine]
