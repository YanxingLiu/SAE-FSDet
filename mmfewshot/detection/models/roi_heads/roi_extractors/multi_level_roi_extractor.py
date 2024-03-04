import torch
import torch.nn as nn
from mmcv import ops
from mmcv.runner import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.models.roi_heads.roi_extractors.base_roi_extractor import \
    BaseRoIExtractor
from scipy import spatial
from sympy import expand


@ROI_EXTRACTORS.register_module()
class MultiRoIExtractor(BaseRoIExtractor):
    """从多尺度的特征图中提取多尺度的ROI特征."""

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 init_cfg=None):
        super().__init__(roi_layer, out_channels, featmap_strides, init_cfg)

    def build_roi_layers(self, layer_cfg, featmap_strides):
        """Build RoI operator to extract feature from each level feature map.

        Args:
            layer_cfg (List[dict]): Dictionary to construct and config RoI layer
                operation. Options are modules under ``mmcv/ops`` such as
                ``RoIAlign``.
            featmap_strides (List[int]): The stride of input feature map w.r.t
                to the original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.

        Returns:
            nn.ModuleList: The RoI extractor modules for each level feature
                map.
        """
        assert len(layer_cfg) == len(
            featmap_strides), 'roi layer num must equal to featmap stride num'
        roi_layers = []
        for i in range(len(layer_cfg)):
            # load cfg
            cfg = layer_cfg[i].copy()
            feature_stride = featmap_strides[i]
            # build roi layer
            layer_type = cfg.pop('type')
            assert hasattr(ops, layer_type)
            layer_cls = getattr(ops, layer_type)
            # roi_layer = nn.ModuleList([layer_cls(spatial_scale=1 / featmap_strides[i], **cfg) ])
            roi_layer = layer_cls(spatial_scale=1 / feature_stride, **cfg)
            roi_layers.append(roi_layer)
        return roi_layers

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois):
        """Forward function."""
        num_levels = len(feats)
        out_size = [x.output_size for x in self.roi_layers]
        roi_feats = []
        for i in range(num_levels):
            roi_feat = feats[i].new_zeros(
                rois.size(0), self.out_channels, *out_size[i])
            roi_feats.append(roi_feat)
        # roi align each level
        res = []
        for i in range(num_levels):
            res.append(self.roi_layers[i](feats[i], rois))
        return res
