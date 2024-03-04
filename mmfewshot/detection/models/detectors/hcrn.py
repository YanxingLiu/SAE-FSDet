# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional

import torch
from mmcv.runner import auto_fp16
from mmcv.utils import ConfigDict
from mmdet.core import bbox2roi
from mmdet.models.builder import DETECTORS
from torch import Tensor

from .query_support_detector import QuerySupportDetector


def cal_clsvec_init(data, fine_labels, num_class):  # 90*64*19*19
    """初始化每个超类的均值,可能是对均值特征向量缩放了一个倍数,同时它也是求的通道维度的均值，以[H,W]作为模板进行匹配。"""
    class_vec = torch.zeros(
        [num_class, data.shape[1], data.shape[2], data.shape[3]]).cuda()
    for i in range(num_class):
        idx = [j for j, x in enumerate(fine_labels)
               if x == i]  # 第idx个样本的标签为i，idx为一个列表，列表保存了所有标签为i的样本的索引
        sigma_cls = torch.zeros(
            [data.shape[0], data.shape[1], data.shape[2],
             data.shape[3]]).cuda()
        for m in range(len(idx)):
            s = data[idx[m], :, :, :]  # 第m个特征向量 shape为[C,H,W]
            avg_s = torch.sum(
                s, dim=0
            ) / len(
                s
            )  # sum(s)为每个通道的特征的和，shape为[H,W]，len(s)为通道数 shape为C，所以这个avg_s是求通道特征均值
            sigma_cls += avg_s
        vec = torch.sum(sigma_cls, dim=0) / len(idx)  # 求样本均值
        class_vec[i] = vec

    return class_vec


@DETECTORS.register_module()
class HCRN(QuerySupportDetector):

    def __init__(self,
                 backbone: ConfigDict,
                 neck: Optional[ConfigDict] = None,
                 support_backbone: Optional[ConfigDict] = None,
                 support_neck: Optional[ConfigDict] = None,
                 rpn_head: Optional[ConfigDict] = None,
                 roi_head: Optional[ConfigDict] = None,
                 train_cfg: Optional[ConfigDict] = None,
                 test_cfg: Optional[ConfigDict] = None,
                 pretrained: Optional[ConfigDict] = None,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            support_backbone=support_backbone,
            support_neck=support_neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.is_model_init = False
        # save support template features for model initialization,
        # `_forward_saved_support_dict` used in :func:`forward_model_init`.
        self._forward_saved_support_dict = {
            'gt_labels': [],
            'coarse_feats': [],
            'fine_feats': [],
        }
        # save processed support template features for inference,
        # the processed support template features are generated
        # in :func:`model_init`
        self.inference_support_dict = {
            'coarse_feats': {},
            'fine_feats': {},
        }

    def forward_train(self,
                      query_data: Dict,
                      support_data: Dict,
                      proposals: Optional[List] = None,
                      **kwargs) -> Dict:
        """Forward function for training.

        Args:
            query_data (dict): In most cases, dict of query data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            support_data (dict):  In most cases, dict of support data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            proposals (list): Override rpn proposals with custom proposals.
                Use when `with_rpn` is False. Default: None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        query_img = query_data['img']
        support_img = support_data['img']
        query_feats = self.extract_query_feat(query_img)
        # query_feast[0]为粗粒度特征，query_feats[1]为细粒度特征
        support_feats = self.extract_support_feat(support_img)
        # support_feast[0]为粗粒度特征，support_feast[1]为细粒度特征

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                query_feats,
                copy.deepcopy(query_data['img_metas']),
                copy.deepcopy(query_data['gt_bboxes']),
                gt_labels=None,
                gt_bboxes_ignore=copy.deepcopy(
                    query_data.get('gt_bboxes_ignore', None)),
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            query_feats,
            support_feats,
            proposals=proposal_list,
            query_img_metas=query_data['img_metas'],
            query_gt_bboxes=query_data['gt_bboxes'],
            query_gt_labels=query_data['gt_labels'],
            query_gt_bboxes_ignore=query_data.get('gt_bboxes_ignore', None),
            support_img_metas=support_data['img_metas'],
            support_gt_bboxes=support_data['gt_bboxes'],
            support_gt_labels=support_data['gt_labels'],
            support_gt_bboxes_ignore=support_data.get('gt_bboxes_ignore',
                                                      None),
            **kwargs)
        losses.update(roi_losses)

        return losses

    @auto_fp16(apply_to=('img', ))
    def extract_support_feat(self, img):
        """Extracting features from support data.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            list[Tensor]: Features of input image, each item with shape
                (N, C, H, W).
        """
        feats = self.backbone(img)
        if self.with_neck:
            feats = self.neck(feats)
        return feats

    def forward_model_init(self,
                           img: Tensor,
                           img_metas: List[Dict],
                           gt_bboxes: List[Tensor] = None,
                           gt_labels: List[Tensor] = None,
                           **kwargs):
        """模测试初始化时前向传播函数."""
        # `is_model_init` flag will be reset when forward new data.
        self.is_model_init = False
        assert len(gt_labels) == img.size(
            0), 'Support instance have more than two labels'
        if self.roi_head.input_indices is None:
            feats = self.extract_support_feat(img)
        else:
            feats = self.extract_support_feat(img)
            feats = [feats[i] for i in self.roi_head.input_indices]
        self._forward_saved_support_dict['gt_labels'].extend(gt_labels)
        self._forward_saved_support_dict['coarse_feats'].extend(feats[0])
        self._forward_saved_support_dict['fine_feats'].extend(feats[1])
        return {
            'gt_labels': gt_labels,
            'coarse_feats': feats[0],
            'fine_feats': feats[1]
        }

    def model_init(self):
        """模型测试初始化函数."""
        gt_labels = torch.cat(self._forward_saved_support_dict['gt_labels'])
        coarse_feats = torch.stack(
            self._forward_saved_support_dict['coarse_feats'])
        fine_feats = torch.stack(
            self._forward_saved_support_dict['fine_feats'])
        class_ids = set(gt_labels.data.tolist())
        # init inference support dict
        self.inference_support_dict.clear()
        self.inference_support_dict['coarse_feats'] = dict()
        self.inference_support_dict['fine_feats'] = dict()
        # hcrn roi_head v1
        # coarse_mean_vector = cal_clsvec_init(coarse_feats, gt_labels, len(class_ids))
        # fine_mean_vector = cal_clsvec_init(fine_feats, gt_labels, len(class_ids))
        # for class_id in class_ids:
        #     self.inference_support_dict['coarse_feats'][class_id] = coarse_mean_vector[class_id][None,:,:,:]
        #     self.inference_support_dict['fine_feats'][class_id]   = fine_mean_vector[class_id][None,:,:,:]
        # hcrn roi head v2
        for class_id in class_ids:
            self.inference_support_dict['coarse_feats'][
                class_id] = coarse_feats[gt_labels == class_id].mean([0], True)
            self.inference_support_dict['fine_feats'][class_id] = fine_feats[
                gt_labels == class_id].mean([0], True)
        # set the init flag
        self.is_model_init = True
        # reset support features buff
        for k in self._forward_saved_support_dict.keys():
            self._forward_saved_support_dict[k].clear()

    def simple_test(self,
                    img: Tensor,
                    img_metas: List[Dict],
                    proposals: Optional[List[Tensor]] = None,
                    rescale: bool = False):
        """模型测试函数.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor] | None): override rpn proposals with
                custom proposals. Use when `with_rpn` is False. Default: None.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # assert self.with_bbox, 'Bbox head must be implemented.'
        assert len(img_metas) == 1, 'Only support single image inference.'
        if not self.is_model_init:
            # process the saved support features
            self.model_init()

        query_feats = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test(query_feats, img_metas)
        else:
            proposal_list = proposals
        return self.roi_head.simple_test(
            query_feats,
            copy.deepcopy(self.inference_support_dict),
            proposal_list,
            img_metas,
            rescale=rescale)


@DETECTORS.register_module()
class HCRNV3(QuerySupportDetector):

    def __init__(self,
                 backbone: ConfigDict,
                 neck: Optional[ConfigDict] = None,
                 support_backbone: Optional[ConfigDict] = None,
                 support_neck: Optional[ConfigDict] = None,
                 rpn_head: Optional[ConfigDict] = None,
                 roi_head: Optional[ConfigDict] = None,
                 train_cfg: Optional[ConfigDict] = None,
                 test_cfg: Optional[ConfigDict] = None,
                 pretrained: Optional[ConfigDict] = None,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            support_backbone=support_backbone,
            support_neck=support_neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.is_model_init = False
        # save support template features for model initialization,
        # `_forward_saved_support_dict` used in :func:`forward_model_init`.
        self._forward_saved_support_dict = {
            'gt_labels': [],
            'coarse_feats': [],
            'fine_feats': [],
        }
        # save processed support template features for inference,
        # the processed support template features are generated
        # in :func:`model_init`
        self.inference_support_dict = {
            'coarse_feats': {},
            'fine_feats': {},
        }

    def forward_train(self,
                      query_data: Dict,
                      support_data: Dict,
                      proposals: Optional[List] = None,
                      **kwargs) -> Dict:
        """Forward function for training.

        Args:
            query_data (dict): In most cases, dict of query data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            support_data (dict):  In most cases, dict of support data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            proposals (list): Override rpn proposals with custom proposals.
                Use when `with_rpn` is False. Default: None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        query_img = query_data['img']
        support_img = support_data['img']
        query_feats = self.extract_query_feat(query_img)
        # query_feast[0]为粗粒度特征，query_feats[1]为细粒度特征
        support_feats = self.extract_support_feat(support_img)
        # support_feast[0]为粗粒度特征，support_feast[1]为细粒度特征

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                query_feats,
                copy.deepcopy(query_data['img_metas']),
                copy.deepcopy(query_data['gt_bboxes']),
                gt_labels=None,
                gt_bboxes_ignore=copy.deepcopy(
                    query_data.get('gt_bboxes_ignore', None)),
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            query_feats,
            support_feats,
            proposals=proposal_list,
            query_img_metas=query_data['img_metas'],
            query_gt_bboxes=query_data['gt_bboxes'],
            query_gt_labels=query_data['gt_labels'],
            query_gt_bboxes_ignore=query_data.get('gt_bboxes_ignore', None),
            support_img_metas=support_data['img_metas'],
            support_gt_bboxes=support_data['gt_bboxes'],
            support_gt_labels=support_data['gt_labels'],
            support_gt_bboxes_ignore=support_data.get('gt_bboxes_ignore',
                                                      None),
            **kwargs)
        losses.update(roi_losses)

        return losses

    @auto_fp16(apply_to=('img', ))
    def extract_support_feat(self, img):
        """Extracting features from support data.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            list[Tensor]: Features of input image, each item with shape
                (N, C, H, W).
        """
        feats = self.backbone(img)
        if self.with_neck:
            feats = self.neck(feats)
        return feats

    def forward_model_init(self,
                           img: Tensor,
                           img_metas: List[Dict],
                           gt_bboxes: List[Tensor] = None,
                           gt_labels: List[Tensor] = None,
                           **kwargs):
        """模测试初始化时前向传播函数."""
        # `is_model_init` flag will be reset when forward new data.
        self.is_model_init = False
        assert len(gt_labels) == img.size(
            0), 'Support instance have more than two labels'
        feats = self.extract_support_feat(img)
        rois = bbox2roi(gt_bboxes)
        roi_feats = self.roi_head.extract_support_roi_feats(feats, rois)
        # feats = [feats[i] for i in self.roi_head.input_indices]
        self._forward_saved_support_dict['gt_labels'].extend(gt_labels)
        self._forward_saved_support_dict['coarse_feats'].extend(roi_feats[0])
        self._forward_saved_support_dict['fine_feats'].extend(roi_feats[1])
        return {
            'gt_labels': gt_labels,
            'coarse_feats': roi_feats[0],
            'fine_feats': roi_feats[1]
        }

    def model_init(self):
        """模型测试初始化函数."""
        gt_labels = torch.cat(self._forward_saved_support_dict['gt_labels'])
        coarse_feats = torch.stack(
            self._forward_saved_support_dict['coarse_feats'])
        fine_feats = torch.stack(
            self._forward_saved_support_dict['fine_feats'])
        class_ids = set(gt_labels.data.tolist())
        # init inference support dict
        self.inference_support_dict.clear()
        self.inference_support_dict['coarse_feats'] = dict()
        self.inference_support_dict['fine_feats'] = dict()
        # hcrn roi_head v1
        # coarse_mean_vector = cal_clsvec_init(coarse_feats, gt_labels, len(class_ids))
        # fine_mean_vector = cal_clsvec_init(fine_feats, gt_labels, len(class_ids))
        # for class_id in class_ids:
        #     self.inference_support_dict['coarse_feats'][class_id] = coarse_mean_vector[class_id][None,:,:,:]
        #     self.inference_support_dict['fine_feats'][class_id]   = fine_mean_vector[class_id][None,:,:,:]
        # hcrn roi head v2
        for class_id in class_ids:
            self.inference_support_dict['coarse_feats'][
                class_id] = coarse_feats[gt_labels == class_id].mean([0], True)
            self.inference_support_dict['fine_feats'][class_id] = fine_feats[
                gt_labels == class_id].mean([0], True)
        # set the init flag
        self.is_model_init = True
        # reset support features buff
        for k in self._forward_saved_support_dict.keys():
            self._forward_saved_support_dict[k].clear()

    def simple_test(self,
                    img: Tensor,
                    img_metas: List[Dict],
                    proposals: Optional[List[Tensor]] = None,
                    rescale: bool = False):
        """模型测试函数.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor] | None): override rpn proposals with
                custom proposals. Use when `with_rpn` is False. Default: None.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # assert self.with_bbox, 'Bbox head must be implemented.'
        assert len(img_metas) == 1, 'Only support single image inference.'
        if not self.is_model_init:
            # process the saved support features
            self.model_init()

        query_feats = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test(query_feats, img_metas)
        else:
            proposal_list = proposals
        return self.roi_head.simple_test(
            query_feats,
            copy.deepcopy(self.inference_support_dict),
            proposal_list,
            img_metas,
            rescale=rescale)
