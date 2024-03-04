from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmdet.core import (bbox2result, bbox2roi, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply,
                        multiclass_nms)
from mmdet.models.builder import HEADS, build_loss, build_roi_extractor
from mmdet.models.roi_heads import StandardRoIHead
from torch import Tensor


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


# class RelationNetworkFine(nn.Module):
#     """docstring for RelationNetwork"""

#     def __init__(self, input_size, hidden_size):
#         super(RelationNetworkFine, self).__init__()
#         self.layer1 = nn.Sequential(
#                         nn.Conv2d(256*2, 256, kernel_size=3, padding=0),
#                         nn.BatchNorm2d(256, momentum=1, affine=True),
#                         nn.ReLU(),
#                         nn.MaxPool2d(2))
#         # self.layer2 = nn.Sequential(
#         #                 nn.Conv2d(256, 256, kernel_size=3, padding=1),
#         #                 nn.BatchNorm2d(256, momentum=1, affine=True),
#         #                 nn.ReLU(),
#         #                 nn.MaxPool2d(2))
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         out = self.layer1(x)
#         # out = self.layer2(out)
#         out = out.view(out.size(0), -1)
#         out = F.relu(self.fc1(out))
#         # out = F.sigmoid(self.fc2(out))  # 细粒度分类器在sigmoid的时候计算loss
#         out = self.fc2(out)  # 不进行sigmoid，在计算loss的时候sigmoid
#         return out


class RelationNetworkFine(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


@HEADS.register_module()
class SimpleMetaRoIHead(StandardRoIHead):

    def __init__(self,
                 num_classes,
                 input_indices,
                 bbox_roi_extractor=None,
                 bbox_coder=None,
                 reg_decoded_bbox=False,
                 loss_cls=None,
                 loss_bbox=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_indices = input_indices
        # 1. build roi extractor
        if bbox_roi_extractor is not None:
            self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)

        # 2. build bbox coder
        if bbox_coder is not None:
            self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox

        # 3. build relation network
        # self.relation_network_fine = RelationNetworkFine(256*2*2, 256)
        self.relation_network_fine = RelationNetworkFine(2048, 256)

        # 4. build loss
        if loss_cls is not None:
            self.loss_cls = build_loss(loss_cls)
        if loss_bbox is not None:
            self.loss_bbox = build_loss(loss_bbox)

        self.num_classes = num_classes

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        # bbox_assigner 和sampler从train_cfg里面读取参数
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    def extract_query_roi_feats(self, feats: List[Tensor],
                                rois: Tensor) -> Tensor:
        """提取查询图像的ROI特征."""
        roi_feats = self.bbox_roi_extractor(
            feats[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        return roi_feats

    def extract_support_roi_feats(self, feats: List[Tensor],
                                  rois: Tensor) -> List[Tensor]:
        """提取支持图像的ROI特征."""
        roi_feats = self.bbox_roi_extractor(
            feats[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        else:
            roi_feats = roi_feats
        return roi_feats

    def extract_support_feats(self, feats: List[Tensor]) -> List[Tensor]:
        """提取支持图像的特征."""
        if self.with_shared_head:
            out = self.shared_head(feats)
        else:
            out = feats
        return out

    def forward_train(self,
                      query_feats: List[Tensor],
                      support_feats: List[Tensor],
                      proposals: List[Tensor],
                      query_img_metas: List[Dict],
                      query_gt_bboxes: List[Tensor],
                      query_gt_labels: List[Tensor],
                      support_gt_bboxes: List[Tensor],
                      support_gt_labels: List[Tensor],
                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      **kwargs):
        """训练前向传播函数."""
        # 1.正负样本分配和采样
        sampling_results = []
        num_imgs = len(query_img_metas)
        if query_gt_bboxes_ignore is None:
            query_gt_bboxes_ignore = [None for _ in range(num_imgs)]
        for i in range(num_imgs):  # 对每一张图片进行正负样本分配和采样
            assign_result = self.bbox_assigner.assign(
                proposals[i], query_gt_bboxes[i], query_gt_bboxes_ignore[i],
                query_gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposals[i],
                query_gt_bboxes[i],
                query_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in query_feats])
            sampling_results.append(sampling_result)
        # 2. 计算损失
        losses = dict()
        bbox_results = self.bbox_forward_train(
            query_feats, support_feats, sampling_results, query_img_metas,
            query_gt_bboxes, query_gt_labels, support_gt_bboxes,
            support_gt_labels)
        if bbox_results is not None:
            losses.update(bbox_results)
        return losses

    def bbox_forward_train(self, query_feats: List[Tensor],
                           support_feats: List[Tensor],
                           sampling_results: object,
                           query_img_metas: List[Dict],
                           query_gt_bboxes: List[Tensor],
                           query_gt_labels: List[Tensor],
                           support_gt_bboxes: List[Tensor],
                           support_gt_labels: List[Tensor]) -> Dict:
        """边界框训练前向传播函数."""
        query_rois = bbox2roi([res.bboxes for res in sampling_results
                               ])  # [batch_id, x1, y1, x2, y2]
        support_rois = bbox2roi(support_gt_bboxes)
        query_feats = [query_feats[i] for i in self.input_indices
                       ]  # select coarse and fine feats
        support_feats = [support_feats[i] for i in self.input_indices]
        query_roi_feats = self.extract_query_roi_feats(query_feats, query_rois)
        support_roi_feats = self.extract_support_roi_feats(
            support_feats, support_rois)
        bbox_targets = self.get_targets(sampling_results, query_gt_bboxes,
                                        query_gt_labels, self.train_cfg)
        # 1. prepare tensor
        (query_fine_label, query_fine_label_weights, bbox_targets,
         bbox_weights) = bbox_targets
        query_coarse_feat = query_roi_feats[0]  # query的浅层特征
        support_coarse_feat = support_roi_feats[0]  # support的浅层特征
        query_fine_feat = query_roi_feats[1]  # query的深层特征
        support_fine_feat = support_roi_feats[1]  # support的深层特征
        support_fine_label = torch.tensor(
            support_gt_labels, dtype=torch.int64).cuda()  # support的细粒度标签

        # sort support labels
        _, sort_ids = torch.sort(support_fine_label)
        support_fine_feat = support_fine_feat[sort_ids]
        support_coarse_feat = support_coarse_feat[sort_ids]
        support_fine_label = support_fine_label[sort_ids]

        query_fine_feat_ext = query_fine_feat.unsqueeze(0).repeat(
            support_fine_feat.size(0), 1, 1).transpose(0, 1)
        support_fine_feat_ext = support_fine_feat.unsqueeze(0).repeat(
            query_fine_feat.size(0), 1, 1)
        relation_pairs = torch.cat(
            (support_fine_feat_ext, query_fine_feat_ext),
            dim=2).view(-1,
                        support_fine_feat_ext.size(2) * 2)

        # relations = torch.mm(query_fine_feat, support_fine_feat.t())  # [num_query, num_support]
        relations = self.relation_network_fine(relation_pairs).view(
            query_fine_feat.size(0), support_fine_feat.size(0))
        loss_cls = self.loss_cls(relations, query_fine_label)

        bbox_results = dict()
        bbox_results.update(loss_cls=loss_cls)
        return bbox_results

        # sort support labels
        # _ ,sort_ids = torch.sort(support_fine_label)
        # support_fine_feat = support_fine_feat[sort_ids]
        # support_coarse_feat = support_coarse_feat[sort_ids]
        # support_fine_label = support_fine_label[sort_ids]

        # support_fine_feat = cal_clsvec_init(support_fine_feat, support_fine_label,self.num_classes)

        # 2. relation calculation
        # query_fine_feat_ext = query_fine_feat.unsqueeze(0).repeat(
        #     support_fine_feat.size(0), 1, 1, 1, 1).transpose(0, 1)
        # support_fine_feat_ext = support_fine_feat.unsqueeze(
        #     0).repeat(query_fine_feat.size(0), 1, 1, 1, 1)
        # relation_pairs = torch.cat((support_fine_feat_ext, query_fine_feat_ext),
        #                            dim=2).view(-1, support_fine_feat_ext.size(2)*2, 7, 7)
        # relations = self.relation_network_fine(relation_pairs).view(query_fine_feat.size(0), support_fine_feat.size(0))
        # loss_cls = self.loss_cls(relations, query_fine_label)

        # bbox_results=dict()
        # bbox_results.update(loss_cls=loss_cls)
        # return bbox_results

    def mask_forward_train(self, query_feats: List[Tensor],
                           support_feats: List[Tensor],
                           sampling_results: object,
                           query_img_metas: List[Dict],
                           query_gt_bboxes: List[Tensor],
                           query_gt_labels: List[Tensor],
                           support_gt_labels: List[Tensor]) -> Dict:
        pass

    def simple_test(self,
                    query_feats: List[Tensor],
                    support_feats_dict: Dict,
                    proposal_list: List[Tensor],
                    query_img_metas: List[Dict],
                    rescale: bool = False) -> List[List[np.ndarray]]:
        det_bboxes, det_labels = self.simple_test_bboxes(
            query_feats,
            support_feats_dict,
            query_img_metas,
            proposal_list,
            self.test_cfg,
            rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], self.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results

    def simple_test_bboxes(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            query_img_metas: List[Dict],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """"""
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)

        query_feats = [query_feats[i] for i in self.input_indices]
        rois = bbox2roi(proposals)
        num_boxes = rois.shape[0]
        query_roi_feats = self.extract_query_roi_feats(query_feats, rois)
        # 1. prepare tensor
        query_coarse_feat = query_roi_feats[0]
        query_fine_feat = query_roi_feats[1]
        support_gt_labels = []
        support_coarse_feat = []
        support_fine_feat = []
        for k, v in support_feats_dict['coarse_feats'].items():
            support_gt_labels.append(k)
            support_coarse_feat.append(v)
            support_fine_feat.append(support_feats_dict['fine_feats'][k])
        support_coarse_feat = torch.cat(support_coarse_feat, dim=0)
        support_fine_feat = torch.cat(support_fine_feat, dim=0)
        support_fine_label = torch.tensor(
            support_gt_labels, dtype=torch.int64).cuda()
        support_feats = [support_coarse_feat, support_fine_feat]
        support_feats = self.extract_support_feats(support_feats)
        support_coarse_feat = support_feats[0]
        support_fine_feat = support_feats[1]

        # sort support labels
        _, sort_ids = torch.sort(support_fine_label)
        support_fine_feat = support_fine_feat[sort_ids]
        support_coarse_feat = support_coarse_feat[sort_ids]
        support_fine_label = support_fine_label[sort_ids]

        query_fine_feat_ext = query_fine_feat.unsqueeze(0).repeat(
            support_fine_feat.size(0), 1, 1).transpose(0, 1)
        support_fine_feat_ext = support_fine_feat.unsqueeze(0).repeat(
            query_fine_feat.size(0), 1, 1)
        relation_pairs = torch.cat(
            (support_fine_feat_ext, query_fine_feat_ext),
            dim=2).view(-1,
                        support_fine_feat_ext.size(2) * 2)

        # relations = torch.mm(query_fine_feat, support_fine_feat.t())  # [num_query, num_support]
        relations = self.relation_network_fine(relation_pairs).view(
            query_fine_feat.size(0), support_fine_feat.size(0))
        relations_sigmoid = F.sigmoid(relations)
        bg_score = (1 - relations_sigmoid).cumprod(dim=1)[:, -1].unsqueeze(
            1)  # 背景类别的分数为不是任何类别的概率的乘积
        pred_score = torch.cat([relations_sigmoid, bg_score], dim=1)
        pred_box = proposals[0][:, 0:4]
        # # multiclass_nms 首先会根据score_thr筛选出置信度大于score_thr的预测框，然后根据nms筛选出最终的预测框
        # # 注意在nms的时候只会对同类样本进行nms,而不会对不同类别的样本进行抑制
        det_bboxes, det_labels = multiclass_nms(pred_box, pred_score,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return ([
            det_bboxes,
        ], [
            det_labels,
        ])

        # 2. relation calculation
        # query_fine_feat_ext = query_fine_feat.unsqueeze(0).repeat(
        #     support_fine_feat.size(0), 1, 1, 1, 1).transpose(0, 1)
        # support_fine_feat_ext = support_fine_feat.unsqueeze(
        #     0).repeat(query_fine_feat.size(0), 1, 1, 1, 1)
        # relation_pairs = torch.cat((support_fine_feat_ext, query_fine_feat_ext),
        #                            dim=2).view(-1, support_fine_feat_ext.size(2)*2, 7, 7)
        # relations = self.relation_network_fine(relation_pairs).view(query_fine_feat.size(0), support_fine_feat.size(0))
        # relations_sigmoid = F.sigmoid(relations)
        # bg_score = (1 - relations_sigmoid).cumprod(dim=1)[:,-1].unsqueeze(1) # 背景类别的分数为不是任何类别的概率的乘积
        # pred_score = torch.cat([relations_sigmoid, bg_score],dim=1)

        # pred_box = proposals[0][:,0:4]
        # # multiclass_nms 首先会根据score_thr筛选出置信度大于score_thr的预测框，然后根据nms筛选出最终的预测框
        # # 注意在nms的时候只会对同类样本进行nms,而不会对不同类别的样本进行抑制
        # det_bboxes, det_labels = multiclass_nms(pred_box,
        #                                         pred_score,
        #                                         rcnn_test_cfg.score_thr,
        #                                         rcnn_test_cfg.nms,
        #                                         rcnn_test_cfg.max_per_img)
        # # return det_bboxes, det_labels
        # return ([det_bboxes,], [det_labels,])

        # pred_scores, pred_labels = torch.max(relations_sigmoid.data,1)
        # score_thr = rcnn_test_cfg.score_thr
        # pred_labels[pred_scores < score_thr] = self.num_classes # confidence低于 score_thr的预测为背景
        #

        # if rescale and pred_boxes.size(0) > 0:
        #     scale_factor = pred_boxes.new_tensor(scale_factors[0])
        #     pred_boxes = (pred_boxes.view(pred_boxes.size(0), -1, 4) / scale_factor).view(
        #         pred_boxes.size()[0], -1)

        # # 3. nms

        # return ([det_bboxes,], [det_labels,])
