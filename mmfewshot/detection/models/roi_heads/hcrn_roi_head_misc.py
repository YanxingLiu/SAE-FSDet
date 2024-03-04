from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmdet.core import (bbox2result, bbox2roi, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply,
                        multiclass_nms)
from mmdet.models.builder import (HEADS, build_loss, build_roi_extractor,
                                  build_shared_head)
from mmdet.models.roi_heads import StandardRoIHead
from sklearn.cluster import SpectralClustering
from sympy import Li
from torch import Tensor
from torch.autograd import Variable


def cal_mean_vector(features, labels):
    """
    simple_test: mean_vector[1][0][0] == (features[2][0][0] + features[3][0][0])/2
    """
    class_ids = set(labels.data.tolist())
    mean_vector = torch.zeros([len(class_ids), *features.size()[1:]]).cuda()
    mean_labels = []
    for class_id in class_ids:
        mean_vector[class_id] = features[labels == class_id].mean([0])
        mean_labels.append(class_id)
    mean_label = torch.tensor(mean_labels).cuda()
    return mean_vector, mean_label


def cal_coarse_clusters(features, clsuter_num):
    features_np = features.detach().cpu().numpy()
    num_classes = features_np.shape[0]
    aff_mat = np.zeros([num_classes, num_classes])
    for a in range(0, num_classes - 1):
        for b in range(a + 1, num_classes):
            distance = np.linalg.norm(features_np[a] - features_np[b])
            aff_mat[a, b] = distance
            aff_mat[b, a] = aff_mat[a, b]
    beta = 0.1
    aff_mat = np.exp(-beta * aff_mat / aff_mat.std())
    for i in range(num_classes):
        aff_mat[i, i] = 0.0001
    sc = SpectralClustering(
        clsuter_num,
        n_init=10,
        affinity='precomputed',
        n_neighbors=10,
        assign_labels='kmeans')
    groups = sc.fit_predict(aff_mat)

    return groups


def get_coare_real_labels(labels, groups):
    """获取细粒度类别的粗粒度标签."""
    coarse_labels = []
    for i in range(len(labels)):
        labels_value = labels[i]
        if labels_value == len(groups):
            # 背景类别
            coarse_value = max(groups) + 1
        else:
            # 非背景类别将细粒度标签转换为细粒度标签
            coarse_value = groups[labels_value]
        coarse_labels.append(coarse_value)

    return coarse_labels


def get_new_labels(labels, depend_labels):
    new_labels = []
    for c in labels:
        label = []
        for j in range(len(depend_labels)):
            if c == depend_labels[j]:
                label.append(1)
            else:
                label.append(0)
        new_labels.append(label)
    return new_labels


class RelationNetworkCoarse(nn.Module):
    """docstring for RelationNetwork."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(256 * 2, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256, momentum=1, affine=True), nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256, momentum=1, affine=True), nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256, momentum=1, affine=True), nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        # out = F.sigmoid(self.fc2(out))
        out = self.fc2(out)  # 不进行sigmoid，在计算loss的时候sigmoid
        return out


class RelationNetworkFine(nn.Module):
    """docstring for RelationNetwork."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(256 * 2, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256, momentum=1, affine=True), nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256, momentum=1, affine=True), nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))  # 细粒度分类器在sigmoid的时候计算loss
        # out = self.fc2(out)  # 不进行sigmoid，在计算loss的时候sigmoid
        return out


class RelationNetworkCoarseV3(nn.Module):
    """docstring for RelationNetwork."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(256 * 2, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256, momentum=1, affine=True), nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256, momentum=1, affine=True), nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        # out = F.sigmoid(self.fc2(out))  # 细粒度分类器在sigmoid的时候计算loss
        out = self.fc2(out)  # 不进行sigmoid，在计算loss的时候sigmoid
        return out


class RelationNetworkFineV3(nn.Module):
    """docstring for RelationNetwork."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(256 * 2, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256, momentum=1, affine=True), nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))  # 细粒度分类器在sigmoid的时候计算loss
        # out = self.fc2(out)  # 不进行sigmoid，在计算loss的时候sigmoid
        return out


class RelationNetworkFine2FC(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class RelationNetworkCoarse2FC(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.sigmoid(self.fc2(out))
        return out


@HEADS.register_module()
class HCRNRoIHeadV2(StandardRoIHead):

    def __init__(self,
                 num_classes,
                 num_clusters,
                 bbox_roi_extractor=None,
                 bbox_coder=None,
                 reg_decoded_bbox=False,
                 loss_cls_coarse=None,
                 loss_cls_fine=None,
                 loss_bbox=None,
                 **kwargs):
        super().__init__(**kwargs)
        # 1. build roi extractor
        if bbox_roi_extractor is not None:
            self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)

        # 2. build bbox coder
        if bbox_coder is not None:
            self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox

        # 3. build relation network
        self.relation_coarse_network = RelationNetworkCoarse(256, 256)
        self.relation_network_fine = RelationNetworkFine(256 * 2 * 2, 256)

        # 4. build loss
        if loss_cls_coarse is not None:
            self.loss_cls_coarse = build_loss(loss_cls_coarse)
        if loss_cls_fine is not None:
            self.loss_cls_fine = build_loss(loss_cls_fine)
        if loss_bbox is not None:
            self.loss_bbox = build_loss(loss_bbox)
        self.mse = nn.MSELoss().cuda()

        self.num_classes = num_classes
        self.num_clusters = num_clusters

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
            query_gt_bboxes, query_gt_labels, support_gt_labels)
        if bbox_results is not None:
            losses.update(bbox_results)
        return losses

    def bbox_forward_train(self, query_feats: List[Tensor],
                           support_feats: List[Tensor],
                           sampling_results: object,
                           query_img_metas: List[Dict],
                           query_gt_bboxes: List[Tensor],
                           query_gt_labels: List[Tensor],
                           support_gt_labels: List[Tensor]) -> Dict:
        """边界框训练前向传播函数."""
        query_rois = bbox2roi([res.bboxes for res in sampling_results
                               ])  # [batch_id, x1, y1, x2, y2]
        query_roi_feats = self.extract_query_roi_feats(query_feats, query_rois)
        bbox_targets = self.get_targets(sampling_results, query_gt_bboxes,
                                        query_gt_labels, self.train_cfg)
        # 1. prepare tensor
        (query_fine_label, query_fine_label_weights, bbox_targets,
         bbox_weights) = bbox_targets
        query_coarse_feat = query_roi_feats[0]  # query的浅层特征
        support_coarse_feat = support_feats[0]  # support的浅层特征
        query_fine_feat = query_roi_feats[1]  # query的深层特征
        support_fine_feat = support_feats[1]  # support的深层特征
        support_fine_label = torch.tensor(
            support_gt_labels, dtype=torch.int64).cuda()  # support的细粒度标签
        # # sort support labels
        _, sort_ids = torch.sort(support_fine_label)
        support_fine_feat = support_fine_feat[sort_ids]
        support_coarse_feat = support_coarse_feat[sort_ids]
        support_fine_label = support_fine_label[sort_ids]

        # 2.calculate support cluster center
        support_fine_mean_feat, support_fine_mean_label = cal_mean_vector(
            support_fine_feat, support_fine_label)  # 计算所有support样本的细粒度特征均值
        support_coarse_mean_feat, _ = cal_mean_vector(
            support_coarse_feat, support_fine_label)  # 计算所有support样本的粗粒度特征均值
        # support粗粒度特征聚类，得到聚类标签
        # support_coarse_cluster_fine_label  # size:[15] 代表每一个细粒度类别的粗粒度标签
        support_coarse_cluster_fine_label = cal_coarse_clusters(
            support_coarse_mean_feat, self.num_clusters)
        # 将粗粒度特征进行聚类，得到聚类中心
        # support_coarse_cluster_feat: shape [5, 256, 28, 28] support 聚类的粗粒度特征
        # support_coarse_cluster_coarse_label: shape [5]   support 聚类的粗粒度标签
        support_coarse_cluster_feat, support_coarse_cluster_coarse_label = cal_mean_vector(
            support_coarse_mean_feat,
            support_coarse_cluster_fine_label)  # size:[5, 256, 28, 28]

        # 3.coarse classification
        query_coarse_feat_exts = query_coarse_feat.unsqueeze(0).repeat(
            support_coarse_cluster_feat.size(0), 1, 1, 1, 1).transpose(0, 1)
        support_coarse_cluster_feat_exts = support_coarse_cluster_feat.unsqueeze(
            0).repeat(query_coarse_feat.size(0), 1, 1, 1, 1)
        coarse_relation_pairs = torch.cat(
            (support_coarse_cluster_feat_exts, query_coarse_feat_exts),
            2).view(-1,
                    support_coarse_cluster_feat_exts.size(2) * 2, 28, 28)
        coarse_relations = self.relation_coarse_network(
            coarse_relation_pairs).view(-1, self.num_clusters)
        query_coarse_labels = get_coare_real_labels(
            query_fine_label,
            support_coarse_cluster_fine_label)  # query 的真实粗分类标签
        loss_coarse_cls = self.loss_cls_coarse(
            coarse_relations,
            torch.tensor(query_coarse_labels).cuda(), query_fine_label_weights)

        coarse_relations_sigmoid = coarse_relations.sigmoid()
        coarse_bg_score = (1 - coarse_relations_sigmoid).cumprod(
            dim=1)[:, -1].unsqueeze(1)
        pred_coarse_score = torch.cat(
            [coarse_relations_sigmoid, coarse_bg_score], dim=1)
        query_predict_coarse_scores, query_predict_coarse_labels = torch.max(
            pred_coarse_score.data, 1)  #query_set的粗分类标签

        # 4. fine classification
        sample_parents = {}  # 记录每一个超类的所有细粒度样本的索引
        for i in range(support_fine_mean_feat.size(0)):
            coarse_label = support_coarse_cluster_fine_label[i]  # 获取粗粒度类别
            if coarse_label not in sample_parents:
                sample_parents[coarse_label] = [i]
            else:
                sample_parents[coarse_label].append(i)
        query_predict_fine_labels = []
        query_avg_factor = 0
        for i in range(query_fine_label.size(0)):  # 样本数目
            batch_label = query_fine_label[i].view(-1, 1)
            if batch_label == self.num_classes:  # 只对正样本进行细分类，如果是背景类的话，在粗分类的时候就应该被筛选掉，对背景类预测出来的细分类标签没有意义，计算损失的话可能反而会降低检测结果
                continue
            else:
                batch_fine_feature = query_fine_feat[i, :]
                predict_batch_coarse_label = query_predict_coarse_labels[
                    i].item()  # query预测的粗分类标签
                if query_predict_coarse_labels[
                        i] == self.num_clusters:  # 预测为背景类别，不需要进行细粒度分类
                    continue
                else:
                    query_avg_factor += 1
                    bro_batch_labels = sample_parents[
                        predict_batch_coarse_label]
                    bro_num = len(bro_batch_labels)
                    batch_fine_feature_ext = batch_fine_feature.unsqueeze(
                        0).repeat(bro_num, 1, 1, 1)
                    sample_fine_features_ext = support_fine_mean_feat[
                        bro_batch_labels, :]
                    fine_pairs = torch.cat(
                        (sample_fine_features_ext, batch_fine_feature_ext),
                        1).view(-1,
                                batch_fine_feature_ext.size(1) * 2, 14, 14)
                    fine_relations = self.relation_network_fine(
                        fine_pairs).view(-1, bro_num)
                    new_predict_batch_fine_label = Variable(
                        torch.Tensor(
                            get_new_labels(batch_label,
                                           bro_batch_labels)).view(
                                               -1, bro_num)
                    ).cuda()  # one hot 向量，代表batch_label是否在bro_batch_labels中
                    if query_avg_factor == 1:
                        loss_fine_cls = self.mse(fine_relations,
                                                 new_predict_batch_fine_label)
                    else:
                        loss_fine_cls += self.mse(
                            fine_relations, new_predict_batch_fine_label)
                    _, predict_fine_label = torch.max(fine_relations.data, 1)
                    query_predict_fine_labels.append(
                        bro_batch_labels[predict_fine_label[0]])
        if query_avg_factor == 0:  # query中没有正样本
            bbox_results = dict()
            bbox_results.update(loss_coarse_cls=loss_coarse_cls)
            return bbox_results
        else:
            loss_fine_cls = 0.1 * loss_fine_cls / query_avg_factor
            bbox_results = dict()
            bbox_results.update(
                loss_coarse_cls=loss_coarse_cls, loss_fine_cls=loss_fine_cls)
            return bbox_results

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
        # sort support labels
        _, sort_ids = torch.sort(support_fine_label)
        support_fine_feat = support_fine_feat[sort_ids]
        support_coarse_feat = support_coarse_feat[sort_ids]
        support_fine_label = support_fine_label[sort_ids]

        # 2.support cluster
        self.num_clusters = 5
        support_fine_mean_feat, support_fine_mean_label = cal_mean_vector(
            support_fine_feat, support_fine_label)  # 计算所有support样本的细粒度特征均值
        support_coarse_mean_feat, _ = cal_mean_vector(
            support_coarse_feat, support_fine_label)  # 计算所有support样本的粗粒度特征均值
        # support粗粒度特征聚类，得到聚类标签
        # support_coarse_cluster_fine_label  # size:[15] 代表每一个细粒度类别的粗粒度标签
        support_coarse_cluster_fine_label = cal_coarse_clusters(
            support_coarse_mean_feat, self.num_clusters)
        # 将粗粒度特征进行聚类，得到聚类中心
        # support_coarse_cluster_feat: shape [5, 256, 28, 28] support 聚类的粗粒度特征
        # support_coarse_cluster_coarse_label: shape [5]   support 聚类的粗粒度标签
        support_coarse_cluster_feat, support_coarse_cluster_coarse_label = cal_mean_vector(
            support_coarse_mean_feat,
            support_coarse_cluster_fine_label)  # size:[5, 256, 28, 28]

        # 3. coarse clssfication
        query_coarse_feat_exts = query_coarse_feat.unsqueeze(0).repeat(
            support_coarse_cluster_feat.size(0), 1, 1, 1, 1).transpose(0, 1)
        support_coarse_cluster_feat_exts = support_coarse_cluster_feat.unsqueeze(
            0).repeat(query_coarse_feat.size(0), 1, 1, 1, 1)
        coarse_relation_pairs = torch.cat(
            (support_coarse_cluster_feat_exts, query_coarse_feat_exts),
            2).view(-1,
                    support_coarse_cluster_feat_exts.size(2) * 2, 28, 28)
        coarse_relations = self.relation_coarse_network(
            coarse_relation_pairs).view(-1, self.num_clusters)
        coarse_relations_sigmoid = coarse_relations.sigmoid()
        coarse_bg_score = (1 - coarse_relations_sigmoid).cumprod(
            dim=1)[:, -1].unsqueeze(1)
        pred_coarse_score = torch.cat(
            [coarse_relations_sigmoid, coarse_bg_score], dim=1)
        query_predict_coarse_score, query_predict_coarse_label = torch.max(
            pred_coarse_score.data, 1)  #query_set的粗分类标签

        # 4. fine classification
        sample_parents = {}  # 记录每一个超类的所有细粒度样本的索引
        for i in range(support_fine_mean_feat.size(0)):
            coarse_label = support_coarse_cluster_fine_label[i]  # 获取粗粒度类别
            if coarse_label not in sample_parents:
                sample_parents[coarse_label] = [i]
            else:
                sample_parents[coarse_label].append(i)
        query_predict_fine_labels = []
        query_predict_fine_scores = []
        for i in range(query_fine_feat.size(0)):
            predict_batch_coarse_label = query_predict_coarse_label[i].item()
            if predict_batch_coarse_label == self.num_clusters:
                query_predict_fine_labels.append(
                    self.num_classes
                )  # 如果预测出粗分类类别为背景类别，那直接不进行细分类，而是将细分类标签直接设置为背景类别
                query_predict_fine_scores.append(
                    query_predict_coarse_score[i].item())  # 粗分类的背景类别分数作为细分类的分数
            else:
                batch_fine_feature = query_fine_feat[i, :]  # 获取query_set的细粒度特征
                bro_batch_labels = sample_parents[
                    predict_batch_coarse_label]  # 获取同一个粗分类分支下所有的细分类类别
                bro_num = len(bro_batch_labels)
                batch_fine_feature_ext = batch_fine_feature.unsqueeze(
                    0).repeat(bro_num, 1, 1, 1)
                sample_fine_features_ext = support_fine_mean_feat[
                    bro_batch_labels, :]  # 获取同一个粗分类分支下所有的细分类特征
                fine_pairs = torch.cat(
                    (sample_fine_features_ext, batch_fine_feature_ext),
                    1).view(-1,
                            batch_fine_feature_ext.size(1) * 2, 14, 14)
                fine_relations = self.relation_network_fine(fine_pairs).view(
                    -1, bro_num)
                predict_fine_score, predict_fine_label = torch.max(
                    fine_relations.data, 1)
                query_predict_fine_labels.append(
                    bro_batch_labels[predict_fine_label[0]])
                query_predict_fine_scores.append(
                    query_predict_coarse_score[i] *
                    predict_fine_score)  # 粗分类的分数乘以细分类的分数作为最终的分数

        pred_score_final = torch.zeros(
            query_fine_feat.size(0), self.num_classes + 1).cuda()
        # 转为one-hot的向量以进行nms
        for i in range(pred_score_final.size(0)):
            pred_score_final[
                i, query_predict_fine_labels[i]] = query_predict_fine_scores[i]
        pred_box = proposals[0][:, 0:4]
        det_bboxes, det_labels = multiclass_nms(pred_box, pred_score_final,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return ([
            det_bboxes,
        ], [
            det_labels,
        ])


@HEADS.register_module()
class HCRNRoIHeadV3(StandardRoIHead):

    def __init__(self,
                 num_classes,
                 num_clusters,
                 input_indices=None,
                 bbox_roi_extractor=None,
                 bbox_coder=None,
                 shared_head=None,
                 reg_decoded_bbox=False,
                 loss_cls_coarse=None,
                 loss_cls_fine_fraction=None,
                 loss_cls_fine=None,
                 loss_bbox=None,
                 **kwargs):
        super().__init__(**kwargs)
        # 1. build roi extractor
        self.input_indices = input_indices  # 用于指示输入的是哪个支路的特征
        if bbox_roi_extractor is not None:
            self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)

        # 2. build bbox coder
        if bbox_coder is not None:
            self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox

        # 3. build shared head
        if shared_head is not None:
            self.shared_head = build_shared_head(shared_head)
        else:
            self.shared_head = None

        # 4. build relation network
        self.relation_coarse_network = RelationNetworkCoarseV3(
            256 * 2 * 2, 256)
        self.relation_network_fine = RelationNetworkFineV3(256 * 2 * 2, 256)
        # self.relation_coarse_network = RelationNetworkCoarse2FC(2048*2,1024)
        # self.relation_network_fine = RelationNetworkFine2FC(1024*2,512)

        # 5. build loss
        if loss_cls_coarse is not None:
            self.loss_cls_coarse = build_loss(loss_cls_coarse)
        if loss_cls_fine is not None:
            self.loss_cls_fine_fraction = loss_cls_fine_fraction  # 细粒度分类的比例系数
            self.loss_cls_fine = build_loss(loss_cls_fine)
        if loss_bbox is not None:
            self.loss_bbox = build_loss(loss_bbox)
        self.mse = nn.MSELoss().cuda()

        self.num_classes = num_classes
        self.num_clusters = num_clusters

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
        if self.input_indices is not None:
            query_feats = [query_feats[i]
                           for i in self.input_indices]  # 筛选粗粒度特征和细粒度特征
            support_feats = [support_feats[i] for i in self.input_indices]
        query_rois = bbox2roi([res.bboxes for res in sampling_results
                               ])  # [batch_id, x1, y1, x2, y2]
        support_rois = bbox2roi(support_gt_bboxes)
        query_roi_feats = self.extract_query_roi_feats(query_feats, query_rois)
        support_roi_feats = self.extract_support_roi_feats(
            support_feats, support_rois)
        bbox_targets = self.get_targets(sampling_results, query_gt_bboxes,
                                        query_gt_labels, self.train_cfg)
        # 1. prepare tensor
        (query_fine_label, query_fine_label_weights, bbox_targets,
         bbox_weights) = bbox_targets
        query_coarse_feat = query_roi_feats[0]  # query的浅层特征
        query_fine_feat = query_roi_feats[1]  # query的深层特征
        support_coarse_feat = support_roi_feats[0]  # support的浅层特征
        support_fine_feat = support_roi_feats[1]  # support的深层特征
        support_fine_label = torch.tensor(
            support_gt_labels, dtype=torch.int64).cuda()  # support的细粒度标签
        # sort support labels
        _, sort_ids = torch.sort(support_fine_label)
        support_fine_feat = support_fine_feat[sort_ids]
        support_coarse_feat = support_coarse_feat[sort_ids]
        support_fine_label = support_fine_label[sort_ids]

        # 2.calculate support cluster center
        support_fine_mean_feat, support_fine_mean_label = cal_mean_vector(
            support_fine_feat, support_fine_label)  # 计算所有support样本的细粒度特征均值
        support_coarse_mean_feat, _ = cal_mean_vector(
            support_coarse_feat, support_fine_label)  # 计算所有support样本的粗粒度特征均值
        # support粗粒度特征聚类，得到聚类标签
        # support_coarse_cluster_fine_label  # size:[15] 代表每一个细粒度类别的粗粒度标签
        support_coarse_cluster_fine_label = cal_coarse_clusters(
            support_coarse_mean_feat, self.num_clusters)
        # 将粗粒度特征进行聚类，得到聚类中心
        # support_coarse_cluster_feat: shape [5, 256, 28, 28] support 聚类的粗粒度特征
        # support_coarse_cluster_coarse_label: shape [5]   support 聚类的粗粒度标签
        support_coarse_cluster_feat, support_coarse_cluster_coarse_label = cal_mean_vector(
            support_coarse_mean_feat,
            support_coarse_cluster_fine_label)  # size:[5, 256, 28, 28]

        # 3.coarse classification
        if self.with_shared_head:
            query_coarse_feat_exts = query_coarse_feat.unsqueeze(0).repeat(
                support_coarse_cluster_feat.size(0), 1, 1).transpose(0, 1)
            support_coarse_cluster_feat_exts = support_coarse_cluster_feat.unsqueeze(
                0).repeat(query_coarse_feat.size(0), 1, 1)
            coarse_relation_pairs = torch.cat(
                (support_coarse_cluster_feat_exts, query_coarse_feat_exts),
                2).view(-1,
                        support_coarse_cluster_feat_exts.size(2) * 2)
            coarse_relations = self.relation_coarse_network(
                coarse_relation_pairs).view(-1, self.num_clusters)
            query_coarse_labels = get_coare_real_labels(
                query_fine_label,
                support_coarse_cluster_fine_label)  # query 的真实粗分类标签
            loss_coarse_cls = self.loss_cls_coarse(
                coarse_relations,
                torch.tensor(query_coarse_labels).cuda(),
                query_fine_label_weights)
        else:
            query_coarse_feat_exts = query_coarse_feat.unsqueeze(0).repeat(
                support_coarse_cluster_feat.size(0), 1, 1, 1,
                1).transpose(0, 1)
            support_coarse_cluster_feat_exts = support_coarse_cluster_feat.unsqueeze(
                0).repeat(query_coarse_feat.size(0), 1, 1, 1, 1)
            coarse_relation_pairs = torch.cat(
                (support_coarse_cluster_feat_exts, query_coarse_feat_exts),
                2).view(-1,
                        support_coarse_cluster_feat_exts.size(2) * 2, 14, 14)
            coarse_relations = self.relation_coarse_network(
                coarse_relation_pairs).view(-1, self.num_clusters)
            query_coarse_labels = get_coare_real_labels(
                query_fine_label,
                support_coarse_cluster_fine_label)  # query 的真实粗分类标签
            loss_coarse_cls = self.loss_cls_coarse(
                coarse_relations,
                torch.tensor(query_coarse_labels).cuda(),
                query_fine_label_weights)

        coarse_relations_sigmoid = coarse_relations.sigmoid()
        coarse_bg_score = (1 - coarse_relations_sigmoid).cumprod(
            dim=1)[:, -1].unsqueeze(1)
        pred_coarse_score = torch.cat(
            [coarse_relations_sigmoid, coarse_bg_score], dim=1)
        query_predict_coarse_scores, query_predict_coarse_labels = torch.max(
            pred_coarse_score.data, 1)  #query_set的粗分类标签

        # 4. fine classification
        sample_parents = {}  # 记录每一个超类的所有细粒度样本的索引
        for i in range(support_fine_mean_feat.size(0)):
            coarse_label = support_coarse_cluster_fine_label[i]  # 获取粗粒度类别
            if coarse_label not in sample_parents:
                sample_parents[coarse_label] = [i]
            else:
                sample_parents[coarse_label].append(i)
        query_predict_fine_labels = []
        query_avg_factor = 0
        for i in range(query_fine_label.size(0)):  # 样本数目
            batch_label = query_fine_label[i].view(-1, 1)
            if batch_label == self.num_classes:  # 只对正样本进行细分类，如果是背景类的话，在粗分类的时候就应该被筛选掉，对背景类预测出来的细分类标签没有意义，计算损失的话可能反而会降低检测结果
                continue
            else:
                batch_fine_feature = query_fine_feat[i, :]
                predict_batch_coarse_label = query_predict_coarse_labels[
                    i].item()  # query预测的粗分类标签
                if query_predict_coarse_labels[
                        i] == self.num_clusters:  # 预测为背景类别，不需要进行细粒度分类
                    continue
                else:
                    query_avg_factor += 1
                    bro_batch_labels = sample_parents[
                        predict_batch_coarse_label]
                    bro_num = len(bro_batch_labels)
                    if self.shared_head is not None:
                        batch_fine_feature_ext = batch_fine_feature.unsqueeze(
                            0).repeat(bro_num, 1)
                        sample_fine_features_ext = support_fine_mean_feat[
                            bro_batch_labels, :]
                        fine_pairs = torch.cat(
                            (sample_fine_features_ext, batch_fine_feature_ext),
                            1).view(-1,
                                    batch_fine_feature_ext.size(1) * 2)
                    else:
                        batch_fine_feature_ext = batch_fine_feature.unsqueeze(
                            0).repeat(bro_num, 1, 1, 1)
                        sample_fine_features_ext = support_fine_mean_feat[
                            bro_batch_labels, :]
                        fine_pairs = torch.cat(
                            (sample_fine_features_ext, batch_fine_feature_ext),
                            1).view(-1,
                                    batch_fine_feature_ext.size(1) * 2, 7, 7)
                    fine_relations = self.relation_network_fine(
                        fine_pairs).view(-1, bro_num)
                    new_predict_batch_fine_label = Variable(
                        torch.Tensor(
                            get_new_labels(batch_label,
                                           bro_batch_labels)).view(
                                               -1, bro_num)
                    ).cuda()  # one hot 向量，代表batch_label是否在bro_batch_labels中
                    if query_avg_factor == 1:
                        loss_fine_cls = self.mse(fine_relations,
                                                 new_predict_batch_fine_label)
                    else:
                        loss_fine_cls += self.mse(
                            fine_relations, new_predict_batch_fine_label)
                    _, predict_fine_label = torch.max(fine_relations.data, 1)
                    query_predict_fine_labels.append(
                        bro_batch_labels[predict_fine_label[0]])
        if query_avg_factor == 0:  # query中没有正样本
            bbox_results = dict()
            bbox_results.update(loss_coarse_cls=loss_coarse_cls)
            return bbox_results
        else:
            loss_fine_cls = self.loss_cls_fine_fraction * loss_fine_cls / query_avg_factor
            bbox_results = dict()
            bbox_results.update(
                loss_coarse_cls=loss_coarse_cls, loss_fine_cls=loss_fine_cls)
            return bbox_results

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

        rois = bbox2roi(proposals)
        # num_boxes = rois.shape[0]
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
        # sort support labels
        _, sort_ids = torch.sort(support_fine_label)
        support_fine_feat = support_fine_feat[sort_ids]
        support_coarse_feat = support_coarse_feat[sort_ids]
        support_fine_label = support_fine_label[sort_ids]

        # 2.support cluster
        support_fine_mean_feat, support_fine_mean_label = cal_mean_vector(
            support_fine_feat, support_fine_label)  # 计算所有support样本的细粒度特征均值
        support_coarse_mean_feat, _ = cal_mean_vector(
            support_coarse_feat, support_fine_label)  # 计算所有support样本的粗粒度特征均值
        # support粗粒度特征聚类，得到聚类标签
        # support_coarse_cluster_fine_label  # size:[15] 代表每一个细粒度类别的粗粒度标签
        support_coarse_cluster_fine_label = cal_coarse_clusters(
            support_coarse_mean_feat, self.num_clusters)
        # 将粗粒度特征进行聚类，得到聚类中心
        # support_coarse_cluster_feat: shape [5, 256, 28, 28] support 聚类的粗粒度特征
        # support_coarse_cluster_coarse_label: shape [5]   support 聚类的粗粒度标签
        support_coarse_cluster_feat, support_coarse_cluster_coarse_label = cal_mean_vector(
            support_coarse_mean_feat,
            support_coarse_cluster_fine_label)  # size:[5, 256, 28, 28]

        # 3. coarse clssfication
        if self.with_shared_head:
            query_coarse_feat_exts = query_coarse_feat.unsqueeze(0).repeat(
                support_coarse_cluster_feat.size(0), 1, 1).transpose(0, 1)
            support_coarse_cluster_feat_exts = support_coarse_cluster_feat.unsqueeze(
                0).repeat(query_coarse_feat.size(0), 1, 1)
            coarse_relation_pairs = torch.cat(
                (support_coarse_cluster_feat_exts, query_coarse_feat_exts),
                2).view(-1,
                        support_coarse_cluster_feat_exts.size(2) * 2)
        else:
            query_coarse_feat_exts = query_coarse_feat.unsqueeze(0).repeat(
                support_coarse_cluster_feat.size(0), 1, 1, 1,
                1).transpose(0, 1)
            support_coarse_cluster_feat_exts = support_coarse_cluster_feat.unsqueeze(
                0).repeat(query_coarse_feat.size(0), 1, 1, 1, 1)
            coarse_relation_pairs = torch.cat(
                (support_coarse_cluster_feat_exts, query_coarse_feat_exts),
                2).view(-1,
                        support_coarse_cluster_feat_exts.size(2) * 2, 14, 14)
        coarse_relations = self.relation_coarse_network(
            coarse_relation_pairs).view(-1, self.num_clusters)
        coarse_relations_sigmoid = coarse_relations.sigmoid()
        coarse_bg_score = (1 - coarse_relations_sigmoid).cumprod(
            dim=1)[:, -1].unsqueeze(1)
        pred_coarse_score = torch.cat(
            [coarse_relations_sigmoid, coarse_bg_score], dim=1)
        query_predict_coarse_score, query_predict_coarse_label = torch.max(
            pred_coarse_score.data, 1)  #query_set的粗分类标签

        # 4. fine classification
        sample_parents = {}  # 记录每一个超类的所有细粒度样本的索引
        for i in range(support_fine_mean_feat.size(0)):
            coarse_label = support_coarse_cluster_fine_label[i]  # 获取粗粒度类别
            if coarse_label not in sample_parents:
                sample_parents[coarse_label] = [i]
            else:
                sample_parents[coarse_label].append(i)
        query_predict_fine_labels = []
        query_predict_fine_scores = []
        for i in range(query_fine_feat.size(0)):
            predict_batch_coarse_label = query_predict_coarse_label[i].item()
            if predict_batch_coarse_label == self.num_clusters:
                query_predict_fine_labels.append(
                    self.num_classes
                )  # 如果预测出粗分类类别为背景类别，那直接不进行细分类，而是将细分类标签直接设置为背景类别
                query_predict_fine_scores.append(
                    query_predict_coarse_score[i].item())  # 粗分类的背景类别分数作为细分类的分数
            else:
                batch_fine_feature = query_fine_feat[i, :]  # 获取query_set的细粒度特征
                bro_batch_labels = sample_parents[
                    predict_batch_coarse_label]  # 获取同一个粗分类分支下所有的细分类类别
                bro_num = len(bro_batch_labels)
                if self.shared_head is not None:
                    batch_fine_feature_ext = batch_fine_feature.unsqueeze(
                        0).repeat(bro_num, 1)
                    sample_fine_features_ext = support_fine_mean_feat[
                        bro_batch_labels, :]  # 获取同一个粗分类分支下所有的细分类特征
                    fine_pairs = torch.cat(
                        (sample_fine_features_ext, batch_fine_feature_ext),
                        1).view(-1,
                                batch_fine_feature_ext.size(1) * 2)
                else:
                    batch_fine_feature_ext = batch_fine_feature.unsqueeze(
                        0).repeat(bro_num, 1, 1, 1)
                    sample_fine_features_ext = support_fine_mean_feat[
                        bro_batch_labels, :]  # 获取同一个粗分类分支下所有的细分类特征
                    fine_pairs = torch.cat(
                        (sample_fine_features_ext, batch_fine_feature_ext),
                        1).view(-1,
                                batch_fine_feature_ext.size(1) * 2, 7, 7)
                fine_relations = self.relation_network_fine(fine_pairs).view(
                    -1, bro_num)
                predict_fine_score, predict_fine_label = torch.max(
                    fine_relations.data, 1)
                query_predict_fine_labels.append(
                    bro_batch_labels[predict_fine_label[0]])
                query_predict_fine_scores.append(
                    query_predict_coarse_score[i] *
                    predict_fine_score)  # 粗分类的分数乘以细分类的分数作为最终的分数

        pred_score_final = torch.zeros(
            query_fine_feat.size(0), self.num_classes + 1).cuda()
        # 转为one-hot的向量以进行nms
        for i in range(pred_score_final.size(0)):
            pred_score_final[
                i, query_predict_fine_labels[i]] = query_predict_fine_scores[i]
        pred_box = proposals[0][:, 0:4]
        det_bboxes, det_labels = multiclass_nms(pred_box, pred_score_final,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return ([
            det_bboxes,
        ], [
            det_labels,
        ])
