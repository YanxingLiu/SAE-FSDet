import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import ConfigDict
from mmdet.core import (bbox2result, bbox2roi, build_bbox_coder, multi_apply,
                        multiclass_nms)
from mmdet.models.builder import (HEADS, build_loss, build_neck,
                                  build_roi_extractor)
from mmdet.models.roi_heads import StandardRoIHead
from sklearn.cluster import SpectralClustering
from torch import Tensor
from torch.autograd import Variable


def cal_clsvec_init(data, fine_labels, num_class):  # 90*64*19*19
    """初始化每个超类的均值,可能是对均值特征向量缩放了一个倍数,同时它也是求的通道维度的均值，以[28,28]作为模板进行匹配。"""
    class_vec = np.zeros(
        [num_class, data.shape[1], data.shape[2], data.shape[3]])
    for i in range(num_class):
        idx = [j for j, x in enumerate(fine_labels)
               if x == i]  # 第idx个样本的标签为i，idx为一个列表，列表保存了所有标签为i的样本的索引
        sigma_cls = np.zeros(
            [data.shape[0], data.shape[1], data.shape[2], data.shape[3]])
        for m in range(len(idx)):
            s = data[idx[m], :, :, :]  # 第m个特征向量 shape为[C,H,W]
            avg_s = sum(s) / len(
                s
            )  # sum(s)为每个通道的特征的和，shape为[H,W]，len(s)为通道数 shape为C，所以这个avg_s是求通道特征均值
            sigma_cls += avg_s
        vec = sum(sigma_cls) / len(idx)  # 求样本均值
        class_vec[i] = vec

    return class_vec


def gen_superclass(data, fine_labels, num_class, num_clusters):
    class_vec = cal_clsvec_init(data, fine_labels, num_class)  # 获取每个类的表示向量
    aff_mat = np.zeros([num_class, num_class])
    for a in range(0, num_class - 1):
        for b in range(a + 1, num_class):
            distance = np.linalg.norm(class_vec[a] - class_vec[b])
            aff_mat[a, b] = distance
            aff_mat[b, a] = aff_mat[a, b]
    beta = 0.1
    # aff_mat = normalize_2darray(0.0, 1.0, aff_mat)
    aff_mat = np.exp(-beta * aff_mat / aff_mat.std())
    for i in range(num_class):
        aff_mat[i, i] = 0.0001
    sc = SpectralClustering(
        num_clusters,
        n_init=10,
        affinity='precomputed',
        n_neighbors=10,
        assign_labels='kmeans')
    groups = sc.fit_predict(aff_mat)  # 将15个类聚集成五个超类

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
    """"""
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


@HEADS.register_module()
class HCRNRoIHead(StandardRoIHead):

    def __init__(self,
                 num_classes,
                 num_clusters,
                 bbox_roi_extractor=None,
                 bbox_coder=None,
                 reg_decoded_bbox=False,
                 loss_cls_coarse=None,
                 loss_cls_fine=None,
                 loss_bbox=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.relation_coarse_network = RelationNetworkCoarse(256, 256)
        self.relation_fine_network = RelationNetworkFine(256 * 2 * 2, 256)
        if bbox_roi_extractor is not None:
            self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        if loss_cls_coarse is not None:
            self.loss_cls_coarse = build_loss(loss_cls_coarse)
        if loss_cls_fine is not None:
            self.loss_cls_fine = build_loss(loss_cls_fine)
        if loss_bbox is not None:
            self.loss_box = build_loss(loss_bbox)
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.mse = nn.MSELoss().cuda()
        self.coarse_thr = 0.1

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

    def forward_train(self,
                      query_feats: List[Tensor],
                      support_feats: List[Tensor],
                      proposals: List[Tensor],
                      query_img_metas: List[Dict],
                      query_gt_bboxes: List[Tensor],
                      query_gt_labels: List[Tensor],
                      support_gt_labels: List[Tensor],
                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      **kwargs) -> Dict:
        """训练时前向传播函数,返回loss.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            proposals (list[Tensor]): List of region proposals with positive
                and negative pairs.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each
                query image, each item with shape (num_gts, 4)
                in [tl_x, tl_y, br_x, br_y] format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images, each item with shape (num_gts).
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images, each item with shape (1).
            query_gt_bboxes_ignore (list[Tensor] | None): Specify which
                bounding boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """

        # assign gts and sample proposals
        sampling_results = []
        num_imgs = len(query_img_metas)
        if query_gt_bboxes_ignore is None:
            query_gt_bboxes_ignore = [None for _ in range(num_imgs)]
        for i in range(num_imgs):
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

        losses = dict()
        bbox_results = self._bbox_forward_train(
            query_feats, support_feats, sampling_results, query_img_metas,
            query_gt_bboxes, query_gt_labels, support_gt_labels)
        if bbox_results is not None:
            losses.update(bbox_results)
            # if bbox_results['coarse_accuracy'] is not None:
            #     log_vars['coarse_accuracy'] = bbox_results['coarse_accuracy']

        return losses

    def _bbox_forward_train(self, query_feats: List[Tensor],
                            support_feats: List[Tensor],
                            sampling_results: object,
                            query_img_metas: List[Dict],
                            query_gt_bboxes: List[Tensor],
                            query_gt_labels: List[Tensor],
                            support_gt_labels: List[Tensor]) -> Dict:
        """Forward function and calculate loss for box head in training.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            sampling_results (obj:`SamplingResult`): Sampling results.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each query
                image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y]
                format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images.
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images.

        Returns:
            dict: Predicted results and losses.
        """
        query_rois = bbox2roi([res.bboxes for res in sampling_results
                               ])  #[batch_id, x1, y1, x2, y2]
        query_roi_feats = self.extract_query_roi_feat(query_feats, query_rois)
        # support_feats = self.extract_support_feats(support_feats)[0]
        # TODO: 检查assigner
        bbox_targets = self.get_targets(sampling_results, query_gt_bboxes,
                                        query_gt_labels, self.train_cfg)
        (query_fine_labels, query_fine_label_weights, bbox_targets,
         bbox_weights) = bbox_targets

        query_coarse_feat = query_roi_feats[0]  # query的浅层特征
        support_coarse_feat = support_feats[0]  # support的浅层特征
        query_fine_feat = query_roi_feats[1]  # query的深层特征
        support_fine_feat = support_feats[1]  # support的深层特征

        # support set浅层聚类
        cluster_num = self.num_clusters  # 粗分类类别数
        CLASS_NUM = self.num_classes  # 细分类类别数
        support_coarse_feat_np = support_coarse_feat.cpu().detach().numpy(
        )  # support的浅层特征 np.array
        support_fine_labels_tensor = torch.Tensor(
            support_gt_labels)  # support 的细粒度标签 tensor
        support_coarse_labels_np = gen_superclass(
            support_coarse_feat_np, support_fine_labels_tensor, CLASS_NUM,
            cluster_num)  # 根据support粗粒度特征进行聚类，获得support粗粒度标签 np.array
        support_coarse_data_sample_array = cal_clsvec_init(
            support_coarse_feat_np, support_fine_labels_tensor,
            CLASS_NUM)  # 获取support的细粒度特征向量
        support_coarse_clustr_array = cal_clsvec_init(
            support_coarse_data_sample_array, support_coarse_labels_np,
            cluster_num)  # 获取support的粗粒度特征向量
        support_coarse_clustr_array = torch.Tensor(
            support_coarse_clustr_array).type(
                torch.FloatTensor).cuda()  # 将X^C的特征向量转换为tensor
        support_coarse_data_sample_array = torch.Tensor(
            support_coarse_data_sample_array).type(
                torch.FloatTensor).cuda()  # 将X_c的特征向量转换为tensor

        # 粗粒度分类
        query_coarse_feat_exts = query_coarse_feat.unsqueeze(0).repeat(
            support_coarse_clustr_array.size(0), 1, 1, 1, 1).transpose(0, 1)
        support_coarse_clustr_arry_exts = support_coarse_clustr_array.unsqueeze(
            0).repeat(query_coarse_feat_exts.size(0), 1, 1, 1, 1)
        relation_pairs = torch.cat(
            (support_coarse_clustr_arry_exts, query_coarse_feat_exts),
            2).view(-1,
                    support_coarse_clustr_arry_exts.size(2) * 2, 28, 28)

        relations = self.relation_coarse_network(relation_pairs).view(
            -1, cluster_num)  # 计算query和support的粗分类关系
        query_coarse_labels = get_coare_real_labels(
            query_fine_labels, support_coarse_labels_np)  # 获取query的粗粒度标签
        loss_coarse_cls = self.loss_cls_coarse(
            relations,
            torch.tensor(query_coarse_labels).cuda(),
            query_fine_label_weights)  # 计算query的粗粒度分类损失

        #debug
        query_predict_coarse_scores, query_predict_coarse_labels = torch.max(
            relations.data, 1)  #support set预测出的粗分类标签
        query_predict_coarse_labels_debug = query_predict_coarse_labels.clone()
        query_bg_inds = query_predict_coarse_scores < self.coarse_thr  # 最大预测分数小于阈值的设置为背景
        query_predict_coarse_labels_debug[query_bg_inds] = cluster_num
        rewards = [
            1 if query_predict_coarse_labels_debug[j] == query_coarse_labels[j]
            else 0 for j in range(query_predict_coarse_labels_debug.size(0))
        ]  # 精度计算
        coarse_accuracy = np.sum(
            rewards) / 1.0 / query_predict_coarse_labels_debug.size(0)
        # print('coarse_accuracy:',coarse_accuracy)

        # 细粒度分类
        support_fine_feat_np = support_fine_feat.cpu().detach().numpy()
        support_fine_data_sample_array = cal_clsvec_init(
            support_fine_feat_np, support_fine_labels_tensor,
            CLASS_NUM)  # support的细粒度特征向量
        support_fine_data_sample_array = torch.Tensor(
            support_fine_data_sample_array).cuda()
        sample_parents = {}  # 记录每一个超类的所有细粒度样本的索引
        for i in range(support_fine_data_sample_array.size(0)):
            coarse_label = support_coarse_labels_np[i]  # 获取粗粒度类别
            if coarse_label not in sample_parents:
                sample_parents[coarse_label] = [i]
            else:
                sample_parents[coarse_label].append(i)
        predict_batch_fine_labels = []
        for i in range(query_fine_labels.size(0)):  # 样本数目
            # 只对正样本进行细分类，如果是背景类的话，在粗分类的时候就应该被筛选掉，对背景类预测出来的细分类标签没有意义，计算损失的话可能反而会降低检测结果
            batch_label = query_fine_labels[i].view(-1, 1)  # 第i个query的细粒度标签
            if batch_label == self.num_classes:
                continue  # 背景不计算细粒度类别
            else:
                batch_fine_feature = query_fine_feat[i, :]  # 第i个query的细粒度特征
                predict_batch_coarse_label = query_predict_coarse_labels[
                    i].item()  # 第i个query的预测粗分类标签
                bro_batch_labels = sample_parents[
                    predict_batch_coarse_label]  #第i个query的预测粗分类标签存在哪些support
                bro_num = len(bro_batch_labels)
                sample_fine_features_ext = support_fine_data_sample_array[
                    bro_batch_labels, :]  # 第i个query的预测粗分类标签的所有子类的细粒度特征
                batch_fine_feature_ext = batch_fine_feature.unsqueeze(
                    0).repeat(bro_num, 1, 1, 1)  # 第i个query的所有细粒度标签
                fine_pairs = torch.cat(
                    (sample_fine_features_ext, batch_fine_feature_ext),
                    1).view(-1,
                            batch_fine_feature_ext.size(1) * 2, 14, 14)
                fine_relations = self.relation_fine_network(fine_pairs).view(
                    -1, bro_num)
                new_predict_batch_fine_label = Variable(
                    torch.Tensor(
                        get_new_labels(batch_label, bro_batch_labels)).view(
                            -1, bro_num)).cuda()
                if i == 0:
                    loss_fine_cls = self.mse(fine_relations,
                                             new_predict_batch_fine_label)
                else:
                    loss_fine_cls += self.mse(fine_relations,
                                              new_predict_batch_fine_label)
                _, predict_fine_label = torch.max(fine_relations.data, 1)
                predict_batch_fine_labels.append(
                    bro_batch_labels[predict_fine_label[0]])

        # rewards = [1 if predict_batch_fine_labels[j]==batch_labels[j] else 0 for j in range(batch_labels.size(0))]

        # results = dict()
        # results['loss_bbox'] = dict()
        # results['loss_bbox']['loss_cls_coarse'] = loss_coarse_cls
        # results['loss_bbox']['loss_cls_fine'] = loss_fine_cls
        # results['coarse_accuracy'] = coarse_accuracy

        bbox_results = dict()
        bbox_results.update(
            loss_coarse_cls=loss_coarse_cls, loss_fine_cls=loss_fine_cls)
        return bbox_results
        # return results

    def extract_support_feats(self, feats: List[Tensor]) -> List[Tensor]:
        """Forward support features through shared layers.

        Args:
            feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).

        Returns:
            list[Tensor]: List of support features, each item
                with shape (N, C).
        """
        out = []
        if self.with_shared_head:
            for _, x in enumerate(feats):
                out.append(self.shared_head.forward_support(x))
        else:
            out = feats
        return out

    def extract_query_roi_feat(self, feats: List[Tensor],
                               rois: Tensor) -> Tensor:
        """Extracting query BBOX features, which is used in both training and
        testing.

        Args:
            feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            rois (Tensor): shape with (m, 5).

        Returns:
            Tensor: RoI features with shape (N, C).
        """
        roi_feats = self.bbox_roi_extractor(
            feats[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        return roi_feats

    def simple_test(self,
                    query_feats: List[Tensor],
                    support_feats_dict: Dict,
                    proposal_list: List[Tensor],
                    query_img_metas: List[Dict],
                    rescale: bool = False) -> List[List[np.ndarray]]:
        """Test without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            proposal_list (list[Tensors]): list of region proposals.
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            rescale (bool): Whether to rescale the results. Default: False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # assert self.with_bbox, 'Bbox head must be implemented.'
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
        """Test only det bboxes without augmentation.

        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)

        rois = bbox2roi(proposals)
        num_boxes = rois.shape[0]

        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)
        # prepare query feats
        query_coarse_feats = query_roi_feats[0]
        query_fine_feats = query_roi_feats[1]

        # prepare support feats
        support_gt_labels = []
        support_coarse_feats = []
        support_fine_feats = []
        for k, v in support_feats_dict['coarse_feats'].items():
            support_gt_labels.append(k)
            support_coarse_feats.append(v)
            support_fine_feats.append(support_feats_dict['fine_feats'][k])
        support_coarse_feats = torch.cat(support_coarse_feats, dim=0)
        support_fine_feats = torch.cat(support_fine_feats, dim=0)
        support_fine_labels_tensor = torch.tensor(support_gt_labels).cuda()
        # coarse classification
        cluster_num = self.num_clusters
        CLASS_NUM = self.num_classes
        support_coarse_feats_np = support_coarse_feats.cpu().detach().numpy()
        support_gt_labels_np = support_fine_labels_tensor.cpu().detach().numpy(
        )
        support_coarse_labels_np = gen_superclass(support_coarse_feats_np,
                                                  support_gt_labels_np,
                                                  CLASS_NUM, cluster_num)
        sample_coarse_data_sample_array = cal_clsvec_init(
            support_coarse_feats_np, support_gt_labels_np,
            CLASS_NUM)  # 获取每个细粒度类别的聚类中心
        sample_coarse_cluster_array = cal_clsvec_init(
            sample_coarse_data_sample_array, support_coarse_labels_np,
            cluster_num)  # 获取每个粗粒度类别的聚类中心
        sample_coarse_cluster_array = torch.Tensor(
            sample_coarse_cluster_array).type(torch.FloatTensor).cuda()
        test_features_exts = query_coarse_feats.unsqueeze(0).repeat(
            sample_coarse_cluster_array.size(0), 1, 1, 1, 1)
        test_features_exts = test_features_exts.transpose(0, 1)
        sample_coarse_cluster_array_exts = sample_coarse_cluster_array.unsqueeze(
            0).repeat(test_features_exts.size(0), 1, 1, 1, 1)
        relation_pairs = torch.cat(
            (sample_coarse_cluster_array_exts, test_features_exts),
            2).view(-1,
                    sample_coarse_cluster_array_exts.size(2) * 2, 28, 28)
        relations = self.relation_coarse_network(relation_pairs).view(
            -1, cluster_num)
        relations_sigmoid = torch.sigmoid(relations)
        predict_coarse_scores, query_predict_coarse_labels = torch.max(
            relations_sigmoid.data, 1)

        foreground_ids = predict_coarse_scores > 0.1  # 最大粗粒度类别概率大于0.1才被认为是前景类别，否则为背景类别，提取前景类别的索引进行细粒度分类

        test_fine_feats = query_fine_feats[foreground_ids]  # 前景类别的细粒度特征
        test_coarse_labels = query_predict_coarse_labels[
            foreground_ids]  # 前景类别的粗粒度标签
        support_fine_data_sample_array = cal_clsvec_init(
            support_fine_feats.cpu().detach().numpy(), support_gt_labels_np,
            CLASS_NUM)  # 细粒度类别的细粒度特征
        support_fine_data_sample_array = torch.Tensor(
            support_fine_data_sample_array).cuda()
        # 获取每一个细粒度类别的粗分类标签
        test_sample_parents = {}
        for i in range(support_fine_data_sample_array.size(0)):
            coarse_label = support_coarse_labels_np[i]
            if coarse_label not in test_sample_parents:
                test_sample_parents[coarse_label] = [i]
            else:
                test_sample_parents[coarse_label].append(i)
        predict_test_fine_labels = []
        for i in range(test_coarse_labels.size(0)):
            test_fine_images_feature = test_fine_feats[i, :]  # 测试样本的细粒度特征
            predict_test_coarse_label = test_coarse_labels[i].item(
            )  # 测试样本的粗粒度特征
            bro_test_labels = test_sample_parents[
                predict_test_coarse_label]  # 获取测试样本的粗分类类别下面对应的所有细分类类别
            bro_num = len(bro_test_labels)
            sample_fine_features_ext = support_fine_data_sample_array[
                bro_test_labels, :]  # 获取粗分类类别下面的所有细分类类别的support细分类特征
            test_fine_images_feature_ext = test_fine_images_feature.unsqueeze(
                0).repeat(bro_num, 1, 1, 1)  # 所有需要测试的样样本的粗分类特征
            fine_pairs = torch.cat(
                (sample_fine_features_ext, test_fine_images_feature_ext),
                1).view(-1,
                        test_fine_images_feature_ext.size(1) * 2, 14, 14)
            fine_relations = self.relation_fine_network(fine_pairs).view(
                -1, bro_num)  # 细分类分类分数
            _, predict_fine_label = torch.max(fine_relations.data,
                                              1)  # 预测的细分类类别(属于粗分类类别下面的细分类类别)
            predict_test_fine_labels.append(
                bro_test_labels[predict_fine_label[0]])  # 真实的细分类类别
        pred_labels = self.num_classes * torch.ones(
            num_boxes, dtype=torch.int64).cuda()  # 所有预测初始化为背景
        pred_labels[foreground_ids] = torch.tensor(
            predict_test_fine_labels, dtype=torch.int64).cuda()
        pred_boxes = proposals[0][:, 0:4]

        det_bboxes, det_labels = multiclass_nms(pred_boxes, pred_score,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        return ([
            det_bboxes,
        ], [
            det_labels,
        ])
        # return det_bboxes, det_labels
        # return ([det_bboxes,], [det_labels,])

        # return ([pred_boxes,], [pred_labels,])
