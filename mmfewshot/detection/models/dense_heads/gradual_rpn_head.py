import copy
import warnings

import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d, batched_nms, bbox_overlaps
from mmcv.runner import force_fp32
from mmcv.utils import ConfigDict
from mmdet.core import build_prior_generator, multi_apply
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import CascadeRPNHead
from mmdet.models.dense_heads.cascade_rpn_head import StageCascadeRPNHead
from torch.nn.modules.utils import _pair


def aug_proposals(proposal_list, gt_bboxes):
    aug_num = 10
    new_proposal_list = []
    for proposal, gt_bbox in zip(proposal_list, gt_bboxes):
        aug_proposals = torch.empty([0, 4], device=proposal.device)
        w = gt_bbox[:, 2] - gt_bbox[:, 0]
        h = gt_bbox[:, 3] - gt_bbox[:, 1]
        ratio = w / h

        if torch.max(ratio) > 1.5 or torch.min(
                ratio) < 0.667:  # skip too long bbox
            new_proposal_list.append(proposal)
            continue
        ctr_x = 0.5 * (gt_bbox[:, 2] + gt_bbox[:, 0])
        ctr_y = 0.5 * (gt_bbox[:, 3] + gt_bbox[:, 1])
        for i in range(aug_num):
            new_w = w * (0.707 * torch.rand(w.shape, device=w.device) + 0.707)
            new_h = h * (0.707 * torch.rand(h.shape, device=h.device) + 0.707)
            x1 = ctr_x - 0.5 * new_w
            y1 = ctr_y - 0.5 * new_h
            x2 = ctr_x + 0.5 * new_w
            y2 = ctr_y + 0.5 * new_h
            new_box = torch.stack([x1, y1, x2, y2], dim=1)
            aug_proposals = torch.cat([aug_proposals, new_box])
        scores = bbox_overlaps(aug_proposals, gt_bbox).max(
            dim=1, keepdim=True).values
        aug_proposals = torch.cat([aug_proposals, scores], dim=1)
        new_proposals = torch.cat([proposal, aug_proposals], dim=0)
        new_proposal_list.append(new_proposals)
    return new_proposal_list


@HEADS.register_module()
class StageGradualRPNHead(StageCascadeRPNHead):
    """This module replace Adaptive Conv with DCN.

    Args:
        StageCascadeRPNHead (_type_): This module replace Adaptive Conv with DCN
    """

    def __init__(self, point_generator, **kwargs):
        self.point_generator = build_prior_generator(point_generator)
        super().__init__(**kwargs)
        self.debug_dict = dict()

    def _init_layers(self):
        """Init layers of a CascadeRPN stage."""
        self.rpn_conv_offset = nn.Conv2d(
            self.in_channels,
            out_channels=18,
            kernel_size=3,
            stride=_pair(1),
            padding=_pair(1),
            dilation=_pair(1),
            bias=True)
        self.rpn_conv_offset.weight.data.zero_()
        self.rpn_conv_offset.bias.data.zero_()
        self.dcn = DeformConv2d(
            self.in_channels,
            self.feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deform_groups=1,
            bias=False)
        if self.with_cls:
            self.rpn_cls = nn.Conv2d(self.feat_channels,
                                     self.num_anchors * self.cls_out_channels,
                                     1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats):
        """Forward function."""
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        """Forward function of single scale."""
        bridged_x = x
        # dcn
        offsets = self.rpn_conv_offset(x)
        x = x.contiguous()
        offsets = offsets.contiguous()
        x = self.dcn(x, offsets)
        # relu
        x = self.relu(x)
        # features after dcn
        if self.bridged_feature:
            bridged_x = x  # update feature
        cls_score = self.rpn_cls(x) if self.with_cls else None
        bbox_pred = self.rpn_reg(x)
        return bridged_x, cls_score, bbox_pred, offsets

    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        multi_level_points = self.point_generator.grid_priors(
            featmap_sizes, device=device, with_stride=True)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.point_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'])
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             anchor_list,
             valid_flag_list,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             offset_list,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        device = cls_scores[0].device
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            featmap_sizes,
            gt_bboxes_ignore=gt_bboxes_ignore,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # debug

        if self.sampling:
            num_total_samples = num_total_pos + num_total_neg
        else:
            # 200 is hard-coded average factor,
            # which follows guided anchoring.
            num_total_samples = sum([label.numel()
                                     for label in labels_list]) / 200.0

        # change per image, per level anchor_list to per_level, per_image
        mlvl_anchor_list = list(zip(*anchor_list))
        # concat mlvl_anchor_list
        mlvl_anchor_list = [
            torch.cat(anchors, dim=0) for anchors in mlvl_anchor_list
        ]

        losses = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            mlvl_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)

        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas, device)
        mlvl_center_list = list(zip(*center_list))
        mlvl_center_list = [
            torch.cat(centers, dim=0)[:, :2] for centers in mlvl_center_list
        ]
        losses_offset = self.loss_offsets(offset_list, mlvl_center_list,
                                          mlvl_anchor_list, bbox_weights_list,
                                          img_metas)
        return dict(
            loss_rpn_cls=losses[0],
            loss_rpn_reg=losses[1],
            losses_offset=losses_offset)

    def loss_offsets(self, offset_list, center_list, anchor_list, weights_list,
                     img_metas):
        # resize weights
        mlvl_weights_list = [
            x.reshape(x.shape[0] * x.shape[1], 4) for x in weights_list
        ]
        # get positive anchors ids
        pos_ids = [x.nonzero()[:, 0].unique() for x in mlvl_weights_list]
        num_pos = sum([len(x) for x in pos_ids])
        if num_pos == 0:
            return [torch.tensor(0.0, device=pos_ids[0].device)]

        # get anchors
        pos_anchor_list = [
            anchors[pos_ids[idx]] for idx, anchors in enumerate(anchor_list)
        ]
        mlvl_anchors = torch.cat(pos_anchor_list, dim=0)

        # get strides
        strides = [
            torch.full((len(anchors), 1),
                       self.anchor_strides[idx],
                       device=mlvl_anchors.device)
            for idx, anchors in enumerate(pos_anchor_list)
        ]
        mlvl_strides = torch.cat(strides, dim=0)

        # get offsets
        num_points = offset_list[0].shape[1] // 2
        mlvl_offset_list = [
            offset.permute(0, 2, 3, 1).reshape(-1, num_points * 2)
            for offset in offset_list
        ]
        mlvl_offset_list = [
            offset[pos_ids[idx]] for idx, offset in enumerate(mlvl_offset_list)
        ]
        mlvl_offsets = torch.cat(mlvl_offset_list, dim=0)

        # get point centers
        pos_center_list = [
            centers[pos_ids[idx]] for idx, centers in enumerate(center_list)
        ]
        mlvl_centers = torch.cat(pos_center_list, dim=0)

        # calculate sampled points coordinates
        # mlvl_offsets layout [y1,x1,y2,x2,y3,x3]
        y_pts_offset = mlvl_offsets[..., 0::2]
        x_pts_offset = mlvl_offsets[..., 1::2]
        xy_pts_offsets = torch.stack([x_pts_offset, y_pts_offset], -1)
        xy_pts_offsets = xy_pts_offsets.view(*mlvl_offsets.shape[:-1], -1)
        pts = xy_pts_offsets * mlvl_strides + \
            mlvl_centers.repeat(1, num_points)

        # calculate loss
        pts_list = [pt_per_anchor.view(num_points, 2) for pt_per_anchor in pts]
        sample_anchors_list = torch.split(mlvl_anchors, 1, dim=0)
        outside_flag_list = multi_apply(self.cal_outside_flags, pts_list,
                                        sample_anchors_list)[0]
        outside_flags = torch.stack(outside_flag_list)
        loss = 1.0 / len(sample_anchors_list) * outside_flags * \
            torch.norm(xy_pts_offsets, p=2, dim=1).sum()
        return loss

    def cal_outside_flags(self, pts, box):
        """calculate if points belong to the box.

        Args:
            pts (torch.Tensor): shape (n,2)
            box (torch.Tensor): shape (1,4)
        """
        single_box = box[0]
        inside_flag = ((pts[:, 0] >= single_box[0]) &
                       (pts[:, 0] <= single_box[2]) &
                       (pts[:, 1] >= single_box[1]) &
                       (pts[:, 1] <= single_box[3]))
        outside_flag = ~inside_flag
        return (outside_flag, )

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has
                shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference from all scale
                levels of a single image, each item has shape
                (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default False.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        if isinstance(nms_pre, tuple):
            for idx in range(len(cls_scores)):
                rpn_cls_score = cls_scores[idx]
                rpn_bbox_pred = bbox_preds[idx]
                assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
                rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
                if self.use_sigmoid_cls:
                    rpn_cls_score = rpn_cls_score.reshape(-1)
                    scores = rpn_cls_score.sigmoid()
                else:
                    rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                    # We set FG labels to [0, num_class-1] and BG label to
                    # num_class in RPN head since mmdet v2.5, which is unified to
                    # be consistent with other head since mmdet v2.0. In mmdet v2.0
                    # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                    scores = rpn_cls_score.softmax(dim=1)[:, 0]
                rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                anchors = mlvl_anchors[idx]
                # first nms
                lvl_proposals = self.bbox_coder.decode(
                    anchors, rpn_bbox_pred, max_shape=img_shape)
                lvl_ids = scores.new_full((scores.size(0), ),
                                          idx,
                                          dtype=torch.long)
                lvl_dets, _ = batched_nms(lvl_proposals, scores, lvl_ids,
                                          cfg.nms)
                # sample samples
                lvl_scores = lvl_dets[:, 4][:nms_pre[idx]]
                lvl_bboxes = lvl_dets[:, :4][:nms_pre[idx]]
                level_ids.append(
                    lvl_scores.new_full((lvl_scores.size(0), ),
                                        idx,
                                        dtype=torch.long))
                mlvl_scores.append(lvl_scores)
                mlvl_bboxes.append(lvl_bboxes)

            proposals = torch.cat(mlvl_bboxes)
            scores = torch.cat(mlvl_scores)
            ids = torch.cat(level_ids)
            if cfg.min_bbox_size >= 0:
                w = proposals[:, 2] - proposals[:, 0]
                h = proposals[:, 3] - proposals[:, 1]
                valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
                if not valid_mask.all():
                    proposals = proposals[valid_mask]
                    scores = scores[valid_mask]
                    ids = ids[valid_mask]
            # deprecate arguments warning
            if 'nms' not in cfg or 'max_num' in cfg or 'nms_thr' in cfg:
                warnings.warn(
                    'In rpn_proposal or test_cfg, '
                    'nms_thr has been moved to a dict named nms as '
                    'iou_threshold, max_num has been renamed as max_per_img, '
                    'name of original arguments and the way to specify '
                    'iou_threshold of NMS will be deprecated.')
            if 'nms' not in cfg:
                cfg.nms = ConfigDict(
                    dict(type='nms', iou_threshold=cfg.nms_thr))
            if 'max_num' in cfg:
                if 'max_per_img' in cfg:
                    assert cfg.max_num == cfg.max_per_img, f'You ' \
                        f'set max_num and ' \
                        f'max_per_img at the same time, but get {cfg.max_num} ' \
                        f'and {cfg.max_per_img} respectively' \
                        'Please delete max_num which will be deprecated.'
                else:
                    cfg.max_per_img = cfg.max_num
            if 'nms_thr' in cfg:
                assert cfg.nms.iou_threshold == cfg.nms_thr, f'You set' \
                    f' iou_threshold in nms and ' \
                    f'nms_thr at the same time, but get' \
                    f' {cfg.nms.iou_threshold} and {cfg.nms_thr}' \
                    f' respectively. Please delete the nms_thr ' \
                    f'which will be deprecated.'

            if proposals.numel() > 0:
                dets, _ = batched_nms(proposals, scores, ids, cfg.nms)
            else:
                return proposals.new_zeros(0, 5)

            return dets[:cfg.max_per_img]
        else:
            for idx in range(len(cls_scores)):
                rpn_cls_score = cls_scores[idx]
                rpn_bbox_pred = bbox_preds[idx]
                assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
                rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
                if self.use_sigmoid_cls:
                    rpn_cls_score = rpn_cls_score.reshape(-1)
                    scores = rpn_cls_score.sigmoid()
                else:
                    rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                    # We set FG labels to [0, num_class-1] and BG label to
                    # num_class in RPN head since mmdet v2.5, which is unified to
                    # be consistent with other head since mmdet v2.0. In mmdet v2.0
                    # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                    scores = rpn_cls_score.softmax(dim=1)[:, 0]
                rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                anchors = mlvl_anchors[idx]

                if 0 < nms_pre < scores.shape[0]:
                    # sort is faster than topk
                    # _, topk_inds = scores.topk(cfg.nms_pre)
                    ranked_scores, rank_inds = scores.sort(descending=True)
                    topk_inds = rank_inds[:nms_pre]
                    scores = ranked_scores[:nms_pre]
                    rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                    anchors = anchors[topk_inds, :]
                mlvl_scores.append(scores)
                mlvl_bbox_preds.append(rpn_bbox_pred)
                mlvl_valid_anchors.append(anchors)
                level_ids.append(
                    scores.new_full((scores.size(0), ), idx, dtype=torch.long))

            scores = torch.cat(mlvl_scores)
            anchors = torch.cat(mlvl_valid_anchors)
            rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
            proposals = self.bbox_coder.decode(
                anchors, rpn_bbox_pred, max_shape=img_shape)
            ids = torch.cat(level_ids)

            if cfg.min_bbox_size >= 0:
                w = proposals[:, 2] - proposals[:, 0]
                h = proposals[:, 3] - proposals[:, 1]
                valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
                if not valid_mask.all():
                    proposals = proposals[valid_mask]
                    scores = scores[valid_mask]
                    ids = ids[valid_mask]

            # deprecate arguments warning
            if 'nms' not in cfg or 'max_num' in cfg or 'nms_thr' in cfg:
                warnings.warn(
                    'In rpn_proposal or test_cfg, '
                    'nms_thr has been moved to a dict named nms as '
                    'iou_threshold, max_num has been renamed as max_per_img, '
                    'name of original arguments and the way to specify '
                    'iou_threshold of NMS will be deprecated.')
            if 'nms' not in cfg:
                cfg.nms = ConfigDict(
                    dict(type='nms', iou_threshold=cfg.nms_thr))
            if 'max_num' in cfg:
                if 'max_per_img' in cfg:
                    assert cfg.max_num == cfg.max_per_img, f'You ' \
                        f'set max_num and ' \
                        f'max_per_img at the same time, but get {cfg.max_num} ' \
                        f'and {cfg.max_per_img} respectively' \
                        'Please delete max_num which will be deprecated.'
                else:
                    cfg.max_per_img = cfg.max_num
            if 'nms_thr' in cfg:
                assert cfg.nms.iou_threshold == cfg.nms_thr, f'You set' \
                    f' iou_threshold in nms and ' \
                    f'nms_thr at the same time, but get' \
                    f' {cfg.nms.iou_threshold} and {cfg.nms_thr}' \
                    f' respectively. Please delete the nms_thr ' \
                    f'which will be deprecated.'

            if proposals.numel() > 0:
                dets, _ = batched_nms(proposals, scores, ids, cfg.nms)
            else:
                return proposals.new_zeros(0, 5)

            return dets[:cfg.max_per_img]


@HEADS.register_module()
class GradualRPNHead(CascadeRPNHead):

    def __init__(self, aug_proposal=False, **kwargs):
        self.aug_proposal = aug_proposal
        super().__init__(**kwargs)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None):
        assert gt_labels is None, 'RPN does not require gt_labels'
        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        device = x[0].device
        anchor_list, valid_flag_list = self.stages[0].get_anchors(
            featmap_sizes, img_metas, device=device)

        losses = dict()
        # transform rbox to h box
        for i in range(self.num_stages):
            stage = self.stages[i]
            if stage.adapt_cfg['type'] == 'offset':
                offset_list = stage.anchor_offset(anchor_list,
                                                  stage.anchor_strides,
                                                  featmap_sizes)
                x, cls_score, bbox_pred = stage(x, offset_list)
            elif stage.adapt_cfg['type'] == 'dilation':
                offset_list = None
                x, cls_score, bbox_pred = stage(x, offset_list)
            elif stage.adapt_cfg['type'] == 'dcn':
                x, cls_score, bbox_pred, offset_list = stage(x)
            else:
                raise NotImplementedError

            rpn_loss_inputs = list((anchor_list, valid_flag_list, cls_score,
                                    bbox_pred, gt_bboxes, img_metas))
            if stage.adapt_cfg['type'] == 'dcn':
                rpn_loss_inputs.append(offset_list)
                if gt_bboxes_ignore is not None:
                    rpn_loss_inputs.append(gt_bboxes_ignore)
            stage_loss = stage.loss(*rpn_loss_inputs)
            for name, value in stage_loss.items():
                losses[f's{i}.{name}'] = value
            # refine boxes
            if i < self.num_stages - 1:
                anchor_list = stage.refine_bboxes(anchor_list, bbox_pred,
                                                  img_metas)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.stages[-1].get_bboxes(anchor_list, cls_score,
                                                       bbox_pred, img_metas,
                                                       self.test_cfg)
            if self.aug_proposal:
                proposal_list = aug_proposals(proposal_list, gt_bboxes)
            return losses, proposal_list

    def simple_test_rpn(self, x, img_metas):
        """Simple forward test function."""
        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        device = x[0].device
        anchor_list, _ = self.stages[0].get_anchors(
            featmap_sizes, img_metas, device=device)
        for i in range(self.num_stages):
            stage = self.stages[i]
            if stage.adapt_cfg['type'] == 'offset':
                offset_list = stage.anchor_offset(anchor_list,
                                                  stage.anchor_strides,
                                                  featmap_sizes)
                x, cls_score, bbox_pred = stage(x, offset_list)
            elif stage.adapt_cfg['type'] == 'dilation':
                offset_list = None
                x, cls_score, bbox_pred = stage(x, offset_list)
            elif stage.adapt_cfg['type'] == 'dcn':
                x, cls_score, bbox_pred, offset_list = stage(x)
            else:
                raise NotImplementedError

            if i < self.num_stages - 1:
                anchor_list = stage.refine_bboxes(anchor_list, bbox_pred,
                                                  img_metas)

        proposal_list = self.stages[-1].get_bboxes(anchor_list, cls_score,
                                                   bbox_pred, img_metas,
                                                   self.test_cfg)
        return proposal_list

    def forward(self, x):
        """This function is used for get params and flops.

        Args:
            x (_type_): _description_
        """
        offset_list = None
        for i in range(self.num_stages):
            stage = self.stages[i]
            if stage.adapt_cfg['type'] == 'dilation':
                offset_list = None
                x, cls_score, bbox_pred = stage(x, offset_list)
            elif stage.adapt_cfg['type'] == 'dcn':
                x, cls_score, bbox_pred, offset_list = stage(x)

        return cls_score, bbox_pred
