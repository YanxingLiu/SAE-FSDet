_base_ = [
    '../../../_base_/datasets/fine_tune_based/base_vhr10.py',
    '../../../_base_/schedules/schedule.py',
    '../../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../../_base_/default_runtime.py'
]

# region experiments setting
exp_doc = 'vhr10 base training'
exp_version = 'v1'
# hyper parameters
split_id = 1
val_interval = 300
train_lr = 0.01
train_warmup_iters = 100
train_steps = [800, 1350]
train_max_iters = 1500  # around 18 epochs
model_num_classes = 7
rpn_weight = 0.7
# endregion

checkpoint_config = dict(interval=val_interval)
evaluation = dict(interval=val_interval, metric='mAP')
lr_config = dict(warmup_iters=train_warmup_iters, step=train_steps)
runner = dict(max_iters=train_max_iters)
# using regular sampler can get a better base model
use_infinite_sampler = False
optimizer = dict(type='SGD', lr=train_lr, momentum=0.9, weight_decay=0.0001)

# data setting
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(classes=f'BASE_CLASSES_SPLIT{split_id}'),
    val=dict(classes=f'BASE_CLASSES_SPLIT{split_id}'),
    test=dict(classes=f'BASE_CLASSES_SPLIT{split_id}'))

# model settings
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    rpn_head=dict(
        _delete_=True,
        type='GradualRPNHead',
        num_stages=2,
        stages=[
            dict(
                type='StageCascadeRPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[1.0],
                    strides=[4, 8, 16, 32, 64]),
                adapt_cfg=dict(type='dilation', dilation=3),
                bridged_feature=True,
                with_cls=False,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.5, 0.5)),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight)),
            dict(
                type='StageGradualRPNHead',
                in_channels=256,
                feat_channels=256,
                adapt_cfg=dict(type='dcn'),
                bridged_feature=False,
                with_cls=True,
                reg_decoded_bbox=True,
                point_generator=dict(
                    type='MlvlPointGenerator', strides=[4, 8, 16, 32, 64]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.05, 0.05, 0.1, 0.1)),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0 * rpn_weight),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight))
        ]),
    roi_head=dict(
        bbox_head=dict(reg_class_agnostic=True,
                       num_classes=model_num_classes)),
    train_cfg=dict(
        rpn=[
            dict(
                assigner=dict(
                    type='RegionAssigner', center_ratio=0.2, ignore_ratio=0.5),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.3,
                    ignore_iof_thr=0.3,
                    iou_calculator=dict(type='BboxOverlaps2D')),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=0.3),
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0)))
