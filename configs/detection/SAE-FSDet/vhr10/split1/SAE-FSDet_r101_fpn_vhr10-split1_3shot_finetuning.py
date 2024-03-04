_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_vhr10.py',
    '../../../_base_/schedules/schedule.py',
    '../../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../../_base_/default_runtime.py'
]
# region experiments setting
exp_doc = '完整版实验'
exp_version = 'v3'
# hyper parameters
split_id = 1
val_interval = 800
train_lr = 0.001
train_warmup_iters = 100
train_steps = [
    2100,
]
train_max_iters = 2400
model_num_classes = 10
evaluation_class_splits = ['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1']
rpn_weight = 0.7
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='MMfewshotWandbHook',
            init_kwargs={
                'project': 'SAE-FSDet',
                'name': '{{ fileBasenameNoExtension }}',
                'notes': exp_doc,
            },
            interval=50,
            log_checkpoint=False,
            log_checkpoint_metadata=False,
            num_eval_images=0,
            bbox_score_thr=0.3)
    ])

load_from = 'work_dirs/SAE-FSDet_r101_fpn_vhr10-split1_base-training/latest.pth'
# endregion

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        ann_cfg=[
            dict(type='ann_file', split=f'SPLIT{split_id}', setting='3SHOT')
        ],
        num_base_shots=3,
        num_novel_shots=3,
        classes=f'ALL_CLASSES_SPLIT{split_id}'),
    val=dict(classes=f'ALL_CLASSES_SPLIT{split_id}'),
    test=dict(classes=f'ALL_CLASSES_SPLIT{split_id}'))

checkpoint_config = dict(interval=val_interval)
evaluation = dict(
    interval=val_interval, metric='mAP', class_splits=evaluation_class_splits)
lr_config = dict(warmup_iters=train_warmup_iters, step=train_steps)
runner = dict(max_iters=train_max_iters)
# using regular sampler can get a better base model
use_infinite_sampler = False
optimizer = dict(type='SGD', lr=train_lr, momentum=0.9, weight_decay=0.0001)

model = dict(
    frozen_parameters=[
        'backbone',
    ],
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    rpn_head=dict(
        _delete_=True,
        type='GradualRPNHead',
        aug_proposal=True,
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
        type='LCCRoIHead',
        bbox_head=dict(
            _delete_=True,
            type='LCCBoxHead',
            use_dropout=False,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_shared_fcs=2,
            num_classes=7,
            num_novel_classes=3,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='LCCFocalLoss',
                num_base_classes=7,
                num_novel_classes=3,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            init_cfg=[
                dict(
                    type='Caffe2Xavier',
                    override=dict(type='Caffe2Xavier', name='shared_fcs')),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_cls', std=0.01)),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_reg', std=0.001))
            ],
        )),
    train_cfg=dict(rpn=[
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
    ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0)))
