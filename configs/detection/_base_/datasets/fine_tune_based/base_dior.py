# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(800, 800)],
        keep_ratio=True,
        multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = 'data/DIOR/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='FewShotDIORDatasetV2',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'ImageSets/Main/trainval.txt')
        ],
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes=None,
        use_difficult=True,
        min_bbox_area=2,  # some error data that bbox area is 0
        instance_wise=False),
    val=dict(
        type='FewShotDIORDatasetV2',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'ImageSets/Main/test.txt')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=None,
    ),
    test=dict(
        type='FewShotDIORDatasetV2',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'ImageSets/Main/test.txt')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        classes=None,
    ))
evaluation = dict(interval=3000, metric='mAP')
