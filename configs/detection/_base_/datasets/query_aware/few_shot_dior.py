# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='Resize',
            img_scale=[(800, 800)],
            keep_ratio=True,
            multiscale_mode='value'),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    support=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='CropResizeInstance',
            num_context_pixels=16,
            target_size=(320, 320)),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
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
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# classes splits are predefined in FewShotVOCDataset
data_root = 'data/DIOR/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='QueryAwareDataset',
        num_support_ways=None,
        num_support_shots=None,
        save_dataset=True,
        dataset=dict(
            type='FewShotDIORDefaultDataset',
            img_prefix=data_root,
            ann_cfg=None,
            multi_pipelines=train_multi_pipelines,
            classes=None,
            use_difficult=False,
            instance_wise=False,
            min_bbox_area=32 * 32,
            dataset_name='query_support_dataset')),
    val=dict(
        type='FewShotDIORDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'ImageSets/Main/test.txt')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=None),
    test=dict(
        type='FewShotDIORDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'ImageSets/Main/test.txt')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        classes=None),
    model_init=dict(
        copy_from_train_dataset=True,
        samples_per_gpu=16,
        workers_per_gpu=1,
        type='FewShotDIORDefaultDataset',
        ann_cfg=None,
        img_prefix=data_root,
        pipeline=train_multi_pipelines['support'],
        use_difficult=False,
        instance_wise=True,
        num_novel_shots=None,
        classes=None,
        min_bbox_area=32 * 32,
        dataset_name='model_init_dataset'))
evaluation = dict(interval=3000, metric='mAP', class_splits=None)
