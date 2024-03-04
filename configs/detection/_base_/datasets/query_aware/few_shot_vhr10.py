# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='Resize',
            img_scale=[(1024, 1024)],
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
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = 'data/NWPU VHR-10 dataset/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='QueryAwareDataset',
        num_support_ways=None,
        num_support_shots=None,
        save_dataset=True,
        dataset=dict(
            type='FewShotVHR10DefaultDataset',
            img_prefix=data_root,
            ann_cfg=None,
            multi_pipelines=train_multi_pipelines,
            classes=None,
            use_difficult=False,
            instance_wise=False,
            min_bbox_area=2 * 2,
            dataset_name='query_support_dataset')),
    val=dict(
        type='FewShotVHR10Dataset',
        ann_cfg=[dict(type='ann_file', ann_file=data_root + 'Main/test.txt')],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=None),
    test=dict(
        type='FewShotVHR10Dataset',
        ann_cfg=[dict(type='ann_file', ann_file=data_root + 'Main/test.txt')],
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        classes=None),
    model_init=dict(
        copy_from_train_dataset=False,
        samples_per_gpu=16,
        workers_per_gpu=1,
        type='FewShotVHR10DefaultDataset',
        ann_cfg=None,
        img_prefix=data_root,
        pipeline=train_multi_pipelines['support'],
        use_difficult=False,
        instance_wise=True,
        num_novel_shots=None,
        classes=None,
        min_bbox_area=2 * 2,
        dataset_name='model_init_dataset'))
