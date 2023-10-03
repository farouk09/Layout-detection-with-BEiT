auto_scale_lr = dict(base_batch_size=8, enable=False)
backend_args = None
data_root = '/home/farouk/BEiT_CV_parsing/object_detection/dataset/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True,
        test_out_dir=
        '/home/farouk/BEiT_CV_parsing/object_detection/work_dirs/faster-rcnn_beit_base_12_512_cvparsing/results',
        type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/home/farouk/BEiT_CV_parsing/object_detection/work_dirs/faster-rcnn_beit_base_12_512_cvparsing/epoch_11.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=100)
metainfo = dict(
    classes='section', palette=[
        (
            220,
            20,
            60,
        ),
    ])
model = dict(
    backbone=dict(
        depth=6,
        drop_path_rate=0.15,
        embed_dim=768,
        img_size=256,
        init_values=0.1,
        mlp_ratio=4,
        num_heads=6,
        out_indices=[
            1,
            3,
            5,
        ],
        patch_size=16,
        qkv_bias=True,
        type='BEiT',
        use_abs_pos_emb=False,
        use_rel_pos_bias=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            768,
            768,
            768,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='IoULoss'),
        loss_centerness=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=1,
        stacked_convs=4,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type='FCOSHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.1, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.1, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='FasterRCNN')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(decay_mult=1.0, lr_mult=0.1)),
        norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=100,
        gamma=0.1,
        milestones=[
            16,
            22,
            36,
            45,
            66,
            77,
            88,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        '/home/farouk/BEiT_CV_parsing/object_detection/dataset/annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='images/test/'),
        data_root='/home/farouk/BEiT_CV_parsing/object_detection/dataset/',
        metainfo=dict(classes='section', palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                256,
                256,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/home/farouk/BEiT_CV_parsing/object_detection/dataset/annotations/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        256,
        256,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=10)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file=
        '/home/farouk/BEiT_CV_parsing/object_detection/dataset/annotations/train.json',
        backend_args=None,
        data_prefix=dict(img='images/train/'),
        data_root='/home/farouk/BEiT_CV_parsing/object_detection/dataset/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes='section', palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_mask=True),
            dict(scale=(
                256,
                256,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_mask=True),
    dict(scale=(
        256,
        256,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        '/home/farouk/BEiT_CV_parsing/object_detection/dataset/annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='images/test/'),
        data_root='/home/farouk/BEiT_CV_parsing/object_detection/dataset/',
        metainfo=dict(classes='section', palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                256,
                256,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/home/farouk/BEiT_CV_parsing/object_detection/dataset/annotations/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/faster-rcnn_beit_base_12_512_cvparsing'
