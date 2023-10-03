# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/CV_dataset.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='FasterRCNN',

    backbone=dict(
        type='BEiT',
        img_size=256,
        patch_size=16,
        embed_dim=768,
        depth=6,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=0.1,
        drop_path_rate=0.15,
        out_indices=[1, 3, 5]
    ),

    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768],
        out_channels=256,
        num_outs=5),

    rpn_head=dict(
        _delete_=True,  # ignore the unused old settings
        type='FCOSHead',
        # num_classes = 1 for rpn,
        # if num_classes > 1, it will be set to 1 in
        # TwoStageDetector automatically
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    roi_head=dict(  # update featmap_strides
        bbox_roi_extractor=dict(featmap_strides=[8, 16, 32, 64, 128])),


)








# Modify dataset related settings
"""data_root = '/home/farouk/BEiT_CV_parsing/object_detection/publaynet/'
metainfo = {
    'classes': ("text","title","list","table","figure"),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228)]  
}"""
