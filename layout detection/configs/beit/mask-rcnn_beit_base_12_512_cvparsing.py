_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/Publaynet.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

"""vis_backends = [dict(type='LocalVisBackend'), dict(type='WandBVisBackend')]
visualizer = dict(vis_backends=vis_backends)

# MMEngine support the following two ways, users can choose
# according to convenience
# default_hooks = dict(checkpoint=dict(interval=4))
_base_.default_hooks.checkpoint.interval = 4

# train_cfg = dict(val_interval=2)
_base_.train_cfg.val_interval = 2"""


model = dict(
    type='MaskRCNN',

    backbone=dict(
        type='BEiT',
        img_size=512,
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
        num_outs=5))
