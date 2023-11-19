checkpoint_file = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth'  # noqa
backbone_norm_cfg = dict(type='LN', requires_grad=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

num_classes = 3

model = dict(
    type='DeformableDETR_CellDet_Prompt_GP',
    backbone=dict(
        type='PromptedSwinTransformer',
        prompt_config=dict(
            DEEP=False,
            LOCATION="prepend",
            NUM_TOKENS=64,
            INITIATION="random",
            DROPOUT=0.0,
            PROJECT=-1,
            GPForPrompt=True,
        ),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(1, 2, 3),
        window_size=12,
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        use_abs_pos_embed=False,
        convert_weights=True,
        with_cp=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DeformableDETRHead_CellDet_GP_Modify',
        num_query=300,
        num_classes=num_classes,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        gp_classifier=dict(
            embed_dim=256,
            num_heads=[8, 8],
            num_group_tokens=[64, num_classes],
            num_output_groups=[64, num_classes],
            hard_assignment=True,
            prompt=True,
        ),
        transformer=dict(
            type='DeformableDetrTransformer_CellDet',
            two_stage_num_proposals=300,
            encoder=dict(
                type='DetrTransformerEncoder_CellDet',
                num_layers=3,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder_CellDet',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer_CellDet',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_point=dict(type='L1Loss', loss_weight=5.0), ),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner_CellDet',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='PointL1Cost', weight=5.0))
    ),
    test_cfg=dict(max_per_img=300)
)

albu_train_transforms = [
    # dict(
    #     type='ShiftScaleRotate',
    #     shift_limit=0.0625,
    #     scale_limit=0.0,
    #     rotate_limit=0,
    #     interpolation=1,
    #     p=0.5),
    # dict(
    #     type='RandomBrightnessContrast',
    #     brightness_limit=[0.1, 0.3],
    #     contrast_limit=[0.1, 0.3],
    #     p=0.2),
    # dict(
    #     type='HueSaturationValue',
    #     hue_shift_limit=20,
    #     sat_shift_limit=30,
    #     val_shift_limit=20,
    #     p=0.2),
    # dict(type="CLAHE", clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
    # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    # dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=0.5),
            dict(type='MedianBlur', blur_limit=3, p=0.5),
            dict(type="MotionBlur", p=0.5)
        ],
        p=0.5),
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    # img_scale=[(600, 1333), (700, 1333),
                    #            (800, 1333), (900, 1333), (1000, 1333),
                    #            (1100, 1333), (1200, 1333)],
                    img_scale=[(640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(800, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(800, 800),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    # img_scale=[(600, 1333), (700, 1333),
                    #            (800, 1333), (900, 1333), (1000, 1333),
                    #            (1100, 1333), (1200, 1333)],
                    img_scale=[(640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(800, 1333),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=1),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

dataset_type = 'CellDetDataset_BRCA'
data_root = '/Path to Dataset/SAHI/'
ann_root = '/Path to Dataset/SAHI/annotations/'
ann_root_hovernet = '/Path to Dataset/Val/Labels/'


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=ann_root + 'BRCA_train.json',
        img_prefix=data_root + 'BRCA_Train/',
        pipeline=train_pipeline,
        data_root=ann_root_hovernet),
    val=dict(
        type=dataset_type,
        ann_file=ann_root + 'BRCA_val.json',
        img_prefix=data_root + 'BRCA_Val/',
        pipeline=test_pipeline,
        data_root=ann_root_hovernet),
    test=dict(
        type=dataset_type,
        ann_file=ann_root + 'BRCA_test.json',
        img_prefix=data_root + 'BRCA_Test/',
        pipeline=test_pipeline,
        data_root=ann_root_hovernet))

evaluation = dict(interval=800, metric='points', save_best='F1d', rule="greater")

optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=320000)
checkpoint_config = dict(by_epoch=False, interval=32000)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=2)

load_from = 'Path to pretrained model'
