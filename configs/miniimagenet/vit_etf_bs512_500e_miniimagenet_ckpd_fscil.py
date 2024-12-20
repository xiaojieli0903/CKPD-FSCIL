_base_ = [
    '../_base_/models/resnet_etf.py',
    '../_base_/datasets/mini_imagenet_fscil.py',
    # '../_base_/schedules/mini_imagenet_500e.py',
    '../_base_/default_runtime.py'
]

# dataset
img_size = 224
_img_resize_size = 256
model = dict(
    type='ImageClassifierCILv1',
    backbone=dict(
        _delete_=True,
        type='VisionTransformer',
        arch='b',
        patch_size=16,
        img_size=img_size,
        drop_path_rate=0.1,
        frozen_stages=11,
        pre_norm=True,
        out_type='featmap',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./pretrained_models/clip-vit-base-p16_openai.pth'),
    ),
    neck=dict(type='MambaNeck',
              version='ss2d',
              in_channels=768,
              out_channels=1024,
              feat_size=img_size // 16,
              num_layers=3,
              use_residual_proj=True),
    head=dict(type='ETFHead',
              in_channels=1024,
              num_classes=100,
              eval_classes=60,
              with_len=True),
    mixup=0,
    mixup_prob=0)

img_size = 224
_img_resize_size = 256
img_norm_cfg = dict(mean=[255 * 0.48145466, 255 * 0.4578275, 255 * 0.40821073],
                    std=[255 * 0.26862954, 255 * 0.26130258, 255 * 0.27577711],
                    to_rgb=True)
meta_keys = ('filename', 'ori_filename', 'ori_shape', 'img_shape', 'flip',
             'flip_direction', 'img_norm_cfg', 'cls_id', 'img_id')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=img_size),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(int(img_size * 1.15), -1)),
    dict(type='CenterCrop', crop_size=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

data = dict(samples_per_gpu=64,
            workers_per_gpu=8,
            train_dataloader=dict(persistent_workers=True, ),
            val_dataloader=dict(persistent_workers=True, ),
            test_dataloader=dict(persistent_workers=True, ),
            train=dict(type='RepeatDataset',
                       times=10,
                       dataset=dict(
                           type='MiniImageNetFSCILDataset',
                           data_prefix='./data/miniimagenet',
                           pipeline=train_pipeline,
                           num_cls=60,
                           subset='train',
                       )),
            val=dict(
                type='MiniImageNetFSCILDataset',
                data_prefix='./data/miniimagenet',
                pipeline=test_pipeline,
                num_cls=60,
                subset='test',
            ),
            test=dict(
                type='MiniImageNetFSCILDataset',
                data_prefix='./data/miniimagenet',
                pipeline=test_pipeline,
                num_cls=100,
                subset='test',
            ))

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealingCooldown',
    min_lr=None,
    min_lr_ratio=0.1,
    cool_down_ratio=0.1,
    cool_down_time=10,
    by_epoch=False,
    # warmup
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    warmup_by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=20)
