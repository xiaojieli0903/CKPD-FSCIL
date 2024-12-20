_base_ = [
    '../_base_/models/resnet_etf.py',
    '../_base_/datasets/mini_imagenet_fscil.py',
    '../_base_/schedules/mini_imagenet_500e.py', '../_base_/default_runtime.py'
]
img_size = 224
_img_resize_size = 256
# model settings
model = dict(
    type='ImageClassifierCILv1',
    backbone=dict(
        _delete_=True,
        type='VisionTransformer',
        arch='b',
        patch_size=16,
        img_size=img_size,
        drop_path_rate=0.1,
        frozen_stages=12,
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
              use_residual_proj=True,
              use_new_branch=True,
              detach_residual=False,
              num_layers_new=3,
              loss_weight_supp=0,
              loss_weight_supp_novel=0,
              loss_weight_sep_new=0,
              loss_weight_sep=0,
              param_avg_dim='0-1-3'),
    head=dict(type='ETFHead',
              in_channels=1024,
              with_len=True,
              num_classes=100,
              eval_classes=60),
    mixup=0.5,
    mixup_prob=0.3)

mean_copy = 5
mean_base_copy = 2
copy_list = [mean_copy for _ in range(10)]
base_copy_list = [mean_base_copy for _ in range(10)]

replay_copy = 5
replay_base_copy = 2
replay_copy_list = [replay_copy for _ in range(10)]
replay_base_copy_list = [replay_base_copy for _ in range(10)]
finetune_backbone = True

num_step = 1000
step_list = [num_step for _ in range(10)]

finetune_lr = 0.01
mult = 0.1
# optimizer
optimizer = dict(type='SGD',
                 lr=finetune_lr,
                 momentum=0.9,
                 weight_decay=0.0005,
                 paramwise_cfg=dict(
                     custom_keys={
                         'backbone.': dict(lr_mult=1e-1),
                         'neck.block.': dict(lr_mult=mult),
                         'neck.residual_proj': dict(lr_mult=mult),
                         'neck.pos_embed': dict(lr_mult=mult),
                         'neck.mlp_proj.': dict(lr_mult=mult),
                         'neck.pos_embed_new': dict(lr_mult=1)
                     }))
find_unused_parameters = True

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

data = dict(samples_per_gpu=128,
            workers_per_gpu=8,
            train_dataloader=dict(persistent_workers=True, ),
            val_dataloader=dict(persistent_workers=True, ),
            test_dataloader=dict(persistent_workers=True, ),
            train=dict(type='RepeatDataset',
                       times=10,
                       dataset=dict(type='MiniImageNetFSCILDataset',
                                    data_prefix='./data/miniimagenet',
                                    pipeline=train_pipeline,
                                    num_cls=60,
                                    subset='train',
                                    submit_mode=False,
                                    augment=True)),
            val=dict(type='MiniImageNetFSCILDataset',
                     data_prefix='./data/miniimagenet',
                     pipeline=test_pipeline,
                     num_cls=60,
                     subset='test',
                     submit_mode=False),
            test=dict(type='MiniImageNetFSCILDataset',
                      data_prefix='./data/miniimagenet',
                      pipeline=test_pipeline,
                      num_cls=100,
                      subset='test',
                      submit_mode=False))

use_ckpd = True
update_ckpd = True
