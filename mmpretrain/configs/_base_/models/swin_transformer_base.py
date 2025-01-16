# Copyright (c) OpenMMLab. All rights reserved.
# This is a BETA new format config file, and the usage may change recently.
from mmpretrain.models import (CrossEntropyLoss, GlobalAveragePooling,
                               ImageClassifier, LinearClsHead, SwinTransformer)

# model settings
model = dict(type=ImageClassifier,
             backbone=dict(type=SwinTransformer,
                           arch='base',
                           img_size=384,
                           stage_cfgs=dict(block_cfgs=dict(window_size=12))),
             neck=dict(type=GlobalAveragePooling),
             head=dict(type=LinearClsHead,
                       num_classes=1000,
                       in_channels=1024,
                       loss=dict(type=CrossEntropyLoss, loss_weight=1.0),
                       topk=(1, 5)))
