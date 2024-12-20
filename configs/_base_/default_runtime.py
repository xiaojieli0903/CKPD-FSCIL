# checkpoint saving
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
evaluation = dict(interval=1, save_best='auto')
log_config = dict(interval=10, hooks=[
    dict(type='TextLoggerHook'),
])

dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]

load_from = None
resume_from = None

# Test configs
mean_neck_feat = True
mean_cur_feat = False
feat_test = False
grad_clip = None
finetune_lr = 0.1
inc_start = 60
inc_end = 100
inc_step = 5

copy_list = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
step_list = (50, 50, 50, 50, 50, 50, 50, 50, 50, 50)
base_copy_list = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
vis_tsne = False
vis_tsne_norm = False
vis_tsne_type = 'all'
vis_structure = False
finetune_backbone = False
aug_list = (True, True, True, True, True, True, True, True, True, True)
cur_session = 0
use_ckpd = False
ckpd_rank = 128
ckpd_version = 'origin'
update_ckpd = False
test_after_merge = False
ckpd_mode = 'random'
peft_mode = 'ckpd'
adaptive_chosen = False
top_n = 6
ckpd_keys = 'layers.11.attn.proj-layers.11.attn.qkv-layers.11.ffn.layers.1-layers.10.attn.proj-layers.10.attn.qkv-layers.10.ffn.layers.1'
num_calib = None
