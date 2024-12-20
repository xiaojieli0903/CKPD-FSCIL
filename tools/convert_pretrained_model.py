# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import clip
import mmengine
import timm
import torch


def convert_timm(ckpt):
    """Convert TIMM model checkpoint keys to the desired format."""
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('head'):
            new_k = k.replace('head.', 'head.layers.head.')
        elif k.startswith('patch_embed'):
            new_k = k.replace('proj.', 'projection.') if 'proj.' in k else k
        elif k.startswith('norm_pre'):
            new_k = k.replace('norm_pre', 'pre_norm')
        elif k.startswith('blocks'):
            new_k = k.replace('blocks.', 'layers.')
            new_k = new_k.replace('norm1', 'ln1').replace('norm2', 'ln2')
            new_k = new_k.replace('mlp.fc1', 'ffn.layers.0.0').replace(
                'mlp.fc2', 'ffn.layers.1')
        elif k.startswith('norm'):
            new_k = k.replace('norm', 'ln1')
        else:
            new_k = k

        if not new_k.startswith('head'):
            new_k = 'backbone.' + new_k

        new_ckpt[new_k] = v

    return new_ckpt


def convert_clip_openai(ckpt):
    """Convert OpenAI CLIP checkpoint keys to the desired format."""
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        if 'visual' not in k:
            continue
        new_k = k.replace('visual.', '')
        new_v = v

        if new_k.startswith('conv1'):
            new_k = new_k.replace('conv1', 'patch_embed.projection')
        elif new_k.startswith('positional_embedding'):
            new_k = new_k.replace('positional_embedding', 'pos_embed')
            new_v = new_v.reshape(1, new_v.shape[0], new_v.shape[1])
        elif new_k.startswith('class_embedding'):
            new_k = new_k.replace('class_embedding', 'cls_token')
            new_v = new_v.reshape(1, 1, new_v.shape[0])
        elif new_k.startswith('ln_pre'):
            new_k = new_k.replace('ln_pre', 'pre_norm')
        elif new_k.startswith('ln_post'):
            new_k = new_k.replace('ln_post', 'ln1')
        elif 'resblocks' in new_k:
            new_k = new_k.replace('transformer.resblocks', 'layers')
            if 'attn.in_proj_weight' in new_k:
                new_k = new_k.replace('in_proj_weight', 'qkv.weight')
            elif 'attn.in_proj_bias' in new_k:
                new_k = new_k.replace('in_proj_bias', 'qkv.bias')
            elif 'attn.out_proj.' in new_k:
                new_k = new_k.replace('attn.out_proj.', 'attn.proj.')
            elif 'mlp.c_fc' in new_k:
                new_k = new_k.replace('mlp.c_fc', 'ffn.layers.0.0')
            elif 'mlp.c_proj' in new_k:
                new_k = new_k.replace('mlp.c_proj', 'ffn.layers.1')
            elif 'ln_1' in new_k:
                new_k = new_k.replace('ln_1', 'ln1')
            elif 'ln_2' in new_k:
                new_k = new_k.replace('ln_2', 'ln2')
        else:
            print(f'No matched key for {new_k}')

        new_state_dict[new_k] = new_v
    return new_state_dict


def main():
    parser = argparse.ArgumentParser(
        description='Convert model keys to a different format.')
    parser.add_argument(
        'src',
        help=
        'Source model name or path (e.g., ViT-B/32 for CLIP or vit_base_patch16_224 for TIMM model name).'
    )
    parser.add_argument('dst',
                        help='Destination path to save the converted model.')
    parser.add_argument('--model-type',
                        choices=['clip', 'timm'],
                        required=True,
                        help='Type of the model to convert.')
    args = parser.parse_args()

    if args.model_type == 'clip':
        checkpoint, _ = clip.load(args.src, device='cpu')
        state_dict = checkpoint.state_dict()
        converted_weights = convert_clip_openai(state_dict)
    elif args.model_type == 'timm':
        model = timm.create_model(args.src, pretrained=True)
        state_dict = model.state_dict()
        converted_weights = convert_timm(state_dict)

    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(converted_weights, args.dst)

    print('Conversion complete!')


if __name__ == '__main__':
    main()
