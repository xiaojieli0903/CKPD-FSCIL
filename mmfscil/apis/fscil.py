import copy
import os
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import build_optimizer, get_dist_info, save_checkpoint
from scipy.special import softmax
from torch.nn.utils import clip_grad

from ckpdlib.decomposition import CKPD_Adapter
from mmcls.core import CosineAnnealingCooldownLrUpdaterHook
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.utils import (get_root_logger, wrap_distributed_model,
                         wrap_non_distributed_model)
from mmfscil.datasets import MemoryDataset

CLASSES = []


class Runner:
    """"simple runner for lr scheduler"""

    def __init__(self, max_iters, optimizer):
        self.max_iters = max_iters
        self.iter = 0
        self.optimizer = optimizer

    def step(self):
        self.iter += 1

    def current_lr(self):
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr


def get_test_loader_cfg(cfg, is_distributed):
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=is_distributed,
        round_up=True,
        seed=cfg.get('seed'),
        sampler_cfg=cfg.get('sampler', None),
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    loader_cfg.update({
        'shuffle': False,
        'persistent_workers': False,
        'pin_memory': False,
        'round_up': False
    })
    # The specific dataloader settings
    test_loader_cfg = {**loader_cfg, **cfg.data.get('test_dataloader', {})}
    return test_loader_cfg


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        elif isinstance(loss_value, dict):
            for name, value in loss_value.items():
                log_vars[name] = value
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()
    return loss, log_vars


def get_training_memory(cfg,
                        model,
                        logger,
                        distributed,
                        reduce='mean',
                        replay=False):
    if model is not None:
        model.eval()

    if not replay:
        add_str = ''
    else:
        add_str = 'replay'

    if cfg.finetune_backbone:
        add_str += 'finetune'

    proto_path = os.path.join(cfg.work_dir, f'proto_{add_str}memory.npy')
    proto_label_path = os.path.join(cfg.work_dir, f'proto_{add_str}label.npy')
    if os.path.exists(proto_path):
        proto_memory = torch.tensor(np.load(proto_path))
        proto_label = torch.tensor(np.load(proto_label_path))
        logger.info(
            f'Loading proto memory from {proto_path} ({proto_memory.shape})')
        return proto_memory, proto_label

    rank, world_size = get_dist_info()
    train_dataset_cfg = copy.deepcopy(cfg.data.train)
    if train_dataset_cfg['type'] == 'RepeatDataset':
        train_dataset_cfg = train_dataset_cfg['dataset']

    train_dataset_cfg['pipeline'] = copy.deepcopy(cfg.data.test.pipeline)

    if replay:
        train_dataset_cfg['replay'] = replay
        train_dataset_cfg['num_cls'] = cfg.model.head.num_classes

    if train_dataset_cfg.augment:
        if not cfg.aug_list[cfg.cur_session]:
            train_dataset_cfg.augment = False
            logger.info(
                f'Augment flag cfg.aug_list[{cfg.cur_session}] is False,'
                f' set augment to False')

    train_ds = build_dataset(train_dataset_cfg)
    logger.info(f"Extracting proto memory to {proto_path} (reduce={reduce}).\n"
                f"The feat dataset config is: {train_dataset_cfg}.\n"
                f"The feat dataset have samples: {len(train_ds)}.\n")
    global CLASSES
    CLASSES = train_ds.CLASSES_ALL
    test_loader_cfg = get_test_loader_cfg(cfg, is_distributed=distributed)
    train_loader = build_dataloader(train_ds, **test_loader_cfg)

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(train_ds))
    else:
        prog_bar = None

    memory = OrderedDict()
    for data in train_loader:
        if model is not None:
            with torch.no_grad():
                result = model(return_loss=False, return_backbone=True, **data)
        else:
            result = data['img']

        for idx, cur in enumerate(data['img_metas'].data[0]):
            cls_id = cur['cls_id']
            img_id = cur['img_id']
            if cls_id not in memory:
                memory[cls_id] = []
            memory[cls_id].append((img_id, result[idx].to(device='cpu')))

        if rank == 0:
            prog_bar.update(len(data['img']) * world_size)

    # To circumvent MMCV bug
    if rank == 0:
        print()

    logger.info(f"Proto features init done with {len(memory)} classes.")
    dist.barrier()
    if distributed:
        dist.barrier(device_ids=[torch.cuda.current_device()])
        for cls in sorted(memory.keys()):
            memory_cls = memory[cls]
            if world_size > 1:
                recv_list = [None for _ in range(world_size)]
                # gather all result part
                dist.all_gather_object(recv_list, memory_cls)
                memory_cls = []
                for itm in recv_list:
                    memory_cls.extend(itm)
            memory_cls.sort(key=lambda x: x[0])
            if reduce == 'mean':
                memory[cls] = torch.mean(torch.stack(
                    list(map(lambda x: x[1], memory_cls))),
                                         dim=0)
            elif reduce == 'random':
                memory[cls] = memory_cls[0][1]
            elif isinstance(reduce, int):
                memory[cls] = torch.stack(
                    list(map(lambda x: x[1], memory_cls[:reduce])))
            else:
                memory[cls] = torch.stack(list(map(lambda x: x[1],
                                                   memory_cls)))
    else:
        for cls in memory:
            memory_cls = memory[cls]
            if reduce == 'mean':
                memory[cls] = torch.mean(torch.stack(
                    list(map(lambda x: x[1], memory_cls))),
                                         dim=0)
            elif reduce == 'random':
                memory_cls.sort(key=lambda x: x[0])
                memory[cls] = memory_cls[0][1]
            else:
                memory[cls] = torch.stack(list(map(lambda x: x[1],
                                                   memory_cls)))

    logger.info(f"Proto memory done with {len(memory)} classes.")

    memory_tensor = []
    memory_label_tensor = []
    if reduce in ['mean', 'random']:
        for cls in memory:
            memory_tensor.append(memory[cls])
            memory_label_tensor.append(cls)
    else:
        for cls in memory:
            memory_tensor.append(memory[cls])
            memory_label_tensor.extend([cls for _ in range(len(memory[cls]))])
    return torch.stack(memory_tensor), torch.tensor(memory_label_tensor)


def get_test_memory(cfg, model, logger, distributed, session_idx=-1):
    if model is not None:
        model.eval()
    if cfg.finetune_backbone:
        add_str = 'finetune'
    else:
        add_str = ''

    test_path = os.path.join(cfg.work_dir, f'test_{add_str}memory.npy')
    test_label_path = os.path.join(cfg.work_dir, f'test_{add_str}label.npy')
    test_filename_path = os.path.join(cfg.work_dir,
                                      f'test_{add_str}filename.npy')

    if os.path.exists(test_path):
        test_memory = np.load(test_path)
        test_label = np.load(test_label_path)
        test_filename = np.load(test_filename_path).tolist()
        logger.info(
            f'Loading test memory from {test_path} ({test_memory.shape})')
        return test_memory, test_label, test_filename

    rank, world_size = get_dist_info()
    test_dataset_cfg = copy.deepcopy(cfg.data.test)

    logger.info(f"Extracting test memory to {test_path}.\n"
                f"The test dataset config is: {test_dataset_cfg}")

    test_ds = build_dataset(test_dataset_cfg)
    global CLASSES
    CLASSES = test_ds.CLASSES_ALL
    test_loader_cfg = get_test_loader_cfg(cfg, is_distributed=distributed)
    test_loader = build_dataloader(test_ds, **test_loader_cfg)

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(test_ds))
    else:
        prog_bar = None
    test_memory = OrderedDict()
    test_set = []
    test_gt_label = []
    test_filename = []
    for data in test_loader:
        if model is not None:
            with torch.no_grad():
                result = model(return_loss=False, return_backbone=True, **data)
        else:
            result = data['img']
        for idx, cur in enumerate(data['img_metas'].data[0]):
            cls_id = cur['cls_id']
            img_id = cur['img_id']
            filename = cur['filename']
            if cls_id not in test_memory:
                test_memory[cls_id] = []
            test_memory[cls_id].append(
                (img_id, result[idx].to(device='cpu'), filename))
        if rank == 0:
            prog_bar.update(len(data['img']) * world_size)

    # To circumvent MMCV bug
    if rank == 0:
        print()

    if distributed:
        dist.barrier(device_ids=[torch.cuda.current_device()])
        for cls in sorted(test_memory.keys()):
            memory_cls = test_memory[cls]
            if world_size > 1:
                recv_list = [None for _ in range(world_size)]
                # gather all result part
                dist.all_gather_object(recv_list, memory_cls)
                memory_cls = []
                for itm in recv_list:
                    memory_cls.extend(itm)
            memory_cls.sort(key=lambda x: x[0])
            test_filename.extend(list(map(lambda x: x[2], memory_cls)))
            test_memory[cls] = torch.stack(
                list(map(lambda x: x[1], memory_cls)))
            test_set.append(test_memory[cls])
            test_gt_label.append(
                torch.ones((len(test_memory[cls]), ), dtype=torch.int) * cls)
    else:
        for cls in test_memory:
            memory_cls = test_memory[cls]
            test_filename = list(map(lambda x: x[2], memory_cls))
            test_memory[cls] = torch.stack(
                list(map(lambda x: x[1], memory_cls)))
            test_set.append(test_memory[cls])
            test_gt_label.append(
                torch.ones((len(test_memory[cls]), ), dtype=torch.int) * cls)

    test_set = np.concatenate(test_set)
    test_gt_label = np.concatenate(test_gt_label)
    logger.info("Test memory done with {} classes".format(len(test_memory)))
    return test_set, test_gt_label, test_filename


def get_inc_memory(cfg, model, logger, distributed, inc_start, inc_end):
    if model is not None:
        model.eval()
    if cfg.finetune_backbone:
        add_str = 'finetune'
    else:
        add_str = ''

    inc_path = os.path.join(cfg.work_dir, f'inc_{add_str}memory.npy')
    inc_label_path = os.path.join(cfg.work_dir, 'inc_label.npy')

    if os.path.exists(inc_path):
        inc_memory = torch.tensor(np.load(inc_path))
        inc_label = torch.tensor(np.load(inc_label_path))
        logger.info(f'Loading inc memory from {inc_path} ({inc_memory.shape})')
        return inc_memory, inc_label

    rank, world_size = get_dist_info()
    # get incremental feat memory
    inc_dataset_cfg = copy.deepcopy(cfg.data.train)
    repeat_times = inc_dataset_cfg['times'] if inc_dataset_cfg.get(
        'repeat', False) else 1
    num_shot = inc_dataset_cfg['dataset'].get('num_shot', 5)
    if inc_dataset_cfg['type'] == 'RepeatDataset':
        inc_dataset_cfg = inc_dataset_cfg['dataset']

    if repeat_times > 1:
        inc_dataset_cfg['pipeline'] = copy.deepcopy(
            cfg.data.train.dataset.pipeline)
    else:
        inc_dataset_cfg['pipeline'] = copy.deepcopy(cfg.data.test.pipeline)

    inc_dataset_cfg.update({'few_cls': tuple(range(inc_start, inc_end))})
    inc_dataset_cfg.update({'num_shot': num_shot})

    if inc_dataset_cfg.augment:
        if not cfg.aug_list[cfg.cur_session]:
            inc_dataset_cfg.augment = False
            logger.info(
                f'Augment flag cfg.aug_list[{cfg.cur_session}] is False,'
                f' set augment to False')

    logger.info(f'Extracting incremental feat memory to {inc_path}.\n'
                f'repeat times: {repeat_times}, num_shot: {num_shot}.\n'
                f'The incremental dataset config is : {inc_dataset_cfg}\n')

    inc_ds = build_dataset(inc_dataset_cfg)
    global CLASSES
    CLASSES = inc_ds.CLASSES_ALL
    inc_loader_cfg = get_test_loader_cfg(cfg, is_distributed=distributed)
    inc_loader = build_dataloader(inc_ds, **inc_loader_cfg)
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(inc_ds))
    else:
        prog_bar = None
    inc_memory = OrderedDict()
    for cls_id in range(inc_start, inc_end):
        inc_memory[cls_id] = []
    inc_set = []
    inc_gt_label = []
    for i in range(repeat_times):
        for data in inc_loader:
            if model is not None:
                with torch.no_grad():
                    result = model(return_loss=False,
                                   return_backbone=True,
                                   **data)
            else:
                result = data['img']
            for idx, cur in enumerate(data['img_metas'].data[0]):
                cls_id = cur['cls_id']
                img_id = cur['img_id']
                inc_memory[cls_id].append(
                    (img_id, result[idx].to(device='cpu')))

            if rank == 0:
                prog_bar.update(len(data['img']) * world_size)

    if rank == 0:
        print()

    if distributed:
        dist.barrier(device_ids=[torch.cuda.current_device()])
        for cls in sorted(inc_memory.keys()):
            memory_cls = inc_memory[cls]
            if world_size > 1:
                recv_list = [None for _ in range(world_size)]
                # gather all result part
                dist.all_gather_object(recv_list, memory_cls)
                memory_cls = []
                for itm in recv_list:
                    memory_cls.extend(itm)
            memory_cls.sort(key=lambda x: x[0])
            inc_memory[cls] = torch.stack(list(map(lambda x: x[1],
                                                   memory_cls)))
            inc_set.append(inc_memory[cls])
            inc_gt_label.append(
                torch.ones((len(inc_memory[cls]), ), dtype=torch.int) * cls)
    else:
        for cls in inc_memory:
            memory_cls = inc_memory[cls]
            inc_memory[cls] = torch.stack(list(map(lambda x: x[1],
                                                   memory_cls)))
            inc_set.append(inc_memory[cls])
            inc_gt_label.append(
                torch.ones((len(inc_memory[cls]), ), dtype=torch.int) * cls)
    inc_set = torch.cat(inc_set, dim=0)
    inc_gt_label = torch.cat(inc_gt_label, dim=0)
    logger.info("Incremental memory done with {} classes".format(
        len(torch.unique(inc_gt_label))))
    return inc_set, inc_gt_label


def determine_model_usage(logits, threshold=0.7):
    assert len(logits) == 2
    softmax_logits = softmax(logits, axis=-1)
    if softmax_logits[0] > threshold:
        return 'base'
    else:
        return 'novel'


def test_session(cfg,
                 model,
                 distributed,
                 test_feat: torch.Tensor,
                 test_label: torch.Tensor,
                 logger,
                 session_idx: int,
                 inc_start: int,
                 inc_end: int,
                 base_num: int,
                 mode='test',
                 model_base=None):
    model.eval()

    if model.module.neck.loss_weight_cls > 0:
        model.module.correct = 0
        model.module.total = 0

    if mode == 'train':
        return 0

    rank, world_size = get_dist_info()
    logger.info("Evaluating session {}, from {} to {}.".format(
        session_idx, inc_start, inc_end))
    mask = np.logical_and(np.greater_equal(test_label, inc_start),
                          np.less(test_label, inc_end))
    session_feats = test_feat[mask]
    session_labels = test_label[mask]
    logger.info(f'Session labels is: {session_labels}')
    test_set_memory = MemoryDataset(feats=torch.tensor(session_feats),
                                    labels=torch.tensor(session_labels))

    test_loader_memory = build_dataloader(
        test_set_memory,
        samples_per_gpu=32,
        workers_per_gpu=8,
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.get('seed'),
        shuffle=False,
        persistent_workers=False,
        pin_memory=False,
        round_up=False,
    )

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(test_set_memory))
    else:
        prog_bar = None

    result_list = []
    label_list = []
    for data in test_loader_memory:
        with torch.no_grad():
            result = model(return_loss=False,
                           return_acc=False,
                           return_pred=True,
                           img=data['feat'],
                           gt_label=data['gt_label'])
        result_list.extend(result)
        label_list.extend(data['gt_label'].tolist())
        if rank == 0:
            prog_bar.update(len(result) * world_size)

    if rank == 0:
        print()

    if distributed:
        recv_list = [None for _ in range(world_size)]
        dist.all_gather_object(recv_list, result_list)
        results = []
        for machine in recv_list:
            results.extend(machine)
        recv_list = [None for _ in range(world_size)]
        dist.all_gather_object(recv_list, label_list)
        labels = []
        for machine in recv_list:
            labels.extend(machine)
    else:
        results = result_list
        labels = label_list

    assert len(results) == len(
        test_set_memory), f'{len(results)} vs {len(test_set_memory)}'
    assert len(results) == len(labels), f'{len(results)} vs {len(labels)}'

    if model.module.neck.loss_weight_cls > 0:
        results, result_cls = [result[:-2] for result in results
                               ], [result[-2:] for result in results]
    results = torch.tensor([result.argmax(dim=-1) for result in results])
    labels = torch.tensor(labels)

    results = torch.eq(results, labels).to(dtype=torch.float32).cpu()
    acc = torch.mean(results).item() * 100.
    acc_b = torch.mean(results[labels < base_num]).item() * 100.
    acc_i = torch.mean(results[labels >= base_num]).item() * 100.
    if session_idx > 1:
        acc_i_new = torch.mean(
            results[labels >= inc_end - cfg.inc_step]).item() * 100.
    else:
        acc_i_new = torch.mean(results[labels >= 0]).item() * 100.
    acc_i_old = torch.mean(results[torch.logical_and(
        torch.less(labels, inc_end - cfg.inc_step), torch.ge(
            labels, base_num))]).item() * 100.
    logger.info(
        "[{:02d}]Evaluation results : acc : {:.2f} ; acc_base : {:.2f} ; acc_inc : {:.2f}"
        .format(session_idx, acc, acc_b, acc_i))
    logger.info(
        "[{:02d}]Evaluation results : acc_incremental_old : {:.2f} ; acc_incremental_new : {:.2f}"
        .format(session_idx, acc_i_old, acc_i_new))

    # acc for each class
    unique_labels = torch.unique(labels)
    if len(CLASSES) > inc_end:
        class_acc = {}
        for label in unique_labels:
            mask = labels == label
            class_acc[label.item()] = torch.mean(results[mask]).item() * 100.

        logger.info(f"Class-wise accuracy for session {session_idx}:")
        for label, acc_class in class_acc.items():
            logger.info(f"Class {CLASSES[label]}: {acc_class:.2f}%")
    acc_return = acc
    acc_i_new_return = acc_i_new

    return acc_return, acc_i_new_return


def build_ckpd(cfg, model, calib_datas):
    from ckpdlib.act_aware_utils import (calib_cov_distribution,
                                         calib_input_distribution)
    from ckpdlib.decomposition import build_model
    if cfg.peft_mode == 'ckpd':
        calib_cov_distribution(model, calib_datas, keys=cfg.ckpd_keys)
        cfg.act_aware = False
        cfg.cov_aware = True
        cfg.first_eigen = False
    elif cfg.peft_mode == 'asvd':
        calib_input_distribution(model, calib_datas, keys=cfg.ckpd_keys)
        cfg.act_aware = True
        cfg.cov_aware = False
        cfg.first_eigen = False
    elif cfg.peft_mode == 'svd':
        cfg.act_aware = False
        cfg.cov_aware = False
        cfg.first_eigen = False
    elif cfg.peft_mode == 'lora':
        cfg.act_aware = False
        cfg.cov_aware = False
        cfg.first_eigen = False
    build_model(model, cfg)


def fscil(model,
          cfg,
          distributed=False,
          validate=False,
          timestamp=None,
          meta=None):
    inc_start = cfg.inc_start
    inc_end = cfg.inc_end
    inc_step = cfg.inc_step
    logger = get_root_logger()
    rank, world_size = get_dist_info()

    if cfg.num_step != cfg.step_list[0]:
        cfg.step_list = [cfg.num_step for _ in range(20)]
        logger.info(f'Resetting step_list to {cfg.step_list}')
    if cfg.replay_copy != cfg.replay_copy_list[0]:
        cfg.replay_copy_list = [cfg.replay_copy for _ in range(20)]
        logger.info(f'Resetting replay_copy_list to {cfg.replay_copy_list}')
    if cfg.replay_base_copy != cfg.replay_base_copy_list[0]:
        cfg.replay_base_copy_list = [cfg.replay_base_copy for _ in range(20)]
        logger.info(
            f'Resetting replay_base_copy_list to {cfg.replay_base_copy_list}')
    if cfg.mean_copy != cfg.copy_list[0]:
        cfg.copy_list = [cfg.mean_copy for _ in range(20)]
        logger.info(f'Resetting copy_list to {cfg.copy_list}')
    if cfg.mean_base_copy != cfg.base_copy_list[0]:
        cfg.base_copy_list = [cfg.mean_base_copy for _ in range(20)]
        logger.info(f'Resetting base_copy_list to {cfg.base_copy_list}')

    model_base = None

    if cfg.use_ckpd:
        replay_memory, replay_memory_label = get_training_memory(
            cfg,
            None,
            logger,
            distributed,
            reduce=cfg.ckpd_mode,
            replay=True if not isinstance(cfg.ckpd_mode, int) else False)
        logger.info(f'replay_memory shape is {replay_memory.shape}')
        calib_loader = []
        ckpd_memory = replay_memory
        ckpd_memory_label = replay_memory_label
        for idx in range(len(ckpd_memory)):
            label = ckpd_memory_label[idx]
            if label >= inc_start:
                continue
            if isinstance(cfg.ckpd_mode, int):
                for idx_ in range(ckpd_memory[idx].shape[0]):
                    calib_loader.append({
                        'img':
                        ckpd_memory[idx][idx_].unsqueeze(0),
                        'gt_label': {label}
                    })
            else:
                calib_loader.append({
                    'img': ckpd_memory[idx].unsqueeze(0),
                    'gt_label': {label}
                })
        logger.info(f'Build model for CKPD_FSCIL '
                    f'using {len(calib_loader)} samples for calibration.')
        cfg.ckpd_keys = cfg.ckpd_keys.split('-')
        build_ckpd(cfg, model, calib_loader)

    model_finetune = copy.deepcopy(model)
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model_finetune = wrap_distributed_model(
            model_finetune,
            cfg.device,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model_finetune = wrap_non_distributed_model(model_finetune,
                                                    cfg.device,
                                                    device_ids=cfg.gpu_ids)

    if sum(cfg.base_copy_list) > 0:
        proto_memory, proto_memory_label = get_training_memory(cfg,
                                                               model_finetune,
                                                               logger,
                                                               distributed,
                                                               reduce='mean')
        logger.info(f'proto_memory feat shape is {proto_memory.shape}')
        assert proto_memory.shape[0] == len(proto_memory_label), \
            f'Mismatch!: {proto_memory.shape[0]} vs {len(proto_memory_label)}'
    if sum(cfg.replay_copy_list) > 0 or sum(cfg.replay_base_copy_list) > 0:
        replay_memory, replay_memory_label = get_training_memory(
            cfg,
            None if cfg.finetune_backbone else model_finetune,
            logger,
            distributed,
            replay=True,
            reduce='random')
        logger.info(f'replay_memory feat shape is {replay_memory.shape}')
    test_feat, test_label, test_filename = get_test_memory(
        cfg, None if cfg.finetune_backbone else model_finetune, logger,
        distributed)
    inc_feat, inc_label = get_inc_memory(
        cfg,
        None if cfg.finetune_backbone else model_finetune,
        logger,
        distributed,
        inc_start,
        inc_end,
    )

    if not cfg.finetune_backbone:
        model_finetune.module.backbone = None
        torch.cuda.empty_cache()

    acc_list = []
    acc_inc_list = []
    logger.info(f"Start to execute the incremental sessions.\n"
                f"Incremental features shape: {inc_feat.shape}.")

    save_checkpoint(model_finetune,
                    os.path.join(cfg.work_dir, 'session_{}.pth'.format(0)))
    for i in range((inc_end - inc_start) // inc_step):
        cfg.cur_session = i
        model_finetune.module.neck.cur_session = i
        label_start = inc_start + i * inc_step
        label_end = inc_start + (i + 1) * inc_step
        model_finetune.module.head.eval_classes = label_end
        model_finetune.module.head.inc_step = inc_step

        dash_line = '-' * 60
        logger.info("Starting session : {} {}".format(i + 2, dash_line))
        logger.info("Newly added classes are from {} to {}.".format(
            label_start, label_end))
        logger.info("Model now can classify {} classes".format(
            model_finetune.module.head.eval_classes))

        if cfg.finetune_backbone and i > 0:
            logger.info(
                f"Finetune_backbone={cfg.finetune_backbone}, "
                f"Re-collecting data to increase variance at session-{i}.")
            if sum(cfg.base_copy_list) > 0:
                proto_memory, proto_memory_label = get_training_memory(
                    cfg, model_finetune, logger, distributed, reduce='mean')
                logger.info(f'proto_memory feat shape is {proto_memory.shape}')
                assert proto_memory.shape[0] == len(proto_memory_label), \
                    f'Mismatch!: {proto_memory.shape[0]} vs {len(proto_memory_label)}'
            if sum(cfg.replay_copy_list) > 0 or sum(
                    cfg.replay_base_copy_list) > 0:
                replay_memory, replay_memory_label = get_training_memory(
                    cfg,
                    None,
                    logger,
                    distributed,
                    replay=True,
                    reduce='random')
                logger.info(
                    f'replay_memory feat shape is {replay_memory.shape}')
            test_feat, test_label, test_filename = get_test_memory(
                cfg, None, logger, distributed)
            inc_feat, inc_label = get_inc_memory(
                cfg,
                None,
                logger,
                distributed,
                inc_start,
                inc_end,
            )

            if cfg.use_ckpd and cfg.update_ckpd:

                def find_calib_key(name, keys):
                    matched = False
                    for key in keys:
                        if name.find(key) >= 0:
                            matched = True
                            break
                    return matched

                full_name_dict = {
                    module: name
                    for name, module in model_finetune.module.named_modules()
                }
                linear_info = {}
                modules = [model_finetune.module]
                while len(modules) > 0:
                    submodule = modules.pop()
                    for name, raw_linear in submodule.named_children():
                        full_name = full_name_dict[raw_linear]
                        if full_name.find('backbone') >= 0 and find_calib_key(
                                full_name, cfg.ckpd_keys):
                            linear_info[raw_linear] = {
                                "father": submodule,
                                "name": name,
                                "full_name": full_name,
                            }
                        else:
                            modules.append(raw_linear)
                print("\nbegin merge. \n")
                module_dict = {
                    module: name
                    for name, module in model_finetune.module.named_modules()
                }
                for module in module_dict.keys():
                    if type(module).__name__.find("CKPD_Adapter") >= 0:
                        info = linear_info[module]
                        in_features = module.BLinear.in_features
                        out_features = module.ALinear.out_features
                        new_linear = nn.Linear(in_features,
                                               out_features,
                                               bias=module.ALinear.bias
                                               is not None)
                        merged_weight = module.ALinear.weight.data @ module.BLinear.weight.data + module.weight_residual
                        new_linear.weight.data = merged_weight
                        if module.ALinear.bias is not None:
                            new_linear.bias.data = module.ALinear.bias.data
                        delattr(info["father"], info["name"])
                        setattr(info["father"], info["name"], new_linear)
                    elif type(module).__name__.find("LoRALinear") >= 0:
                        info = linear_info[module]
                        in_features = module.original_layer.in_features
                        out_features = module.original_layer.out_features

                        new_linear = nn.Linear(in_features,
                                               out_features,
                                               bias=module.original_layer.bias
                                               is not None)

                        lora_weight = (
                            module.lora_up.weight
                            @ module.lora_down.weight) * module.scaling
                        merged_weight = module.original_layer.weight.data + lora_weight

                        new_linear.weight.data = merged_weight

                        if module.original_layer.bias is not None:
                            new_linear.bias.data = module.original_layer.bias.data
                        delattr(info["father"], info["name"])
                        setattr(info["father"], info["name"], new_linear)
                calib_loader = []
                for idx in range(len(ckpd_memory)):
                    label = ckpd_memory_label[idx]
                    if label >= label_start:
                        continue

                    if isinstance(cfg.ckpd_mode, int):
                        for idx_ in range(ckpd_memory[idx].shape[0]):
                            calib_loader.append({
                                'img':
                                ckpd_memory[idx][idx_].unsqueeze(0),
                                'gt_label': {label}
                            })
                    else:
                        calib_loader.append({
                            'img':
                            ckpd_memory[idx].unsqueeze(0),
                            'gt_label': {label}
                        })

                logger.info(
                    f'Build model for CKPD_FSCIL in session-{i} '
                    f'using {len(calib_loader)} samples for calibration.')
                build_ckpd(cfg, model_finetune.module.cpu(), calib_loader)

                model_finetune = copy.deepcopy(model_finetune.module)
                if distributed:
                    find_unused_parameters = cfg.get('find_unused_parameters',
                                                     False)
                    model_finetune = wrap_distributed_model(
                        model_finetune,
                        cfg.device,
                        device_ids=[torch.cuda.current_device()],
                        broadcast_buffers=False,
                        find_unused_parameters=find_unused_parameters)
                else:
                    model_finetune = wrap_non_distributed_model(
                        model_finetune, cfg.device, device_ids=cfg.gpu_ids)
                if not cfg.finetune_backbone:
                    model_finetune.module.backbone = None
                    model_finetune.module.backbone_clip = None
                    model_finetune.module.backbone_depth = None
                    torch.cuda.empty_cache()
                if cfg.test_after_merge:
                    acc, acc_inc_new = test_session(cfg,
                                                    model_finetune,
                                                    distributed,
                                                    test_feat,
                                                    test_label,
                                                    logger,
                                                    i + 1,
                                                    0,
                                                    label_end - inc_step,
                                                    inc_start,
                                                    model_base=None)
        model_finetune.train()
        num_steps = cfg.step_list[i]
        logger.info("{} steps".format(num_steps))

        for name, param in model_finetune.named_parameters():
            if param.requires_grad:
                logger.info(
                    f'model_finetune.{name}, requires_grad={param.requires_grad},'
                    f' norm={torch.norm(param)} ({param.shape})')

        if num_steps > 0:
            cur_session_feats = inc_feat[torch.logical_and(
                torch.ge(inc_label, label_start),
                torch.less(inc_label, label_end))]
            cur_session_labels = inc_label[torch.logical_and(
                torch.ge(inc_label, label_start),
                torch.less(inc_label, label_end))]
            cur_session_feats_mean = []
            cur_session_labels_mean = []

            if sum(cfg.replay_copy_list) > 0:
                logger.info(
                    f"Extracting Replay-Inc features from {inc_start} to "
                    f"{label_start} for {cfg.replay_copy_list[i]} duplications"
                )
                inc_feat_replay = []
                inc_label_replay = []
                for idx in range(inc_start, label_start):
                    assert len(replay_memory[replay_memory_label == idx]) == 1
                    inc_feat_replay.append(
                        replay_memory[replay_memory_label == idx][0:1])
                    inc_label_replay.append(
                        replay_memory_label[replay_memory_label == idx][0:1])
                if label_start > inc_start:
                    if len(cur_session_feats.shape) == 4:
                        cur_session_feats = torch.cat([
                            cur_session_feats,
                            torch.cat(inc_feat_replay).repeat(
                                cfg.replay_copy_list[i], 1, 1, 1)
                        ])
                    else:
                        cur_session_feats = torch.cat([
                            cur_session_feats,
                            torch.cat(inc_feat_replay).repeat(
                                cfg.replay_copy_list[i], 1, 1)
                        ])
                    cur_session_labels = torch.cat([
                        cur_session_labels,
                        torch.cat(inc_label_replay).repeat(
                            cfg.replay_copy_list[i])
                    ])
            if sum(cfg.replay_base_copy_list) > 0:
                logger.info(
                    f"Extracting Replay-Base features from 0 to "
                    f"{inc_start} for {cfg.replay_base_copy_list[i]} duplications"
                )
                base_feat_replay = []
                base_label_replay = []
                for idx in range(0, inc_start):
                    assert len(replay_memory[replay_memory_label == idx]) == 1, \
                        f'{idx}--{len(replay_memory[replay_memory_label == idx])}'
                    base_feat_replay.append(
                        replay_memory[replay_memory_label == idx][0:1])
                    base_label_replay.append(
                        replay_memory_label[replay_memory_label == idx][0:1])

                if len(cur_session_feats.shape) == 4:
                    cur_session_feats = torch.cat([
                        cur_session_feats,
                        torch.cat(base_feat_replay).repeat(
                            cfg.replay_base_copy_list[i], 1, 1, 1),
                    ],
                                                  dim=0)
                else:
                    cur_session_feats = torch.cat([
                        cur_session_feats,
                        torch.cat(base_feat_replay).repeat(
                            cfg.replay_base_copy_list[i], 1, 1),
                    ],
                                                  dim=0)
                cur_session_labels = torch.cat([
                    cur_session_labels,
                    torch.cat(base_label_replay).repeat(
                        cfg.replay_base_copy_list[i]),
                ],
                                               dim=0)
            if sum(cfg.copy_list) > 0:
                logger.info(
                    f"Extracting Mean-Inc features from {inc_start} to "
                    f"{label_start} for {cfg.copy_list[i]} duplications")
                inc_feat_mean = []
                inc_label_mean = []
                if cfg.finetune_backbone:
                    inc_feat, inc_label = get_inc_memory(
                        cfg, model_finetune, logger, distributed, inc_start,
                        inc_end)
                    logger.info(f'inc_feat.shape={inc_feat.shape}')
                for idx in range(inc_start, label_start):
                    inc_feat_mean.append(inc_feat[inc_label == idx].mean(
                        dim=0, keepdim=True))
                    inc_label_mean.append(inc_label[inc_label == idx][0:1])
                if label_start > inc_start:
                    if not cfg.finetune_backbone:
                        cur_session_feats = torch.cat([
                            cur_session_feats,
                            torch.cat(inc_feat_mean).repeat(
                                cfg.copy_list[i], 1, 1, 1)
                        ])
                        cur_session_labels = torch.cat([
                            cur_session_labels,
                            torch.cat(inc_label_mean).repeat(cfg.copy_list[i])
                        ])
                    else:
                        if len(cur_session_feats_mean) == 0:
                            cur_session_feats_mean = torch.cat(
                                inc_feat_mean).repeat(cfg.copy_list[i], 1, 1,
                                                      1)
                            cur_session_labels_mean = torch.cat(
                                inc_label_mean).repeat(cfg.copy_list[i])
                        else:
                            cur_session_feats_mean = torch.cat([
                                cur_session_feats_mean,
                                torch.cat(inc_feat_mean).repeat(
                                    cfg.copy_list[i], 1, 1, 1)
                            ])
                            cur_session_labels_mean = torch.cat([
                                cur_session_labels_mean,
                                torch.cat(inc_label_mean).repeat(
                                    cfg.copy_list[i])
                            ])
            if sum(cfg.base_copy_list) > 0:
                logger.info(
                    f"Extracting Mean-Base features from 0 to "
                    f"{inc_start} for {cfg.base_copy_list[i]} duplications")
                logger.info(f'proto_memory.shape={proto_memory.shape}')
                if not cfg.finetune_backbone:
                    cur_session_feats = torch.cat([
                        cur_session_feats,
                        proto_memory.repeat(cfg.base_copy_list[i], 1, 1, 1)
                    ],
                                                  dim=0)
                    cur_session_labels = torch.cat([
                        cur_session_labels,
                        proto_memory_label.repeat(cfg.base_copy_list[i])
                    ],
                                                   dim=0)
                else:
                    if len(cur_session_feats_mean) == 0:
                        cur_session_feats_mean = proto_memory.repeat(
                            cfg.base_copy_list[i], 1, 1, 1)
                        cur_session_labels_mean = proto_memory_label.repeat(
                            cfg.base_copy_list[i])
                    else:
                        cur_session_feats_mean = torch.cat([
                            cur_session_feats_mean,
                            proto_memory.repeat(cfg.base_copy_list[i], 1, 1, 1)
                        ],
                                                           dim=0)
                        cur_session_labels_mean = torch.cat([
                            cur_session_labels_mean,
                            proto_memory_label.repeat(cfg.base_copy_list[i])
                        ],
                                                            dim=0)

            logger.info("Session-{} dataset has {} samples.".format(
                i + 2, len(cur_session_feats)))
            logger.info("Labels : {} ({})".format(cur_session_labels.tolist(),
                                                  len(cur_session_labels)))
            cur_dataset = MemoryDataset(feats=cur_session_feats,
                                        labels=cur_session_labels)
            cur_session_loader = build_dataloader(
                cur_dataset,
                samples_per_gpu=32,
                workers_per_gpu=8,
                num_gpus=len(cfg.gpu_ids),
                dist=distributed,
                seed=cfg.get('seed'),
                shuffle=True,
                persistent_workers=False,
                pin_memory=False,
                round_up=False,
                drop_last=True,
            )

            if cfg.finetune_backbone and len(cur_session_feats_mean) != 0:
                cur_dataset_mean = MemoryDataset(
                    feats=cur_session_feats_mean,
                    labels=cur_session_labels_mean)
                cur_session_loader_mean = build_dataloader(
                    cur_dataset_mean,
                    samples_per_gpu=32,
                    workers_per_gpu=8,
                    num_gpus=len(cfg.gpu_ids),
                    dist=distributed,
                    seed=cfg.get('seed'),
                    shuffle=True,
                    persistent_workers=False,
                    pin_memory=False,
                    round_up=False,
                    drop_last=True,
                )
                logger.info(
                    "Session-{} dataset has totally {} samples.".format(
                        i + 2,
                        len(cur_session_feats) + len(cur_session_feats_mean)))

            if isinstance(cfg.finetune_lr, float):
                cfg.finetune_lr = [cfg.finetune_lr] * len(cfg.copy_list)
            logger.info("Finetune lr is : {}".format(cfg.finetune_lr))
            if cfg.optimizer.type == 'SGD':
                optimizer = build_optimizer(
                    model=model_finetune,
                    cfg=dict(type='SGD',
                             lr=cfg.finetune_lr[i],
                             momentum=0.9,
                             weight_decay=0.0005,
                             paramwise_cfg=cfg.optimizer.get(
                                 'paramwise_cfg', None)))
            else:
                cfg.optimizer.lr = cfg.finetune_lr[i]
                cfg.optimizer.paramwise_cfg = cfg.optimizer.get(
                    'paramwise_cfg', None)
                logger.info(f'Optimizer is {cfg.optimizer}')
                optimizer = build_optimizer(model=model_finetune,
                                            cfg=cfg.optimizer)
            runner = Runner(num_steps, optimizer)
            lr_scheduler = CosineAnnealingCooldownLrUpdaterHook(
                min_lr=None,
                min_lr_ratio=1.e-2,
                cool_down_ratio=0.1,
                cool_down_time=5,
                by_epoch=False,
                # warmup
                warmup='linear',
                warmup_iters=5,
                warmup_ratio=0.1,
                warmup_by_epoch=False)
            lr_scheduler.before_run(runner)
            cur_session_loader_iter = iter(cur_session_loader)
            if len(cur_session_feats_mean) > 0:
                cur_session_loader_iter_mean = iter(cur_session_loader_mean)
            aug_modity_flag = False
            if not cfg.aug_list[cfg.cur_session]:
                model_finetune.module.selector.view_selector.input_view_noise = 0.0
                aug_modity_flag = True
                logger.info(f'Set input_view_noise to False')

            for idx in range(num_steps):
                runner.step()
                lr_scheduler.before_train_iter(runner)
                try:
                    data = next(cur_session_loader_iter)
                except StopIteration:
                    cur_session_loader_iter = iter(cur_session_loader)
                    if distributed:
                        cur_session_loader.sampler.set_epoch(idx + 1)
                    data = next(cur_session_loader_iter)

                if len(cur_session_feats_mean) > 0:
                    try:
                        data_mean_feats = next(cur_session_loader_iter_mean)
                    except StopIteration:
                        cur_session_loader_iter_mean = iter(
                            cur_session_loader_mean)
                        if distributed:
                            cur_session_loader_mean.sampler.set_epoch(idx + 1)
                        data_mean_feats = next(cur_session_loader_iter_mean)
                else:
                    data_mean_feats = None

                optimizer.zero_grad()

                losses = model_finetune(return_loss=True,
                                        img=data['feat'],
                                        feats=data_mean_feats,
                                        gt_label=data['gt_label'])
                loss, log_vars = parse_losses(losses)
                try:
                    loss.backward()
                except:
                    raise RuntimeError(f'Backward failed, skip!')

                if cfg.grad_clip:
                    params = model_finetune.module.parameters()
                    params = list(
                        filter(
                            lambda p: p.requires_grad and p.grad is not None,
                            params))
                    if len(params) > 0:
                        max_norm = clip_grad.clip_grad_norm_(
                            params, max_norm=cfg.grad_clip)
                        logger.info("max norm : {}".format(max_norm.item()))
                optimizer.step()
                if rank == 0:
                    logger.info(
                        "[{:03d}/{:03d}] Training session : {} ; lr : {} ; loss : {} ; acc@1 : {}"
                        .format(idx + 1, num_steps, i + 2,
                                runner.current_lr()[-1], log_vars['loss'],
                                losses['accuracy']['top-1'].item()))
                    info = ""
                    for key in log_vars:
                        info = info + f"| {key}={log_vars[key]} |"
                    if info != "":
                        logger.info(info)

            if aug_modity_flag:
                model_finetune.module.selector.view_selector.input_view_noise = cfg.model.backbone.get(
                    'input_view_noise', 0)
                logger.info(
                    f'Modify input_view_noise back to {model_finetune.module.selector.view_selector.input_view_noise}'
                )

        acc, acc_inc_new = test_session(cfg,
                                        model_finetune,
                                        distributed,
                                        test_feat,
                                        test_label,
                                        logger,
                                        i + 2,
                                        0,
                                        label_end,
                                        inc_start,
                                        model_base=None)

        acc_list.append(acc)
        acc_inc_list.append(acc_inc_new)

        save_checkpoint(
            model_finetune,
            os.path.join(cfg.work_dir, 'session_{}.pth'.format(i + 1)))
        torch.cuda.empty_cache()

    acc_str = "Overall accuracy: "
    acc_inc_str = "Inc classes accuracy: "
    str_copy = ""
    for acc in acc_list:
        acc_str += "{:.2f} ".format(acc)
        str_copy += "{:.2f} ".format(acc)
    str_copy += "{:.2f} ".format(np.mean(np.array(acc_list)))
    for acc_inc_new in acc_inc_list:
        acc_inc_str += "{:.2f} ".format(acc_inc_new)
        str_copy += "{:.2f} ".format(acc_inc_new)
    str_copy += "{:.2f} ".format(np.mean(np.array(acc_inc_list)))
    local_evaluation = (np.mean(np.array(acc_list)) +
                        np.mean(np.array(acc_inc_list))) / 2
    str_copy += "{:.2f} ".format(local_evaluation)
    logger.info(f'{acc_str}')
    logger.info(f'{acc_inc_str}')
    logger.info(f'Local accuracy is: {local_evaluation}')
    logger.info(f'Copypaste: {str_copy}')
