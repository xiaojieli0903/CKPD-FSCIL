import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from mmcls.models.builder import (CLASSIFIERS, build_head,
                                  build_neck)
from mmcls.models.classifiers.base import BaseClassifier
from mmcls.models.heads import MultiLabelClsHead
from mmcls.models.utils.augment import Augments
from mmcls.utils import get_root_logger
from mmpretrain.registry import MODELS


@CLASSIFIERS.register_module()
class ImageClassifierCILv1(BaseClassifier):
    """Classifier for Incremental Learning in Image Classification.

    This classifier integrates a configurable backbone, optional neck, and head components,
    supports feature extraction at various stages, and includes capabilities for advanced
    training schemes like Mixup.

    Args:
        backbone (dict): Configuration dict for backbone.
        neck (dict, optional): Configuration dict for neck.
        head (dict, optional): Configuration dict for head.
        pretrained (str, optional): Path to the pretrained model.
        train_cfg (dict, optional): Training configurations.
        init_cfg (dict, optional): Initialization config for the classifier.
        mixup (float, optional): Alpha value for Mixup regularization. Defaults to 0.
        mixup_prob (float, optional): Probability of applying Mixup per batch. Defaults to 0.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None,
                 mixup: float = 0.,
                 mixup_prob: float = 0.,
                 use_kd=False):
        super().__init__(init_cfg)
        self.logger = get_root_logger()
        if pretrained:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        # self.backbone = build_backbone(backbone)
        self.backbone = MODELS.build(backbone)
        self.mamba_neck = False if neck.type.find('Mamba') < 0 else True
        self.use_kd = use_kd

        if neck:
            self.neck = build_neck(neck)

        if head:
            self.head = build_head(head)

        self.augments = None
        if train_cfg:
            augments_cfg = train_cfg.get('augments')
            if augments_cfg:
                self.augments = Augments(augments_cfg)

        if self.neck.loss_weight_cls > 0:
            self.correct = 0
            self.total = 0

        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.logger.info(self.backbone)
        self.logger.info(self.neck)
        self.logger.info(self.head)

    def extract_feat(self, img, stage='neck', skip_backbone=False):
        """Directly extract features from the specified stage.

        Args:
            img (Tensor): The input images. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from
                "backbone", "neck" and "pre_logits". Defaults to "neck".

        Returns:
            tuple | Tensor: The output of specified stage.
                The output depends on detailed implementation. In general, the
                output of backbone and neck is a tuple and the output of
                pre_logits is a tensor.
        """
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        if self.backbone and not skip_backbone:
            x = self.backbone(img)
        else:
            x = img

        if stage == 'backbone':
            if isinstance(x, tuple):
                x = x[-1]
            return x

        if self.neck:
            x = self.neck(x)

        if stage == 'neck':
            return x

        if self.head and hasattr(self.head, 'pre_logits'):
            x = self.head.pre_logits(x)
        return x

    def forward_train(self, img, gt_label, feats=None, **kwargs):
        """Forward computation during training.

            This method handles data augmentation, feature extraction, and loss computation for training.

            Args:
                img (Tensor): Input images of shape (N, C, H, W).
                gt_label (Tensor): Ground truth labels. Shape (N, 1) for single label tasks,
                                   or (N, C) for multi-label tasks.

            Returns:
                dict[str, Tensor]: Dictionary of all computed loss components.
            """
        # Apply data augmentation if configured
        if self.augments:
            img, lam, gt_label, gt_label_aux = self.augments(img, gt_label)
        else:
            gt_label_aux = None
            lam = None
        if feats is None:
            x = self.extract_feat(img)
        else:
            x_backbone = self.extract_feat(img, stage='backbone')
            if isinstance(x_backbone, tuple):
                x_backbone = x_backbone[-1]
            x_feats = torch.cat([x_backbone, feats['feat']], dim=0)
            x = self.extract_feat(x_feats, skip_backbone=True)
            gt_label = torch.cat([gt_label, feats['gt_label']], dim=0)

        if isinstance(x, dict):
            x_main = x.get('main')
            x_residual = x.get('residual')
            dts = x.get('dts')
            Bs = x.get('Bs')
            Cs = x.get('Cs')
            dts_new = x.get('dts_new')
            Bs_new = x.get('Bs_new')
            Cs_new = x.get('Cs_new')
            x = x['out']

        losses = dict()
        if self.mixup == 0.:
            loss = self.head.forward_train(x, gt_label)
            losses.update(loss)

        # make sure not mixup feat
        if gt_label_aux is not None:
            assert self.mixup == 0.
            loss = self.head.forward_train(x, gt_label_aux)
            losses['loss_main'] = losses['loss'] * lam
            losses['loss_aux'] = loss['loss'] * (1 - lam)
            del losses['loss']

        # Calculate norms for different feature sets
        indices_base = gt_label < self.head.base_classes
        indices_novel = gt_label >= self.head.base_classes
        self.calculate_norms(losses, indices_base, indices_novel, x, x_main,
                             x_residual, dts, Bs, Cs, dts_new, Bs_new, Cs_new)

        # Losses for feature separation and suppression
        if self.mamba_neck:
            self.calculate_class_sensitive_losses(losses, indices_base,
                                                  indices_novel, x_main, dts,
                                                  Bs, Cs, dts_new, Bs_new,
                                                  Cs_new)
            if self.neck.loss_weight_cls > 0:
                if self.neck.input_cls == 'main':
                    input_cls = x_main.detach()
                elif self.neck.input_cls == 'residual':
                    input_cls = x_residual.detach()
                elif self.neck.input_cls == 'sum':
                    input_cls = x.detach()
                elif self.neck.input_cls == 'dts_base':
                    input_cls = dts.detach()
                elif self.neck.input_cls == 'Bs_base':
                    input_cls = Bs.detach()
                elif self.neck.input_cls == 'Cs_base':
                    input_cls = Cs.detach()
                elif self.neck.input_cls == 'dts_inc':
                    input_cls = dts_new.detach()
                elif self.neck.input_cls == 'Bs_inc':
                    input_cls = Bs_new.detach()
                elif self.neck.input_cls == 'Cs_inc':
                    input_cls = Cs_new.detach()
                elif self.neck.input_cls == 'dts_cat':
                    input_cls = torch.cat((dts, dts_new), dim=2).detach()
                elif self.neck.input_cls == 'Bs_cat':
                    input_cls = torch.cat((Bs, Bs_new), dim=2).detach()
                elif self.neck.input_cls == 'Cs_cat':
                    input_cls = torch.cat((Cs, Cs_new), dim=2).detach()
                elif self.neck.input_cls == 'dts_sum':
                    input_cls = (dts + dts_new).detach()
                elif self.neck.input_cls == 'Bs_sum':
                    input_cls = (Bs + Bs_new).detach()
                elif self.neck.input_cls == 'Cs_sum':
                    input_cls = (Cs + Cs_new).detach()
                elif self.neck.input_cls == 'g_base_cat':
                    input_cls = torch.cat((dts, Bs, Cs), dim=2).detach()
                elif self.neck.input_cls == 'g_inc_cat':
                    input_cls = torch.cat((dts_new, Bs_new, Cs_new),
                                          dim=2).detach()
                elif self.neck.input_cls == 'g_base_sum':
                    input_cls = (dts + Bs + Cs).detach()
                elif self.neck.input_cls == 'g_inc_sum':
                    input_cls = (dts_new + Bs_new + Cs_new).detach()

                if self.neck.version == 'ss2d':
                    # ss3d: x_main [32, 14, 14, 1024], dts [32, 4, 256, 196]
                    if self.neck.input_cls in ['main', 'residual', 'sum']:
                        input_cls = input_cls.mean((1, 2))
                    elif self.neck.input_cls.split('_')[0] in [
                            'dts', 'Bs', 'Cs'
                    ]:
                        input_cls = input_cls.mean((1, 3))
                    elif self.neck.input_cls in [
                            'g_base_sum', 'g_inc_sum', 'g_base_cat',
                            'g_inc_cat'
                    ]:
                        input_cls = input_cls.mean((1, 3))
                elif self.neck.version == 'ssm':
                    # ssm: x_main [64, 768, 196], dts [64, 1, 256, 196]
                    if self.neck.input_cls in ['main', 'residual', 'sum']:
                        input_cls = input_cls.mean(-1)
                    elif self.neck.input_cls.split('_')[0] in [
                            'dts', 'Bs', 'Cs'
                    ]:
                        input_cls = input_cls.mean((1, 3))
                    elif self.neck.input_cls in [
                            'g_base_sum', 'g_inc_sum', 'g_base_cat',
                            'g_inc_cat'
                    ]:
                        input_cls = input_cls.mean((1, 3))

                cls_pred = self.neck.mlp_cls(input_cls)
                if self.neck.cur_session < self.neck.num_cls_session:
                    losses['loss_cls'] = self.neck.loss_cls(
                        cls_pred, indices_novel.long())
                with torch.no_grad():
                    predictions = torch.argmax(cls_pred, dim=1)
                    correct = (predictions == indices_novel).sum().item()
                    total = indices_novel.size(0)
                    accuracy = correct / total
                self.logger.info(f"Cls Accuracy: {accuracy * 100:.2f}%")

        # mixup feat when mixup > 0, this cannot be with augment mixup
        if self.mixup > 0. and self.mixup_prob > 0. and np.random.random() > (
                1 - self.mixup_prob):
            x, gt_a, gt_b, lam = self.mixup_feat(x, gt_label, alpha=self.mixup)
            loss1 = self.head.forward_train(x, gt_a)
            loss2 = self.head.forward_train(x, gt_b)
            losses['loss'] = lam * loss1['loss'] + (1 - lam) * loss2['loss']
            losses['ls_mixup_1'] = loss1['loss']
            losses['ls_mixup_2'] = loss2['loss']
            losses['accuracy'] = loss1['accuracy']
        else:
            loss = self.head.forward_train(x, gt_label)
            losses.update(loss)

        return losses

    def calculate_sep_loss(self, params, indices_base, indices_novel):
        """
        Calculates a separation loss by comparing the average features of base and novel classes.

        Args:
            params (Tensor): Features from which to calculate separation.
            indices_base (Tensor): Boolean tensor indicating base class indices.
            indices_novel (Tensor): Boolean tensor indicating novel class indices.

        Returns:
            Tensor: The mean separation loss.
        """
        base_params = params[indices_base]
        novel_params = params[indices_novel]
        if self.neck.loss_sep_version == 'v1':
            avg_features = []
            avg_features.append(
                base_params.mean(self.neck.param_avg_dim).reshape((1, -1)))
            avg_features.append(
                novel_params.mean(self.neck.param_avg_dim).reshape((1, -1)))

            avg_features = torch.cat(avg_features, dim=0)
            normalized_input = F.normalize(avg_features, dim=-1)
            similarity_matrix = torch.matmul(normalized_input,
                                             normalized_input.transpose(0, 1))
            confusion_matrix = torch.abs(
                torch.eye(similarity_matrix.shape[0], device=params.device) -
                similarity_matrix)
            return torch.mean(confusion_matrix)
        else:
            # Compute average features
            avg_base = base_params.mean(self.neck.param_avg_dim).reshape(
                (1, -1))
            avg_novel = novel_params.mean(self.neck.param_avg_dim).reshape(
                (1, -1))

            # Normalize average features
            avg_base_normalized = F.normalize(avg_base, dim=-1)
            avg_novel_normalized = F.normalize(avg_novel, dim=-1)

            # Calculate cosine similarity between base and novel
            cosine_similarity = torch.matmul(
                avg_base_normalized, avg_novel_normalized.transpose(0, 1))

            # Extract the single similarity value
            return cosine_similarity.squeeze()

    def calculate_sep_losses(self, losses, indices_base, indices_novel, dts,
                             Bs, Cs, dts_new, Bs_new, Cs_new):
        """
        Calculates separation losses for ssm branches.

        Args:
            losses (dict): Losses dictionary to update.
            indices_base (Tensor): Indices for base class examples.
            indices_novel (Tensor): Indices for novel class examples.
            dts, Bs, Cs, dts_new, Bs_new, Cs_new (Tensor): Feature tensors for calculating separation.
        """
        if self.neck.loss_weight_sep > 0:
            for key, value in [('dts', dts), ('Bs', Bs), ('Cs', Cs)]:
                if value is not None:
                    losses[
                        f'loss_sep_{key}_base'] = self.neck.loss_weight_sep * self.calculate_sep_loss(
                            value, indices_base, indices_novel)
        if self.neck.loss_weight_sep_new > 0:
            for key, value in [('dts_new', dts_new), ('Bs_new', Bs_new),
                               ('Cs_new', Cs_new)]:
                if value is not None:
                    losses[
                        f'loss_sep_{key}'] = self.neck.loss_weight_sep_new * self.calculate_sep_loss(
                            value, indices_base, indices_novel)

    def calculate_class_sensitive_losses(self, losses, indices_base,
                                         indices_novel, x_main, dts, Bs, Cs,
                                         dts_new, Bs_new, Cs_new):
        """
        Computes class-sensitive losses for feature suppression and separation.

        Args:
            losses (dict): Dictionary to store computed losses.
            indices_base (Tensor): Indices for base class examples.
            indices_novel (Tensor): Indices for novel class examples.
            x_main, dts, Bs, Cs, dts_new, Bs_new, Cs_new (Tensor): Feature tensors for calculating losses.
        """
        num_base = indices_base.sum()
        num_novel = indices_novel.sum()
        dist.all_reduce(num_base, op=dist.ReduceOp.MIN)
        dist.all_reduce(num_novel, op=dist.ReduceOp.MIN)
        # Feature suppression and separation losses
        if num_base > 0 and self.neck.loss_weight_supp > 0:
            losses['loss_supp_base'] = self.neck.loss_weight_supp * torch.norm(
                x_main[indices_base]) / torch.numel(x_main[indices_base])
        if num_novel > 0 and self.neck.loss_weight_supp_novel > 0:
            losses[
                'loss_supp_novel'] = -self.neck.loss_weight_supp_novel * torch.norm(
                    x_main[indices_novel]) / torch.numel(x_main[indices_novel])
        if num_base > 0 and num_novel > 0:
            self.calculate_sep_losses(losses, indices_base, indices_novel, dts,
                                      Bs, Cs, dts_new, Bs_new, Cs_new)

    def calculate_norms(self, losses, indices_base, indices_novel, x, x_main,
                        x_residual, dts, Bs, Cs, dts_new, Bs_new, Cs_new):
        """
        Calculates the norms of feature tensors.

        Args:
            losses (dict): Dictionary to store calculated norms.
            indices_base (Tensor): Indices for base class examples.
            indices_novel (Tensor): Indices for novel class examples.
            feature_tensors (dict): Feature tensors keyed by their type.
        """
        # Base and novel feature norms
        for key, value in [('input', x), ('main', x_main),
                           ('residual', x_residual)]:
            losses[f'norm_{key}_base'] = torch.norm(
                value[indices_base]) / len(indices_base) + 1e-6
            losses[f'norm_{key}_novel'] = torch.norm(
                value[indices_novel]) / len(indices_novel) + 1e-6

        # Input-dependent parameters norms
        for key, value in [('dts', dts), ('Bs', Bs), ('Cs', Cs),
                           ('dts_new', dts_new), ('Bs_new', Bs_new),
                           ('Cs_new', Cs_new)]:
            if value is not None:
                losses[f'norm_{key}_base'] = torch.norm(
                    value[indices_base]) / len(indices_base) + 1e-6
                losses[f'norm_{key}_novel'] = torch.norm(
                    value[indices_novel]) / len(indices_novel) + 1e-6

    def simple_test(self,
                    img,
                    gt_label=None,
                    return_backbone=False,
                    return_feat=False,
                    return_acc=False,
                    return_pred=False,
                    img_metas=None,
                    **kwargs):
        """Test without augmentation."""
        if return_backbone:
            x = self.extract_feat(img, stage='backbone')
            return x
        x = self.extract_feat(img)
        if self.neck.loss_weight_cls > 0:
            x_main = x.get('main')
            x_residual = x.get('residual')
            dts = x.get('dts')
            Bs = x.get('Bs')
            Cs = x.get('Cs')
            dts_new = x.get('dts_new')
            Bs_new = x.get('Bs_new')
            Cs_new = x.get('Cs_new')
            x_sum = x['out']

            if self.neck.input_cls == 'main':
                input_cls = x_main.detach()
            elif self.neck.input_cls == 'residual':
                input_cls = x_residual.detach()
            elif self.neck.input_cls == 'sum':
                input_cls = x_sum.detach()
            elif self.neck.input_cls == 'dts_base':
                input_cls = dts.detach()
            elif self.neck.input_cls == 'Bs_base':
                input_cls = Bs.detach()
            elif self.neck.input_cls == 'Cs_base':
                input_cls = Cs.detach()
            elif self.neck.input_cls == 'dts_inc':
                input_cls = dts_new.detach()
            elif self.neck.input_cls == 'Bs_inc':
                input_cls = Bs_new.detach()
            elif self.neck.input_cls == 'Cs_inc':
                input_cls = Cs_new.detach()
            elif self.neck.input_cls == 'dts_cat':
                input_cls = torch.cat((dts, dts_new), dim=2).detach()
            elif self.neck.input_cls == 'Bs_cat':
                input_cls = torch.cat((Bs, Bs_new), dim=2).detach()
            elif self.neck.input_cls == 'Cs_cat':
                input_cls = torch.cat((Cs, Cs_new), dim=2).detach()
            elif self.neck.input_cls == 'dts_sum':
                input_cls = (dts + dts_new).detach()
            elif self.neck.input_cls == 'Bs_sum':
                input_cls = (Bs + Bs_new).detach()
            elif self.neck.input_cls == 'Cs_sum':
                input_cls = (Cs + Cs_new).detach()
            elif self.neck.input_cls == 'g_base_cat':
                input_cls = torch.cat((dts, Bs, Cs), dim=2).detach()
            elif self.neck.input_cls == 'g_inc_cat':
                input_cls = torch.cat((dts_new, Bs_new, Cs_new),
                                      dim=2).detach()
            elif self.neck.input_cls == 'g_base_sum':
                input_cls = (dts + Bs + Cs).detach()
            elif self.neck.input_cls == 'g_inc_sum':
                input_cls = (dts_new + Bs_new + Cs_new).detach()

            if self.neck.version == 'ss2d':
                # ss3d: x_main [32, 14, 14, 1024], dts [32, 4, 256, 196]
                if self.neck.input_cls in ['main', 'residual', 'sum']:
                    input_cls = input_cls.mean((1, 2))
                elif self.neck.input_cls.split('_')[0] in ['dts', 'Bs', 'Cs']:
                    input_cls = input_cls.mean((1, 3))
                elif self.neck.input_cls in [
                        'g_base_sum', 'g_inc_sum', 'g_base_cat', 'g_inc_cat'
                ]:
                    input_cls = input_cls.mean((1, 3))
            elif self.neck.version == 'ssm':
                # ssm: x_main [64, 768, 196], dts [64, 1, 256, 196]
                if self.neck.input_cls in ['main', 'residual', 'sum']:
                    input_cls = input_cls.mean(-1)
                elif self.neck.input_cls.split('_')[0] in ['dts', 'Bs', 'Cs']:
                    input_cls = input_cls.mean((1, 3))
                elif self.neck.input_cls in [
                        'g_base_sum', 'g_inc_sum', 'g_base_cat', 'g_inc_cat'
                ]:
                    input_cls = input_cls.mean((1, 3))

            indices_novel = gt_label >= self.head.base_classes
            with torch.no_grad():
                cls_pred = self.neck.mlp_cls(input_cls)
                predictions = torch.argmax(cls_pred, dim=1)
                correct = (predictions == indices_novel).sum().item()
                total = indices_novel.size(0)
                self.correct += correct
                self.total += total
                accuracy = self.correct / self.total
                cur_acc = correct / total
                cur_novel_per = (torch.sum(indices_novel) /
                                 predictions.shape[0]) * 100
            self.logger.info(
                f"All acc: {accuracy * 100:.2f}%, Current acc: {cur_acc}, novel_classes percent: {cur_novel_per}"
            )

        if return_feat:
            assert not return_acc
            return x

        if isinstance(self.head, MultiLabelClsHead):
            assert 'softmax' not in kwargs, (
                'Please use `sigmoid` instead of `softmax` '
                'in multi-label tasks.')
        if 'post_process' in kwargs:
            post_process = kwargs.pop('post_process')
        else:
            post_process = None
        res = self.head.simple_test(x,
                                    post_process=not return_acc
                                    if post_process is None else post_process,
                                    **kwargs)
        if return_acc:
            res = res.argmax(dim=-1)
            return torch.eq(
                res, gt_label).to(dtype=torch.float32).cpu().numpy().tolist()
        if return_pred:
            # res = torch.tensor(res).argmax(dim=-1)
            res = torch.tensor(res)
            if self.neck.loss_weight_cls > 0:
                res = torch.cat((res, cls_pred.cpu()), dim=-1)
            return res
        return res

    @staticmethod
    def mixup_feat(feat, gt_labels, alpha=1.0):
        """
        Applies mixup augmentation directly to features.

        Args:
            feat (Tensor): Features to which mixup will be applied.
            gt_labels (Tensor): Ground truth labels corresponding to the features.
            alpha (float): Alpha parameter for the beta distribution.

        Returns:
            Tuple of mixed features and corresponding labels.
        """
        if alpha > 0:
            lam = alpha
        else:
            lam = 0.5
        if isinstance(feat, dict):
            feat = feat['out']
        batch_size = feat.size()[0]
        index = torch.randperm(batch_size).to(device=feat.device)

        mixed_feat = lam * feat + (1 - lam) * feat[index, :]
        gt_a, gt_b = gt_labels, gt_labels[index]

        return mixed_feat, gt_a, gt_b, lam
