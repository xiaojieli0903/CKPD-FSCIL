import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def find_calib_key(name, keys):
    matched = False
    for key in keys:
        if name.find(key) >= 0:
            matched = True
            break
    return matched


class CKPD_Adapter(nn.Module):

    def __init__(self,
                 adapter_U,
                 adapter_S,
                 adapter_V,
                 weight_residual,
                 bias=None,
                 sigma_fuse='UV') -> None:
        super().__init__()
        U, S, V = adapter_U, adapter_S, adapter_V
        rank = V.size(1)
        self.weight_residual = nn.Parameter(
            torch.zeros(U.size(0), V.size(0)).to(adapter_U.device))  # (m , n)
        self.weight_residual.data = weight_residual
        self.weight_residual.requires_grad = False
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias
                                 is not None)  ## r -> m

        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(V.size(0), V.size(1), bias=False)  ## n -> r

        if sigma_fuse == 'UV':
            self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
            self.BLinear.weight.data = V.t().mul(S.sqrt().view(
                -1, 1)).contiguous()
        elif sigma_fuse == 'U':
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.t().contiguous()
        elif sigma_fuse == 'V':
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()

    def set_session(self, session):
        self.session = session
        print(f'Setting session to {session}')

    def forward(self, inp):
        y = self.BLinear(inp)
        y = self.ALinear(y) + F.linear(inp, self.weight_residual)
        return y


class LoRALinear(nn.Module):
    r"""Implements LoRA in a linear layer.

    Args:
        original_layer (nn.Linear): The linear layer to be finetuned.
        alpha (int): The scale factor of LoRA. Defaults to 1.
        rank (int): The rank of LoRA. Defaults to 0.
        drop_rate (float): The drop out rate for LoRA. Defaults to 0.

    Note:
        The forward process of LoRA linear layer is:

        .. math::
            `y = W_0 x + BAx * (\alpha / r)`

        Where :math:`x` is the input, :math:`y` is the output,
        :math:`W_0` is the parameter of the original layer,
        :math:`A` and :math:`B` are the low-rank decomposition matrixs,
        :math: `\alpha` is the scale factor and :math: `r` is the rank.
    """

    def __init__(self,
                 original_layer: nn.Linear,
                 alpha: int = 1,
                 rank: int = 0,
                 drop_rate: float = 0.):
        super(LoRALinear, self).__init__()
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_dropout = nn.Dropout(drop_rate)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.original_layer = original_layer

    def forward(self, x: torch.Tensor):
        out = self.original_layer(x)
        lora_x = self.lora_dropout(x)
        lora_out = self.lora_up(self.lora_down(lora_x)) * self.scaling
        return out + lora_out


def decompose_to_lora(linear: nn.Linear,
                      alpha: int = 1,
                      rank: int = 0,
                      drop_rate: float = 0.0):
    rank = min(rank, min(linear.in_features, linear.out_features))

    lora_layer = LoRALinear(
        original_layer=linear,
        alpha=alpha,
        rank=rank,
        drop_rate=drop_rate,
    )

    return lora_layer, None


def decompose_to_adapter(linear: nn.Linear,
                         act_aware=False,
                         cov_aware=False,
                         alpha=1,
                         sigma_fuse="UV",
                         r=16,
                         first_eigen=False,
                         version='origin'):
    rank = min(linear.in_features, linear.out_features)
    pretrained_w = linear.weight.data.float()
    if act_aware:
        scaling_diag_matrix = 1  # avoid zero division
        if hasattr(linear, "scaling_diag_matrix"):
            scaling_diag_matrix *= linear.scaling_diag_matrix**alpha
        if hasattr(linear, "fisher_info"):
            scaling_diag_matrix *= linear.fisher_info**alpha
        scaling_diag_matrix += 1e-6  # avoid zero division
        w = pretrained_w * scaling_diag_matrix.view(1, -1)
    elif cov_aware:
        assert hasattr(linear, "covariance_matrix")
        covariance_matrix = linear.covariance_matrix.float()
        damp = 0.01
        while True:
            compensate = torch.diag(
                torch.ones(covariance_matrix.size(0)).to(
                    covariance_matrix.device) *
                torch.mean(torch.diag(covariance_matrix)) * damp)
            fix_covariance_matrix = covariance_matrix + compensate
            cov_inv = torch.linalg.inv(fix_covariance_matrix)
            inv_error = torch.dist(
                fix_covariance_matrix @ cov_inv,
                torch.eye(covariance_matrix.size(0)).to(
                    covariance_matrix.device))
            if inv_error.data < 0.05:
                break
            else:
                damp = damp * 2
        w = pretrained_w @ fix_covariance_matrix  ## w: out_dim, in_dim; covariance_matrix: in_dim, in_dim

    try:
        if act_aware or cov_aware:
            U, S, V = torch.linalg.svd(w, full_matrices=False)
            V = V.transpose(0, 1)
        else:
            U, S, V = torch.linalg.svd(pretrained_w, full_matrices=False)
            V = V.transpose(0, 1)
    except:
        raise Exception(f"svd failed for {linear}")

    if act_aware:
        V = V / scaling_diag_matrix.view(-1, 1)
    elif cov_aware:
        if version == 'origin_normed':
            V_cov_inv = V.t() @ cov_inv
            row_norms = torch.norm(V_cov_inv, dim=1, keepdim=True)
            V_cov_inv_normalized = V_cov_inv / row_norms
            V = V_cov_inv_normalized.transpose(0, 1)
            S = S * row_norms.view(-1)
        else:
            V = (V.t() @ cov_inv).transpose(0, 1)

    if linear.bias is not None:
        bias = linear.bias.data
    else:
        bias = None

    # nan or inf check
    if torch.isnan(S).any() or torch.isinf(S).any():
        raise Exception("nan or inf in S")
    if torch.isnan(U).any() or torch.isinf(U).any():
        raise Exception("nan or inf in U")
    if torch.isnan(V).any() or torch.isinf(V).any():
        raise Exception("nan or inf in V")
    
    S_origin = S
    ## Use the last r principal components
    if not first_eigen:
        U = U[:, -r:]  ## m, r
        S = S[-r:]  ## r
        V = V[:, -r:]  ## n, r
    ## Use the first r principal components following PiSSA !!!
    elif first_eigen:
        U = U[:, :r]  ## m, r
        S = S[:r]  ## r
        V = V[:, :r]  ## n, r

    weight_residual = pretrained_w - U @ torch.diag(S) @ V.transpose(0,
                                                                     1)  ## m,n
    # Calculate the ASR value if adaptive_chosen
    metric =  sum(S_origin[-r:]) / sum(S_origin)

    if torch.isnan(weight_residual).any() or torch.isinf(
            weight_residual).any():
        raise Exception("nan or inf in weight_residual")

    linear_with_adapter = CKPD_Adapter(U, S, V, weight_residual, bias,
                                       sigma_fuse)

    linear_with_adapter.to(linear.weight.dtype)
    linear_with_adapter.to(linear.weight.device)
    linear_with_adapter.weight_residual = linear_with_adapter.weight_residual.to(
        linear.weight.dtype)
    assert not torch.isnan(linear_with_adapter.weight_residual).any()
    assert not torch.isinf(linear_with_adapter.weight_residual).any()

    del pretrained_w, U, S, V, weight_residual, linear
    torch.cuda.empty_cache()

    return linear_with_adapter, metric


def build_model(model, args):
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    ckpd_keys = args.ckpd_keys
    linear_info = {}
    metrics = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)
    my_layers_keys = []
    for name, module in model.named_modules():
        if isinstance(
                module,
                nn.Linear) and name.find('backbone') >= 0 and find_calib_key(
                    name, ckpd_keys):
            my_layers_keys.append(name)
    print(my_layers_keys)
    print('--- model before svd ----')
    for layername in tqdm(my_layers_keys):
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        with torch.no_grad():
            if args.peft_mode in ['ckpd', 'svd', 'asvd']:
                linear_with_adapter, metric = decompose_to_adapter(
                    raw_linear,
                    act_aware=args.act_aware,
                    cov_aware=args.cov_aware,
                    r=args.ckpd_rank,
                    first_eigen=args.first_eigen,
                    version=args.ckpd_version,
                )
                metrics[layername] = metric
                delattr(info["father"], info["name"])
                if args.cov_aware: delattr(raw_linear, "covariance_matrix")
                if args.act_aware: delattr(raw_linear, "scaling_diag_matrix")
                setattr(info["father"], info["name"], linear_with_adapter)
            elif args.peft_mode in ['lora']:
                linear_with_adapter, metric = decompose_to_lora(
                    raw_linear,
                    rank=args.ckpd_rank,
                )
                metrics[layername] = metric
                delattr(info["father"], info["name"])
                if args.cov_aware: delattr(raw_linear, "covariance_matrix")
                if args.act_aware: delattr(raw_linear, "scaling_diag_matrix")
                setattr(info["father"], info["name"], linear_with_adapter)
            del module_dict[layername], linear_info[raw_linear]
            del raw_linear, info,
            torch.cuda.empty_cache()

    # Select the 10 layers with the smallest ASR values
    if args.adaptive_chosen:
        sorted_layers = sorted(metrics.items(), key=lambda x: x[1])
        chosen_layers = {name for name, _ in sorted_layers[:args.top_n]}
        print(f"Chosen {args.top_n} layers for training: {chosen_layers}")

        # Freeze all adapters except chosen ones
        for name, module in model.named_modules():
            if isinstance(module, CKPD_Adapter) and name not in chosen_layers:
                for param in module.parameters():
                    param.requires_grad = False
                print(f'Freezing adapter: {name}')
    print('--- model after svd ----')
    print(model)
