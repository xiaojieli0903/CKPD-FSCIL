import torch
import torch.nn as nn
from tqdm import tqdm


def find_calib_key(name, keys):
    matched = False
    for key in keys:
        if name.find(key) >= 0:
            matched = True
            break
    return matched


def calib_input_distribution(model,
                             calib_loader,
                             method='abs_mean',
                             keys=['ffn.layers.1']):
    print(f"building input distribution for fscil")
    model.eval()

    def hook(module, input, output):
        if "abs_mean" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
            module.scaling_diag_matrix += abs_mean
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            )

    for name, module in model.named_modules():
        if isinstance(
                module,
                nn.Linear) and name.find('backbone') >= 0 and find_calib_key(
                    name, keys):
            module.scaling_diag_matrix = 0
            module.register_forward_hook(hook)

    for data in tqdm(calib_loader):
        print(f"The input data for calibration is {data['img'].shape}")
        model(return_loss=False, return_backbone=True, **data)

    # remove and save scaling_diag_matrix
    all_scaling_diag_matrix = {}
    for name, module in model.named_modules():
        if isinstance(
                module,
                nn.Linear) and name.find('backbone') >= 0 and find_calib_key(
                    name, keys):
            module._forward_hooks.clear()
            all_scaling_diag_matrix[name] = module.scaling_diag_matrix
    print(all_scaling_diag_matrix.keys())


@torch.no_grad()
def calib_cov_distribution(model, calib_loader, keys=['ffn.layers.1']):
    model.eval()
    print(f"building covariance file for fscil")

    def hook(module, input, output, module_name):
        if isinstance(input, dict):
            print(f'input keys: {input.keys()}')
            input = input['img']
        elif isinstance(input, tuple):
            input = input[0]
        if len(input.shape) == 3:
            input = input.reshape(-1, input.shape[-1])
        input = input.detach().squeeze(0).data
        input = input.float()
        input = input / torch.max(input).abs()

        if torch.isnan(input).any():
            print("nan detected")
            raise Exception("nan in input, break")
        if torch.isinf(input).any():
            print("inf detected")
            raise Exception("inf in input, break")

        covariance = input.t().matmul(input)
        if torch.isnan(covariance).any():
            print("nan detected")
            raise Exception("nan in covariance, break")
        if torch.isinf(covariance).any():
            print("inf detected")
            raise Exception("inf in covariance, break")
        module.covariance_matrix += covariance / 256
        del covariance, input

    for name, module in model.named_modules():
        if isinstance(
                module,
                nn.Linear) and name.find('backbone') >= 0 and find_calib_key(
            name, keys):
            module.covariance_matrix = 0
            module.register_forward_hook(
                (lambda module, input, output, name=name: hook(
                    module, input, output, name)))

    for data in tqdm(calib_loader):
        print(f"The input data for calibration is {data['img'].shape}")
        model(return_loss=False, return_backbone=True, **data)

    all_covariance_matrix = {}
    for name, module in model.named_modules():
        if isinstance(
                module,
                nn.Linear) and name.find('backbone') >= 0 and find_calib_key(
                    name, keys):
            module._forward_hooks.clear()
            if torch.isnan(module.covariance_matrix).any():
                print("nan detected")
                raise Exception("nan in covariance")
            if torch.isinf(module.covariance_matrix).any():
                print("inf detected")
                raise Exception("inf in covariance")
            module.covariance_matrix = module.covariance_matrix  #/ 256
            all_covariance_matrix[name] = module.covariance_matrix
    print("covariance matrices saved")
