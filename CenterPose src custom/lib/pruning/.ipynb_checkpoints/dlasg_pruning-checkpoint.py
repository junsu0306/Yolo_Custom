import torch
import torch.nn as nn
from lib.models.networks.pose_dla_dcn import DLASeg


def filter_pruning(layer, pruning_idx):
    with torch.no_grad():

        pruning_idx = pruning_idx[pruning_idx < layer.weight.shape[0]]
        if pruning_idx.numel() == 0:
            return
        layer.weight[pruning_idx, :, :, :] = 0.0


def bn_pruning(layer, pruning_idx):
    weight = layer.weight
    bias = layer.bias
    mean = layer.running_mean
    var = layer.running_var

    pruning_idx = pruning_idx[pruning_idx < weight.shape[0]]

    if pruning_idx.numel() == 0:
        return  

    with torch.no_grad():
        weight[pruning_idx] = 0.0
        bias[pruning_idx] = 0.0
        mean[pruning_idx] = 0.0
        var[pruning_idx] = 1.0


def get_filter_norms(layer, inf=99999):
    with torch.no_grad():
        # Reshape the weight tensor so that each filter is flattened, then calculate the L2 norm along the flattened dimension.
        filter_norms = torch.norm(layer.weight.view(layer.weight.shape[0], -1), dim=1)

        # Find the index of the maximum value in filter_norms
        max_idx = torch.argmax(filter_norms)

        # Set the maximum value to inf
        filter_norms[max_idx] = inf

        return filter_norms

def get_pruning_indices(filter_norms, sparsity):

    all_norms = torch.cat(filter_norms)
    num_pruning_filters = int(all_norms.numel() * sparsity)

    # Get the global indices of filters to prune based on the top-k smallest norms.
    _, global_pruning_idx = torch.topk(all_norms, num_pruning_filters, largest=False)

    pruning_indices = []
    current_position = 0 # Track the position in the global indices.

    # Iterate through each layer's norms and determine which filters should be pruned.
    for norms in filter_norms:
        layer_size = norms.numel() # Get the number of filters in the current layer.

        # Identify the global indices that correspond to the current layer.
        pruning_idx = global_pruning_idx[(global_pruning_idx >= current_position) & (global_pruning_idx < current_position + layer_size)]

        # Adjust the global indices to local indices relative to the current layer.
        pruning_idx -= current_position

        # Store the local indices for pruning in the current layer.
        pruning_indices.append(pruning_idx)
        current_position += layer_size # Update the position to the start of the next layer.

    return pruning_indices
    

def dlasg_blockwise_pruning(model, sparsity, device='cpu'):
    model = model.to(device)
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'BasicBlock':
            conv1, bn1 = module.conv1, module.bn1
            conv2, bn2 = module.conv2, module.bn2

            norms1 = get_filter_norms(conv1)
            prune_idx1 = get_pruning_indices([norms1], sparsity)[0]
            if prune_idx1.numel() == 0:
                continue

            filter_pruning(conv1, prune_idx1)
            bn_pruning(bn1, prune_idx1)

    return model

def reduce_pruned_model(model):
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'BasicBlock':
            conv1, bn1 = module.conv1, module.bn1
            conv2, bn2 = module.conv2, module.bn2

            # 남아있는 출력 채널
            keep = torch.where(
                conv1.weight.view(conv1.weight.shape[0], -1).abs().sum(1) != 0
            )[0]
            
            # 새로운 conv1
            new_conv1 = nn.Conv2d(
                in_channels=conv1.in_channels,
                out_channels=keep.numel(),
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                dilation=conv1.dilation,
                groups=conv1.groups,
                bias=(conv1.bias is not None)
            )
            new_conv1.weight.data = conv1.weight[keep].clone()
            if conv1.bias is not None:
                new_conv1.bias.data = conv1.bias.data[keep].clone()

            # 새로운 conv2
            new_conv2 = nn.Conv2d(
                in_channels=keep.numel(),
                out_channels=conv2.out_channels,
                kernel_size=conv2.kernel_size,
                stride=conv2.stride,
                padding=conv2.padding,
                dilation=conv2.dilation,
                groups=conv2.groups,
                bias=(conv2.bias is not None)
            )
            new_conv2.weight.data = conv2.weight[:, keep].clone()
            if conv2.bias is not None:
                new_conv2.bias.data = conv2.bias.data.clone()

            _assign_module(model, f"{name}.conv1", new_conv1)
            _assign_module(model, f"{name}.conv2", new_conv2)

            # BatchNorm1 업데이트
            bn_weights = bn1.weight.data
            keep_bn = torch.where(bn_weights != 0)[0]
            if keep_bn.numel() < bn_weights.shape[0]:
                new_bn = nn.BatchNorm2d(keep_bn.numel())
                new_bn.weight.data = bn1.weight.data[keep_bn].clone()
                new_bn.bias.data = bn1.bias.data[keep_bn].clone()
                new_bn.running_mean = bn1.running_mean[keep_bn].clone()
                new_bn.running_var = bn1.running_var[keep_bn].clone()
                _assign_module(model, f"{name}.bn1", new_bn)

    return model



def _assign_module(model, name, new_module):

    names = name.split('.')
    mod = model
    for n in names[:-1]:
        mod = getattr(mod, n)
    setattr(mod, names[-1], new_module)
