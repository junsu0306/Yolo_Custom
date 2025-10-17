import torch
import torch.nn as nn
from lib.models.networks.pose_dla_dcn import DLASeg
import copy

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
    with torch.no_grad():  # Ensure no gradients are tracked during norm calculation.
        # Reshape the weight tensor so that each filter is flattened, then calculate the L2 norm along the flattened dimension.
        filter_norms = torch.norm(layer.weight.view(layer.weight.shape[0], -1), dim=1)

        # Find the index of the maximum value in filter_norms
        max_idx = torch.argmax(filter_norms)

        # Set the maximum value to inf
        filter_norms[max_idx] = inf

        return filter_norms

def get_pruning_indices(filter_norms, sparsity):

    # Concatenate all layer norms into a single tensor to perform global pruning.
    all_norms = torch.cat(filter_norms)
    num_pruning_filters = int(all_norms.numel() * sparsity) # Calculate total number of filters to prune globally.

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
    
# def dlasg_pruning(model, sparsity):
#     """
#     Perform pruning on the DLASeg model based on the specified sparsity ratio.

#     Parameters:
#     model (torch.nn.Module): The DLASeg model containing layers to be pruned.
#     sparsity (float): The global sparsity ratio for pruning.
#     """
#     conv_bn_pairs = []
#     filter_norms = []

#     # 수동으로 Conv2d와 BatchNorm2d 페어링
#     previous_conv = None
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Conv2d):
#             previous_conv = module
#         elif isinstance(module, torch.nn.BatchNorm2d) and previous_conv is not None:
#             conv_bn_pairs.append((previous_conv, module))
#             filter_norms.append(get_filter_norms(previous_conv))
#             previous_conv = None

#     # Calculate pruning indices based on global sparsity
#     pruning_ids = get_pruning_indices(filter_norms, sparsity)

#     # Apply pruning to each Conv-BN pair
#     for idx, (conv, bn) in enumerate(conv_bn_pairs):
#         pruning_idx = pruning_ids[idx]
#         filter_pruning(conv, pruning_idx)
#         bn_pruning(bn, pruning_idx)

#     print("DLASeg pruning completed successfully!")

    
#     return model

    
# def get_basicblock_conv_pairs(model):

#     conv_bn_pairs = []
#     filter_norms = []

#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.modules.conv.Conv2d):
#             continue  # 일괄 탐색 방지
#         if module.__class__.__name__ == 'BasicBlock':
#             # conv1과 bn1
#             conv1 = module.conv1
#             bn1 = module.bn1
#             conv_bn_pairs.append((conv1, bn1))
#             filter_norms.append(get_filter_norms(conv1))

#             # conv2와 bn2
#             conv2 = module.conv2
#             bn2 = module.bn2
#             conv_bn_pairs.append((conv2, bn2))
#             filter_norms.append(get_filter_norms(conv2))

#     return conv_bn_pairs, filter_norms

# def get_basicblock_conv_pairs(model, sparsity):

#     conv_bn_pairs, filter_norms = get_basicblock_conv_pairs(model)

#     # Global pruning index 계산
#     pruning_ids = get_pruning_indices(filter_norms, sparsity)

#     # 각 conv-BN에 pruning 적용
#     for idx, (conv, bn) in enumerate(conv_bn_pairs):
#         pruning_idx = pruning_ids[idx]
#         filter_pruning(conv, pruning_idx)
#         bn_pruning(bn, pruning_idx)

#     # print("BasicBlock conv1/conv2 pruning completed!")
#     return model

def _assign_module(model, name, new_module):

    names = name.split('.')
    mod = model
    for n in names[:-1]:
        mod = getattr(mod, n)
    setattr(mod, names[-1], new_module)


def dlasg_blockwise_pruning(model, sparsity):

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'BasicBlock':
            conv1 = module.conv1
            bn1 = module.bn1
            conv2 = module.conv2
            bn2 = module.bn2

            # conv1 출력 채널 L2 norm 기준으로 pruning 대상 계산
            conv1_norms = get_filter_norms(conv1)
            pruning_idx = get_pruning_indices([conv1_norms], sparsity)[0]

            if pruning_idx.numel() == 0:
                continue

            # conv1의 출력 채널 pruning
            filter_pruning(conv1, pruning_idx)
            bn_pruning(bn1, pruning_idx)

            # conv2의 입력 채널 pruning (주의: 입력 기준)
            with torch.no_grad():
                keep_idx = torch.tensor([i for i in range(conv2.in_channels) if i not in pruning_idx.cpu().numpy()])

                if keep_idx.numel() != conv2.in_channels:
                    new_conv2 = nn.Conv2d(
                        in_channels=keep_idx.numel(),
                        out_channels=conv2.out_channels,
                        kernel_size=conv2.kernel_size,
                        stride=conv2.stride,
                        padding=conv2.padding,
                        dilation=conv2.dilation,
                        groups=conv2.groups,
                        bias=(conv2.bias is not None)
                    )

                    new_conv2.weight.data = conv2.weight.data[:, keep_idx].clone()
                    if conv2.bias is not None:
                        new_conv2.bias.data = conv2.bias.data.clone()

                    _assign_module(model, f"{name}.conv2", new_conv2)

    return model
