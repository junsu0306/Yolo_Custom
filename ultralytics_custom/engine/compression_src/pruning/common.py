import torch


def get_filter_pruning_sparsity(layer, memory, max_memory):
    with torch.no_grad():
        weight = layer.weight
        num_filters = weight.shape[0]
        survive_limit = max_memory // memory
        if survive_limit > num_filters:
            survive_limit = num_filters
        sparsity = (num_filters - survive_limit) / num_filters
    return sparsity


def filter_pruning(layer, pruning_idx):
    weight = layer.weight
    with torch.no_grad():
        weight[pruning_idx, :, :, :] = 0.0


def bn_pruning(layer, pruning_idx):
    weight = layer.weight
    bias = layer.bias
    mean = layer.running_mean
    var = layer.running_var
    with torch.no_grad():
        weight[pruning_idx] = 0.0
        bias[pruning_idx] = 0.0
        mean[pruning_idx] = 0.0
        var[pruning_idx] = 1.0


def get_filter_pruning_idx(layer, sparsity):
    with torch.no_grad():
        weight = layer.weight
        num_filters = weight.shape[0]
        num_pruning_filters = int(num_filters * sparsity)
        filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)
        _, pruning_idx = torch.topk(filter_norms, num_pruning_filters, largest=False)
    return pruning_idx
