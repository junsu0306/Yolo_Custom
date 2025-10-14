from .common import get_filter_pruning_idx, filter_pruning, bn_pruning


def resnet18_pruning(model, sparsity):
    block_list = [model.layer1, model.layer2, model.layer3, model.layer4]
    for block in block_list:
        for i in range(2):
            pruning_idx = get_filter_pruning_idx(
                layer=block[i].conv1, sparsity=sparsity
            )
            filter_pruning(layer=block[i].conv1, pruning_idx=pruning_idx)
            bn_pruning(layer=block[i].bn1, pruning_idx=pruning_idx)
