import torch
from .common import (
    get_survived_filter_idx,
    conv_reduce,
    bn_reduce,
    copy_layer,
    bn_copy_layer,
)


def resnet18_reduce(model, reduced_model):
    survived_idx = get_survived_filter_idx(model.conv1)
    conv_reduce(
        layer=model.conv1,
        reduced_layer=reduced_model.conv1,
        survived_out_channels_idx=survived_idx,
        survived_in_channels_idx=torch.arange(3),
    )
    bn_reduce(model.bn1, reduced_model.bn1, survived_idx)

    block_list = [model.layer1, model.layer2, model.layer3, model.layer4]
    reduced_block_list = [
        reduced_model.layer1,
        reduced_model.layer2,
        reduced_model.layer3,
        reduced_model.layer4,
    ]
    for block, reduced_block in zip(block_list, reduced_block_list):
        for i in range(2):
            prev_survived_idx = survived_idx
            survived_idx = get_survived_filter_idx(block[i].conv1)
            conv_reduce(
                layer=block[i].conv1,
                reduced_layer=reduced_block[i].conv1,
                survived_out_channels_idx=survived_idx,
                survived_in_channels_idx=prev_survived_idx,
            )
            bn_reduce(block[i].bn1, reduced_block[i].bn1, survived_idx)

            prev_survived_idx = survived_idx
            survived_idx = get_survived_filter_idx(block[i].conv2)
            conv_reduce(
                layer=block[i].conv2,
                reduced_layer=reduced_block[i].conv2,
                survived_out_channels_idx=survived_idx,
                survived_in_channels_idx=prev_survived_idx,
            )
            bn_reduce(block[i].bn2, reduced_block[i].bn2, survived_idx)

    copy_layer(model.layer2[0].shortcut[0], reduced_model.layer2[0].shortcut[0])
    copy_layer(model.layer3[0].shortcut[0], reduced_model.layer3[0].shortcut[0])
    copy_layer(model.layer4[0].shortcut[0], reduced_model.layer4[0].shortcut[0])

    bn_copy_layer(model.layer2[0].shortcut[1], reduced_model.layer2[0].shortcut[1])
    bn_copy_layer(model.layer3[0].shortcut[1], reduced_model.layer3[0].shortcut[1])
    bn_copy_layer(model.layer4[0].shortcut[1], reduced_model.layer4[0].shortcut[1])
    copy_layer(model.linear, reduced_model.linear)
