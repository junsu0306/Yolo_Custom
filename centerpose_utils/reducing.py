"""
CenterPose (DLA-34) Reducing Module

Pruning된 모델에서 0인 필터를 물리적으로 제거하기 위한 함수들입니다.

주요 함수:
    - reduce_pruned_model: 0으로 마스킹된 필터를 물리적으로 제거
    - get_survived_filter_idx: 살아남은 필터 인덱스 반환
    - conv_reduce: Conv 레이어 축소
    - bn_reduce: BatchNorm 레이어 축소
"""

import torch
import torch.nn as nn


def get_survived_filter_idx(layer):
    """
    살아남은 필터 인덱스를 반환합니다 (norm != 0).

    Args:
        layer: Conv2d 레이어

    Returns:
        survived_filter_idx: 0이 아닌 필터의 인덱스
    """
    weight = layer.weight
    num_filters = weight.shape[0]
    filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)
    survived_filter_idx = torch.where(filter_norms != 0)[0]
    return survived_filter_idx


def conv_reduce(layer, reduced_layer, survived_out_channels_idx, survived_in_channels_idx):
    """
    Conv 레이어를 축소합니다.

    Args:
        layer: 원본 Conv2d 레이어
        reduced_layer: 축소된 Conv2d 레이어
        survived_out_channels_idx: 살아남은 출력 채널 인덱스
        survived_in_channels_idx: 살아남은 입력 채널 인덱스
    """
    # 채널 수 설정
    reduced_layer.in_channels = len(survived_in_channels_idx)
    reduced_layer.out_channels = len(survived_out_channels_idx)

    # Depthwise conv의 경우 groups 조정
    if reduced_layer.groups != 1:
        reduced_layer.groups = reduced_layer.out_channels

    # 새로운 weight 텐서 생성
    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(
            reduced_layer.out_channels,
            reduced_layer.in_channels,
            reduced_layer.kernel_size[0],
            reduced_layer.kernel_size[1],
        ),
        requires_grad=True,
    )

    # 검증
    assert len(survived_out_channels_idx) == reduced_layer.weight.shape[0], \
        f"out_channels mismatch: {len(survived_out_channels_idx)} vs {reduced_layer.weight.shape[0]}"
    assert len(survived_in_channels_idx) == reduced_layer.weight.shape[1], \
        f"in_channels mismatch: {len(survived_in_channels_idx)} vs {reduced_layer.weight.shape[1]}"

    # 살아남은 가중치 복사
    with torch.no_grad():
        reduced_layer.weight.copy_(
            layer.weight[survived_out_channels_idx, :, :, :][:, survived_in_channels_idx, :, :]
        )


def bn_reduce(layer, reduced_layer, survived_features_idx):
    """
    BatchNorm 레이어를 축소합니다.

    Args:
        layer: 원본 BatchNorm2d 레이어
        reduced_layer: 축소된 BatchNorm2d 레이어
        survived_features_idx: 살아남은 feature 인덱스
    """
    reduced_layer.num_features = len(survived_features_idx)

    # 새로운 파라미터 텐서 생성
    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(reduced_layer.num_features),
        requires_grad=True,
    )
    reduced_layer.bias = torch.nn.Parameter(
        data=torch.zeros(reduced_layer.num_features),
        requires_grad=True,
    )
    reduced_layer.running_mean = torch.zeros(reduced_layer.num_features)
    reduced_layer.running_var = torch.zeros(reduced_layer.num_features)

    # 검증
    assert len(survived_features_idx) == reduced_layer.weight.shape[0], "bn weight mismatch"
    assert len(survived_features_idx) == reduced_layer.bias.shape[0], "bn bias mismatch"
    assert len(survived_features_idx) == reduced_layer.running_mean.shape[0], "bn mean mismatch"
    assert len(survived_features_idx) == reduced_layer.running_var.shape[0], "bn var mismatch"

    # 살아남은 파라미터 복사
    with torch.no_grad():
        reduced_layer.weight.copy_(layer.weight[survived_features_idx])
        reduced_layer.bias.copy_(layer.bias[survived_features_idx])
        reduced_layer.running_mean.copy_(layer.running_mean[survived_features_idx])
        reduced_layer.running_var.copy_(layer.running_var[survived_features_idx])


def copy_layer(layer, reduced_layer):
    """
    레이어 파라미터를 그대로 복사합니다.

    Args:
        layer: 원본 레이어
        reduced_layer: 대상 레이어
    """
    with torch.no_grad():
        for p, reduced_p in zip(layer.parameters(), reduced_layer.parameters()):
            reduced_p.copy_(p)


def bn_copy_layer(layer, reduced_layer):
    """
    BatchNorm 레이어를 그대로 복사합니다 (running_mean, running_var 포함).

    Args:
        layer: 원본 BatchNorm 레이어
        reduced_layer: 대상 BatchNorm 레이어
    """
    with torch.no_grad():
        reduced_layer.running_mean.copy_(layer.running_mean)
        reduced_layer.running_var.copy_(layer.running_var)

        for p, reduced_p in zip(layer.parameters(), reduced_layer.parameters()):
            reduced_p.copy_(p)


def _assign_module(model, name, new_module):
    """
    모델의 서브모듈을 새로운 모듈로 교체합니다.

    Args:
        model: 대상 모델
        name: 모듈 경로 (예: "base.layer1.0.conv1")
        new_module: 교체할 새 모듈
    """
    names = name.split('.')
    mod = model
    for n in names[:-1]:
        mod = getattr(mod, n)
    setattr(mod, names[-1], new_module)


def reduce_pruned_model(model):
    """
    0으로 마스킹된 필터를 물리적으로 제거하여 모델 크기를 축소합니다.

    Args:
        model: Pruning이 적용된 모델

    Returns:
        model: 물리적으로 축소된 모델

    Example:
        >>> from centerpose_utils import dlasg_blockwise_pruning, reduce_pruned_model
        >>> pruned_model = dlasg_blockwise_pruning(model, sparsity=0.3)
        >>> reduced_model = reduce_pruned_model(pruned_model)
        >>> torch.save(reduced_model, 'reduced_model.pth')
    """
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'BasicBlock':
            conv1, bn1 = module.conv1, module.bn1
            conv2, bn2 = module.conv2, module.bn2

            # 살아남은 채널 인덱스 찾기 (weight sum != 0)
            keep = torch.where(
                conv1.weight.view(conv1.weight.shape[0], -1).abs().sum(1) != 0
            )[0]

            if keep.numel() == conv1.out_channels:
                continue  # 모든 채널이 살아있으면 스킵

            # 새로운 Conv1 생성 (출력 채널 축소)
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

            # 새로운 Conv2 생성 (입력 채널 축소)
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

            # 모듈 교체
            _assign_module(model, f"{name}.conv1", new_conv1)
            _assign_module(model, f"{name}.conv2", new_conv2)

            # BatchNorm1 축소
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
