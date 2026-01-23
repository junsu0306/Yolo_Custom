"""
DLA-34 Structured Pruning Module

DLA-34 모델의 BasicBlock 단위로 L2-norm 기반 Structured Pruning을 적용합니다.

주요 함수:
    - dlasg_blockwise_pruning: BasicBlock 단위 pruning 적용
    - reduce_pruned_model: 0으로 마스킹된 필터를 물리적으로 제거
"""

import torch
import torch.nn as nn


def filter_pruning(layer, pruning_idx):
    """
    Conv 레이어의 지정된 필터를 0으로 마스킹합니다.

    Args:
        layer: Conv2d 레이어
        pruning_idx: 제거할 필터 인덱스
    """
    with torch.no_grad():
        pruning_idx = pruning_idx[pruning_idx < layer.weight.shape[0]]
        if pruning_idx.numel() == 0:
            return
        layer.weight[pruning_idx, :, :, :] = 0.0


def bn_pruning(layer, pruning_idx):
    """
    BatchNorm 레이어의 지정된 채널을 마스킹합니다.

    Args:
        layer: BatchNorm2d 레이어
        pruning_idx: 제거할 채널 인덱스
    """
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
        var[pruning_idx] = 1.0  # var=1로 설정하여 나눗셈 오류 방지


def get_filter_norms(layer, inf=99999):
    """
    각 필터의 L2 norm을 계산합니다.
    가장 큰 norm을 가진 필터는 보호합니다 (inf로 설정).

    Args:
        layer: Conv2d 레이어
        inf: 최대 norm 필터에 설정할 값 (pruning 방지)

    Returns:
        filter_norms: 각 필터의 L2 norm 텐서
    """
    with torch.no_grad():
        # 각 필터를 flatten하여 L2 norm 계산
        filter_norms = torch.norm(
            layer.weight.view(layer.weight.shape[0], -1), dim=1
        )

        # 최대 norm 필터 보호 (inf로 설정하여 pruning 대상에서 제외)
        max_idx = torch.argmax(filter_norms)
        filter_norms[max_idx] = inf

        return filter_norms


def get_pruning_indices(filter_norms, sparsity):
    """
    Global sparsity 기준으로 pruning할 필터 인덱스를 계산합니다.

    Args:
        filter_norms: 각 레이어의 필터 norm 리스트
        sparsity: 제거할 비율 (0.0 ~ 1.0)

    Returns:
        pruning_indices: 각 레이어별 pruning 인덱스 리스트
    """
    all_norms = torch.cat(filter_norms)
    num_pruning_filters = int(all_norms.numel() * sparsity)

    # 가장 작은 norm을 가진 필터들 선택 (global pruning)
    _, global_pruning_idx = torch.topk(all_norms, num_pruning_filters, largest=False)

    pruning_indices = []
    current_position = 0

    # 전역 인덱스를 각 레이어의 로컬 인덱스로 변환
    for norms in filter_norms:
        layer_size = norms.numel()

        # 현재 레이어에 해당하는 전역 인덱스 추출
        pruning_idx = global_pruning_idx[
            (global_pruning_idx >= current_position) &
            (global_pruning_idx < current_position + layer_size)
        ]

        # 로컬 인덱스로 변환
        pruning_idx = pruning_idx - current_position

        pruning_indices.append(pruning_idx)
        current_position += layer_size

    return pruning_indices


def dlasg_blockwise_pruning(model, sparsity, device='cpu'):
    """
    DLA-34 모델의 BasicBlock 단위로 pruning을 적용합니다.

    Args:
        model: DLA-34 모델
        sparsity: 제거할 필터 비율 (0.0 ~ 1.0)
        device: 연산 디바이스 ('cpu' 또는 'cuda:0')

    Returns:
        model: Pruning이 적용된 모델
    """
    model = model.to(device)

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'BasicBlock':
            conv1, bn1 = module.conv1, module.bn1

            # Conv1의 필터 norm 계산
            norms1 = get_filter_norms(conv1)

            # Pruning 인덱스 계산
            prune_idx1 = get_pruning_indices([norms1], sparsity)[0]
            if prune_idx1.numel() == 0:
                continue

            # Pruning 적용
            filter_pruning(conv1, prune_idx1)
            bn_pruning(bn1, prune_idx1)

    return model


def reduce_pruned_model(model):
    """
    0으로 마스킹된 필터를 물리적으로 제거하여 모델 크기를 축소합니다.

    Args:
        model: Pruning이 적용된 모델

    Returns:
        model: 물리적으로 축소된 모델
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
