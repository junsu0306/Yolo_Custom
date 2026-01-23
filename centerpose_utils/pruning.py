"""
CenterPose (DLA-34) Pruning Module

DLA-34 모델의 BasicBlock 단위로 L2-norm 기반 Structured Pruning을 적용합니다.

주요 함수:
    - dlasg_blockwise_pruning: BasicBlock 단위 pruning 적용
    - get_filter_norms: 각 필터의 L2 norm 계산
    - get_pruning_indices: Global sparsity 기준 pruning 인덱스 계산
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

    Example:
        >>> from lib.models.model import create_model, load_model
        >>> model = create_model('dla_34', heads, head_conv)
        >>> model = load_model(model, 'pretrained.pth')
        >>> pruned_model = dlasg_blockwise_pruning(model, sparsity=0.3)
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
