"""
YOLOv8 Pruning Module

L2-norm 기반 Structured Pruning을 위한 함수들입니다.

주요 함수:
    - yolov8_pruning: YOLOv8 모델 전체에 pruning 적용
    - get_filter_pruning_idx: Pruning할 필터 인덱스 계산
    - filter_pruning: Conv 레이어 필터 마스킹
    - bn_pruning: BatchNorm 레이어 채널 마스킹
"""

import torch


def get_filter_pruning_sparsity(layer, memory, max_memory):
    """
    메모리 제약 기반으로 sparsity를 계산합니다.

    Args:
        layer: Conv2d 레이어
        memory: 필터당 메모리 사용량
        max_memory: 최대 허용 메모리

    Returns:
        sparsity: 제거할 필터 비율 (0.0 ~ 1.0)
    """
    with torch.no_grad():
        num_filters = layer.weight.shape[0]
        survive_limit = max_memory // memory
        if survive_limit > num_filters:
            survive_limit = num_filters
        sparsity = (num_filters - survive_limit) / num_filters
    return sparsity


def get_filter_pruning_idx(layer, sparsity):
    """
    L2 norm 기준으로 pruning할 필터 인덱스를 계산합니다.

    Args:
        layer: Conv2d 레이어
        sparsity: 제거할 비율 (0.0 ~ 1.0)

    Returns:
        pruning_idx: 제거할 필터 인덱스
    """
    with torch.no_grad():
        weight = layer.weight
        num_filters = weight.shape[0]
        num_pruning_filters = int(num_filters * sparsity)

        # 각 필터의 L2 norm 계산
        filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)

        # 가장 작은 norm을 가진 필터 선택
        _, pruning_idx = torch.topk(filter_norms, num_pruning_filters, largest=False)

    return pruning_idx


def filter_pruning(layer, pruning_idx):
    """
    Conv 레이어의 지정된 필터를 0으로 마스킹합니다.

    Args:
        layer: Conv2d 레이어
        pruning_idx: 제거할 필터 인덱스
    """
    with torch.no_grad():
        layer.weight[pruning_idx, :, :, :] = 0.0


def bn_pruning(layer, pruning_idx):
    """
    BatchNorm 레이어의 지정된 채널을 마스킹합니다.

    Args:
        layer: BatchNorm2d 레이어
        pruning_idx: 제거할 채널 인덱스
    """
    with torch.no_grad():
        layer.weight[pruning_idx] = 0.0
        layer.bias[pruning_idx] = 0.0
        layer.running_mean[pruning_idx] = 0.0
        layer.running_var[pruning_idx] = 1.0  # var=1로 설정하여 나눗셈 오류 방지


def yolov8_pruning(model, sparsity):
    """
    YOLOv8 모델에 structured pruning을 적용합니다.

    Args:
        model: YOLOv8 모델 (model.model.model 형태)
        sparsity: 제거할 필터 비율 (0.0 ~ 1.0)

    Example:
        >>> from ultralytics import YOLO
        >>> model = YOLO('yolov8n.pt')
        >>> yolov8_pruning(model.model.model, sparsity=0.3)
    """
    block_list = [model[i] for i in range(23)]

    for i, block in enumerate(block_list):
        block_type = type(block).__name__

        # Conv 블록 처리
        if block_type == 'Conv':
            pruning_idx = get_filter_pruning_idx(block.conv, sparsity)
            filter_pruning(block.conv, pruning_idx)
            bn_pruning(block.bn, pruning_idx)

        # C2f, SPPF 블록 처리
        elif block_type in ['C2f', 'SPPF']:
            pruning_idx = get_filter_pruning_idx(block.cv2.conv, sparsity)
            filter_pruning(block.cv2.conv, pruning_idx)
            bn_pruning(block.cv2.bn, pruning_idx)

        # Detect 블록 처리
        elif block_type == 'Detect':
            for scale_idx in range(3):
                for layer_idx in range(2):
                    # cv2 레이어 pruning
                    pruning_idx = get_filter_pruning_idx(
                        block.cv2[scale_idx][layer_idx].conv, sparsity
                    )
                    filter_pruning(block.cv2[scale_idx][layer_idx].conv, pruning_idx)
                    bn_pruning(block.cv2[scale_idx][layer_idx].bn, pruning_idx)

                    # cv3 레이어 pruning
                    pruning_idx = get_filter_pruning_idx(
                        block.cv3[scale_idx][layer_idx].conv, sparsity
                    )
                    filter_pruning(block.cv3[scale_idx][layer_idx].conv, pruning_idx)
                    bn_pruning(block.cv3[scale_idx][layer_idx].bn, pruning_idx)
