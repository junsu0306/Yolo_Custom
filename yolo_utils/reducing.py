"""
YOLOv8 Reducing Module

Pruning된 모델에서 0인 필터를 물리적으로 제거하기 위한 함수들입니다.

주요 함수:
    - yolov8_reducing: YOLOv8 모델 전체에 reducing 적용
    - get_survived_filter_idx: 살아남은 필터 인덱스 반환
    - conv_reduce: Conv 레이어 축소
    - bn_reduce: BatchNorm 레이어 축소
"""

import torch


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


def fc_reduce(layer, reduced_layer, survived_out_features_idx, survived_in_features_idx):
    """
    Fully Connected 레이어를 축소합니다.

    Args:
        layer: 원본 Linear 레이어
        reduced_layer: 축소된 Linear 레이어
        survived_out_features_idx: 살아남은 출력 feature 인덱스
        survived_in_features_idx: 살아남은 입력 feature 인덱스
    """
    # Feature 수 설정
    reduced_layer.in_features = len(survived_in_features_idx)
    reduced_layer.out_features = len(survived_out_features_idx)

    # 새로운 weight, bias 텐서 생성
    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(reduced_layer.out_features, reduced_layer.in_features),
        requires_grad=True,
    )
    reduced_layer.bias = torch.nn.Parameter(
        data=torch.zeros(reduced_layer.out_features),
        requires_grad=True,
    )

    # 검증
    assert len(survived_out_features_idx) == reduced_layer.weight.shape[0], \
        f"out_features mismatch: {len(survived_out_features_idx)} vs {reduced_layer.weight.shape[0]}"
    assert len(survived_in_features_idx) == reduced_layer.weight.shape[1], \
        f"in_features mismatch: {len(survived_in_features_idx)} vs {reduced_layer.weight.shape[1]}"

    # 살아남은 가중치 복사
    with torch.no_grad():
        reduced_layer.weight.copy_(
            layer.weight[survived_out_features_idx, :][:, survived_in_features_idx]
        )
        reduced_layer.bias.copy_(layer.bias[survived_out_features_idx])


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


def _get_concat_survived_idx(block_list, layer_idx):
    """
    Concatenation 레이어의 입력 채널 인덱스를 계산합니다.
    YOLOv8의 FPN 구조에서 여러 레이어의 출력이 연결되는 경우 처리합니다.
    """
    # 레이어 인덱스에 따른 연결 레이어 매핑
    concat_map = {
        11: (8, 5),
        14: (11, 3),
        17: (15, 11),
        20: (18, 8),
    }
    n1, n2 = concat_map[layer_idx]

    # 첫 번째 연결 레이어의 살아남은 인덱스
    if layer_idx in [17, 20]:
        p_surv_idx_1 = get_survived_filter_idx(block_list[n1].conv)
        offset = block_list[n1].conv.out_channels
    else:
        p_surv_idx_1 = get_survived_filter_idx(block_list[n1].cv2.conv)
        offset = block_list[n1].cv2.conv.out_channels

    # 두 번째 연결 레이어의 살아남은 인덱스 (offset 적용)
    p_surv_idx_2 = get_survived_filter_idx(block_list[n2].cv2.conv) + offset

    return torch.concat([p_surv_idx_1, p_surv_idx_2])


def _reduce_detect_block(block, reduced_block, block_list):
    """
    Detect 블록의 cv2, cv3 레이어를 축소합니다.
    """
    # 이전 레이어 인덱스 (FPN 출력)
    prev_indices = [14, 17, 20]

    for scale_idx, prev_idx in enumerate(prev_indices):
        for sub_layer, reduced_sub_layer in zip(
            [block.cv2, block.cv3],
            [reduced_block.cv2, reduced_block.cv3]
        ):
            prev_survived_idx = get_survived_filter_idx(block_list[prev_idx].cv2.conv)

            # Conv+BN 레이어 2개 처리
            for layer_idx in range(2):
                survived_idx = get_survived_filter_idx(sub_layer[scale_idx][layer_idx].conv)
                conv_reduce(
                    sub_layer[scale_idx][layer_idx].conv,
                    reduced_sub_layer[scale_idx][layer_idx].conv,
                    survived_idx,
                    prev_survived_idx,
                )
                bn_reduce(
                    sub_layer[scale_idx][layer_idx].bn,
                    reduced_sub_layer[scale_idx][layer_idx].bn,
                    survived_idx,
                )
                prev_survived_idx = survived_idx

            # 마지막 Conv2d 레이어 (출력 채널은 유지)
            survived_idx = get_survived_filter_idx(sub_layer[scale_idx][2])
            conv_reduce(
                sub_layer[scale_idx][2],
                reduced_sub_layer[scale_idx][2],
                survived_idx,
                prev_survived_idx,
            )


def yolov8_reducing(model, reduced_model):
    """
    Pruning된 YOLOv8 모델에서 0인 필터를 물리적으로 제거합니다.

    Args:
        model: Pruning이 적용된 원본 모델 (model.model.model)
        reduced_model: 축소된 가중치가 복사될 대상 모델 (model.model.model)

    Example:
        >>> from ultralytics import YOLO
        >>> # 학습된 pruned 모델 로드
        >>> model = YOLO('runs/detect/train/weights/best.pt')
        >>> reduced_model = YOLO('yolov8n.yaml')
        >>> yolov8_reducing(model.model.model, reduced_model.model.model)
        >>> torch.save(reduced_model.model.state_dict(), 'compressed.pt')
    """
    # 첫 번째 Conv 레이어 처리 (RGB 입력)
    survived_idx = get_survived_filter_idx(model[0].conv)
    conv_reduce(
        model[0].conv,
        reduced_model[0].conv,
        survived_idx,
        torch.arange(3),  # RGB 입력 채널
    )
    bn_reduce(model[0].bn, reduced_model[0].bn, survived_idx)

    prev_survived_idx = survived_idx

    # 나머지 블록 처리
    block_list = [model[i] for i in range(1, 23)]
    reduced_block_list = [reduced_model[i] for i in range(1, 23)]

    for i, (block, reduced_block) in enumerate(zip(block_list, reduced_block_list)):
        block_type = type(block).__name__

        # Conv 블록 처리
        if block_type == 'Conv':
            survived_idx = get_survived_filter_idx(block.conv)
            conv_reduce(
                block.conv,
                reduced_block.conv,
                survived_idx,
                prev_survived_idx,
            )
            bn_reduce(block.bn, reduced_block.bn, survived_idx)

        # C2f, SPPF 블록 처리
        elif block_type in ['C2f', 'SPPF']:
            # Concatenation 레이어의 경우 특별 처리
            if i in [11, 14, 17, 20]:
                prev_survived_idx = _get_concat_survived_idx(block_list, i)

            # cv1 레이어 처리
            survived_idx = torch.arange(block.cv1.conv.out_channels)
            conv_reduce(
                block.cv1.conv,
                reduced_block.cv1.conv,
                survived_idx,
                prev_survived_idx,
            )

            # cv2 레이어 처리
            prev_survived_idx = torch.arange(block.cv2.conv.in_channels)
            survived_idx = get_survived_filter_idx(block.cv2.conv)
            conv_reduce(
                block.cv2.conv,
                reduced_block.cv2.conv,
                survived_idx,
                prev_survived_idx,
            )
            bn_reduce(block.cv2.bn, reduced_block.cv2.bn, survived_idx)

        # Detect 블록 처리
        elif block_type == 'Detect':
            _reduce_detect_block(block, reduced_block, block_list)

        prev_survived_idx = survived_idx
