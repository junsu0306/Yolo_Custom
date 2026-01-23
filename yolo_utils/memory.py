"""
YOLOv8 Memory Measurement Module

GPU 메모리 사용량 측정 및 메모리 기반 손실 함수를 제공합니다.

주요 함수:
    - measure_memory: 레이어별 메모리 사용량 측정
    - model_memory_usage: 전체 모델 메모리 측정
    - model_memory_usage_with_reducing: Reducing 후 메모리 측정
    - custom_memory_loss_function: 메모리 제약 기반 손실 함수
"""

import copy
import torch
import torch.nn as nn

from .reducing import yolov8_reducing


def measure_memory(x, layers, device):
    """
    레이어 실행 시 GPU 메모리 사용량을 측정합니다.

    Args:
        x: 입력 텐서 (또는 텐서 리스트)
        layers: 측정할 레이어 리스트
        device: GPU 디바이스

    Returns:
        x: 출력 텐서
        memory_diff: 사용된 메모리 (MB)
    """
    if isinstance(x, (tuple, list)):
        x = [item.cpu() for item in x]
    else:
        x = x.cpu()

    for layer in layers:
        layer.cpu()
    torch.cuda.empty_cache()

    before_memory = torch.cuda.memory_allocated(device) / 1024**2

    with torch.no_grad():
        if isinstance(x, (tuple, list)):
            x = [item.to(device) for item in x]
        else:
            x = x.to(device)

        for layer in layers:
            layer.to(device)
            x = layer(x)

    after_memory = torch.cuda.memory_allocated(device) / 1024**2

    if isinstance(x, (tuple, list)):
        x = [item.cpu() for item in x]
    else:
        x = x.cpu()

    for layer in layers:
        layer.cpu()
    torch.cuda.empty_cache()

    return x, after_memory - before_memory


def extract_layers(model):
    """
    모델에서 레이어 리스트를 추출합니다.

    Args:
        model: YOLOv8 모델 (model.model 형태)

    Returns:
        layers_list: 레이어 리스트
    """
    layers_list = []

    for name, layer in model.named_children():
        if isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList):
            for sub_layer in layer:
                layers_list.append(sub_layer)
        else:
            layers_list.append(layer)

    return layers_list


def model_memory_usage(x, model, device):
    """
    YOLOv8 모델의 전체 메모리 사용량을 측정합니다.

    YOLOv8의 특수 구조 (Concat, Detect 레이어)를 고려하여 측정합니다.

    Args:
        x: 입력 텐서
        model: YOLOv8 모델 (YOLO 객체)
        device: GPU 디바이스

    Returns:
        total_memory: 전체 메모리 사용량 (MB)

    Example:
        >>> from ultralytics import YOLO
        >>> model = YOLO('yolov8n.pt')
        >>> x = torch.randn(1, 3, 640, 640)
        >>> memory = model_memory_usage(x, model, 'cuda:0')
        >>> print(f"Memory: {memory:.2f} MB")
    """
    torch.cuda.empty_cache()

    layers_list = extract_layers(model.model.model)

    mem_list = []
    layer_outputs = []

    for i, layer in enumerate(layers_list):
        if type(layer).__name__ == "Concat":
            # YOLOv8 FPN Concatenation 레이어 처리
            if i == 11:
                concat_inputs = [layer_outputs[6], layer_outputs[10]]
            elif i == 14:
                concat_inputs = [layer_outputs[4], layer_outputs[13]]
            elif i == 17:
                concat_inputs = [layer_outputs[12], layer_outputs[16]]
            elif i == 20:
                concat_inputs = [layer_outputs[9], layer_outputs[19]]

            x, used_memory = measure_memory(concat_inputs, [layer], device)
            mem_list.append(used_memory)
            layer_outputs.append(x)

            setattr(layer, 'memory_usg', used_memory)
            continue

        if type(layer).__name__ == "Detect":
            # Detect 레이어는 여러 스케일의 입력을 받음
            inputs = [layer_outputs[15], layer_outputs[18], layer_outputs[21]]

            x, used_memory = measure_memory(inputs, [layer], device)
            mem_list.append(used_memory)

            setattr(layer, 'memory_usg', used_memory)
            continue

        else:
            x, used_memory = measure_memory(x, [layer], device)
            mem_list.append(used_memory)
            layer_outputs.append(x)

            setattr(layer, 'memory_usg', used_memory)

    return sum(mem_list)


def model_memory_usage_with_reducing(x, pruned_model, device):
    """
    Reducing 적용 후 YOLOv8 모델의 메모리 사용량을 측정합니다.

    Pruning된 모델을 복사하여 reducing을 적용한 후 메모리를 측정합니다.
    학습 중 실제 압축 효과를 확인할 때 사용합니다.

    Args:
        x: 입력 텐서
        pruned_model: Pruning이 적용된 YOLO 모델
        device: GPU 디바이스

    Returns:
        total_memory: Reducing 후 메모리 사용량 (MB)

    Example:
        >>> from yolo_utils import yolov8_pruning
        >>> model = YOLO('yolov8n.pt')
        >>> yolov8_pruning(model.model.model, sparsity=0.3)
        >>> x = torch.randn(1, 3, 640, 640)
        >>> memory = model_memory_usage_with_reducing(x, model, 'cuda:0')
    """
    torch.cuda.empty_cache()

    reduced_model = copy.deepcopy(pruned_model)

    # DDP 여부 확인
    if isinstance(pruned_model, torch.nn.parallel.DistributedDataParallel):
        pruned_base_model = pruned_model.module
        reduced_base_model = reduced_model.module
    else:
        pruned_base_model = pruned_model.model
        reduced_base_model = reduced_model.model

    reduced_base_model = reduced_base_model.to(device).eval()
    pruned_base_model = pruned_base_model.to(device).eval()

    # Reducing 적용
    yolov8_reducing(pruned_base_model, reduced_base_model)

    reduced_layers = extract_layers(reduced_base_model)
    pruned_layers = extract_layers(pruned_base_model)

    mem_list = []
    layer_outputs = []

    x = x.to(device)

    for i, layer in enumerate(reduced_layers):
        layer = layer.to(device)

        if type(layer).__name__ == "Concat":
            if i == 11:
                concat_inputs = [layer_outputs[6], layer_outputs[10]]
            elif i == 14:
                concat_inputs = [layer_outputs[4], layer_outputs[13]]
            elif i == 17:
                concat_inputs = [layer_outputs[12], layer_outputs[16]]
            elif i == 20:
                concat_inputs = [layer_outputs[9], layer_outputs[19]]

            concat_inputs = [ci.to(device) for ci in concat_inputs]

            x, used_memory = measure_memory(concat_inputs, [layer], device)
            mem_list.append(used_memory)
            layer_outputs.append(x)

            setattr(pruned_layers[i], 'memory_usg', used_memory)
            continue

        if type(layer).__name__ == "Detect":
            inputs = [layer_outputs[15], layer_outputs[18], layer_outputs[21]]
            inputs = [inp.to(device) for inp in inputs]

            x, used_memory = measure_memory(inputs, [layer], device)
            mem_list.append(used_memory)

            setattr(pruned_layers[i], 'memory_usg', used_memory)
            continue

        else:
            x = x.to(device)
            x, used_memory = measure_memory(x, [layer], device)
            mem_list.append(used_memory)
            layer_outputs.append(x)

            setattr(pruned_layers[i], 'memory_usg', used_memory)

    return sum(mem_list)


def custom_memory_loss_function(memory, hyperparam, device_condition_memory):
    """
    메모리 제약 기반 손실 함수를 계산합니다.

    목표 메모리 사용량을 초과할 경우 페널티를 부여합니다.

    Args:
        memory: 현재 메모리 사용량 (MB)
        hyperparam: 손실 가중치
        device_condition_memory: 목표 메모리 제한 (MB)

    Returns:
        loss: 메모리 손실 텐서

    Example:
        >>> memory = model_memory_usage(x, model, device)
        >>> mem_loss = custom_memory_loss_function(memory, 0.1, 100.0)
        >>> total_loss = task_loss + mem_loss
    """
    memory_diff = memory - device_condition_memory
    memory_loss = max(0, memory_diff)

    final_loss = hyperparam * memory_loss

    return torch.tensor(final_loss, requires_grad=True)
