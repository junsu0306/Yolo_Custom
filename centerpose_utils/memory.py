"""
CenterPose Memory Measurement Module

DLA-34 모델의 GPU 메모리 사용량 측정 및 메모리 기반 손실 함수를 제공합니다.

주요 함수:
    - measure_memory: 레이어별 메모리 사용량 측정
    - measure_model_memory: 전체 모델 메모리 측정
    - measure_pruned_layer_memory: Pruning 후 메모리 측정
    - custom_memory_loss_function: 메모리 제약 기반 손실 함수
"""

import copy
import torch
import torch.nn as nn

from .reducing import reduce_pruned_model


def measure_memory(x, layers, device, pre_img=None, pre_hm=None, pre_hm_hp=None):
    """
    레이어 실행 시 GPU 메모리 사용량을 측정합니다.

    Args:
        x: 입력 텐서 (또는 텐서 리스트)
        layers: 측정할 레이어 리스트
        device: GPU 디바이스
        pre_img: 이전 이미지 (tracking용)
        pre_hm: 이전 heatmap (tracking용)
        pre_hm_hp: 이전 keypoint heatmap (tracking용)

    Returns:
        x: 출력 텐서
        memory_diff: 사용된 메모리 (MB)
    """
    if pre_img is not None or pre_hm is not None or pre_hm_hp is not None:
        if pre_img is not None:
            x = pre_img
        elif pre_hm is not None:
            x = pre_hm
        elif pre_hm_hp is not None:
            x = pre_hm_hp

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
    CenterPose 모델에서 레이어 리스트를 추출합니다.

    Args:
        model: CenterPose 모델

    Returns:
        layers_list: (name, layer) 튜플 리스트
    """
    layers_list = []
    for name, layer in model.named_children():
        if isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList):
            for sub_layer in layer:
                layers_list.append((name, sub_layer))
        elif name == "base":
            for sub_name, sub_layer in layer.named_children():
                layers_list.append((f"base.{sub_name}", sub_layer))
        else:
            layers_list.append((name, layer))
    return layers_list


def measure_model_memory(model, input_tensor, device, file_stream=None):
    """
    CenterPose 모델의 전체 메모리 사용량을 측정합니다.

    Args:
        model: CenterPose 모델
        input_tensor: 입력 딕셔너리 {"input": tensor, "pre_img": ..., ...}
        device: GPU 디바이스
        file_stream: 로그 출력 파일 (선택)

    Returns:
        total_memory: 전체 메모리 사용량 (MB)

    Example:
        >>> input_tensor = {"input": torch.randn(1, 3, 512, 512)}
        >>> memory = measure_model_memory(model, input_tensor, 'cuda:0')
        >>> print(f"Memory: {memory:.2f} MB")
    """
    copy_model = copy.deepcopy(model)
    copy_model.eval()

    mem_list = []
    head_to_output_idx = {
        0: ['tracking', 'tracking_hp'],
        1: ['hm', 'wh', 'reg'],
        2: ['hm_hp', 'hp_offset', 'hps', 'hps_uncertainty'],
        3: ['scale', 'scale_uncertainty']
    }
    head_map = {head: idx for idx, heads in head_to_output_idx.items() for head in heads}

    x = input_tensor["input"].to(device)
    pre_img = input_tensor.get("pre_img", None)
    pre_hm = input_tensor.get("pre_hm", None)
    pre_hm_hp = input_tensor.get("pre_hm_hp", None)

    for name, layer in extract_layers(copy_model):
        if name == 'base.pre_img_layer':
            temp, used_memory = measure_memory(x, [layer], device=device, pre_img=pre_img)

        elif name == 'base.pre_hm_layer':
            temp, used_memory = measure_memory(x, [layer], device=device, pre_hm=pre_hm)

        elif name == 'base.pre_hm_hp_layer':
            temp, used_memory = measure_memory(x, [layer], device=device, pre_hm_hp=pre_hm_hp)

        elif name == 'dla_up':
            x, used_memory = measure_memory(x, [layer], device=device)
            y = [x[i].clone() for i in range(len(x))]

        elif name == 'ida_up':
            layer.cpu()
            if isinstance(y, (tuple, list)):
                y = [item.cpu() for item in y]
            else:
                y = x.cpu()

            torch.cuda.empty_cache()
            before_mem = torch.cuda.memory_allocated(device) / 1024**2

            layer.to(device)

            with torch.no_grad():
                layer(y, 0, len(y))

            after_mem = torch.cuda.memory_allocated(device) / 1024**2
            used_memory = after_mem - before_mem

        elif name == "hm":
            # Head 레이어 메모리 측정
            torch.cuda.empty_cache()
            before_mem = torch.cuda.memory_allocated(device) / 1024**2

            feature = y[-1]
            if feature.dim() == 3:
                feature = feature.unsqueeze(0)
            feature = feature.to(device)

            # ConvGRU 메모리 측정 (있는 경우)
            try:
                from lib.models.networks.convGRU import ConvGRU
                input_channels = feature.shape[1]
                convgru = ConvGRU(
                    input_channels=input_channels,
                    hidden_channels=[64],
                    kernel_size=3,
                    step=4,
                    effective_step=[0, 1, 2, 3]
                ).to(device)

                with torch.no_grad():
                    gru_outputs, _ = convgru(feature)
                after_mem = torch.cuda.memory_allocated(device) / 1024**2
                used_memory = after_mem - before_mem
                mem_list.append(used_memory)

                if file_stream is not None:
                    print(f"Layer ({name}) memory usage: {used_memory:.5f} MB", file=file_stream)

                # 각 Head 메모리 측정
                for head in copy_model.heads:
                    idx = head_map.get(head)
                    if idx is not None:
                        head_input = gru_outputs[idx].cpu()
                        head_layer = getattr(copy_model, head).cpu()

                        torch.cuda.empty_cache()
                        before_mem = torch.cuda.memory_allocated(device) / 1024**2

                        head_input = head_input.to(device)
                        head_layer = head_layer.to(device)

                        with torch.no_grad():
                            out = head_layer(head_input)

                        after_mem = torch.cuda.memory_allocated(device) / 1024**2
                        head_mem = after_mem - before_mem
                        if file_stream is not None:
                            print(f"Head ({head}) memory usage: {head_mem:.5f} MB", file=file_stream)
                        mem_list.append(head_mem)
            except ImportError:
                pass

            break

        else:
            x, used_memory = measure_memory(x, [layer], device=device)

        mem_list.append(used_memory)
        if file_stream is not None:
            print(f"Layer ({name}) memory usage: {used_memory:.5f} MB", file=file_stream)

    sum_mem = sum(mem_list)

    if file_stream is not None:
        print(f"Total memory usage across all layers: {sum_mem:.5f} MB", file=file_stream)
        print("=" * 40, file=file_stream)

    return sum_mem


def measure_pruned_layer_memory(model, input_tensor, device, file_stream=None):
    """
    Pruning 후 CenterPose 모델의 메모리 사용량을 측정합니다.

    Pruning된 모델을 복사하여 reducing을 적용한 후 메모리를 측정합니다.

    Args:
        model: Pruning이 적용된 CenterPose 모델
        input_tensor: 입력 딕셔너리
        device: GPU 디바이스
        file_stream: 로그 출력 파일 (선택)

    Returns:
        total_memory: Reducing 후 메모리 사용량 (MB)
    """
    if file_stream is not None:
        print(f"============ Reducing model memory capture start ============", file=file_stream)

    reduced_model = reduce_pruned_model(copy.deepcopy(model)).to(device)
    reduced_model.eval()

    # measure_model_memory와 동일한 로직 수행
    return measure_model_memory(reduced_model, input_tensor, device, file_stream)


def custom_memory_loss_function(total_memory, hyperparam, device_condition_memory):
    """
    메모리 제약 기반 손실 함수를 계산합니다.

    목표 메모리 사용량을 초과할 경우 페널티를 부여합니다.

    Args:
        total_memory: 현재 메모리 사용량 (MB)
        hyperparam: 손실 가중치
        device_condition_memory: 목표 메모리 제한 (MB)

    Returns:
        loss: 메모리 손실 텐서

    Example:
        >>> memory = measure_model_memory(model, input_tensor, device)
        >>> mem_loss = custom_memory_loss_function(memory, 0.1, 50.0)
        >>> total_loss = task_loss + mem_loss
    """
    memory_diff = total_memory - device_condition_memory
    memory_loss = max(0, memory_diff)

    final_loss = hyperparam * memory_loss

    return torch.tensor(final_loss, requires_grad=True)
