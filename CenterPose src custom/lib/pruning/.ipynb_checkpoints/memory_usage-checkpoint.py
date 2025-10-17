import torch
import os
import time
import torch.nn as nn
from lib.models.model import create_model, load_model
from lib.opts import opts
from lib.datasets.dataset_combined import ObjectPoseDataset
from lib.models.networks.convGRU import ConvGRU
from lib.pruning.dlasg_pruning import dlasg_blockwise_pruning, reduce_pruned_model
import copy

def measure_memory(x, layers, device):
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

def measure_model_memory(model, input_tensor, device):
    mem_list = []
    head_to_output_idx = {
        0: ['tracking', 'tracking_hp'],
        1: ['hm', 'wh', 'reg'],
        2: ['hm_hp', 'hp_offset', 'hps', 'hps_uncertainty'],
        3: ['scale', 'scale_uncertainty']
    }
    head_map = {head: idx for idx, heads in head_to_output_idx.items() for head in heads}

    x = input_tensor.to(device)
    for name, layer in extract_layers(model):
        if name == 'dla_up':
            x, used_memory = measure_memory(x, [layer], device=device)
            y = [x[i].clone() for i in range(len(x))]

        elif name == 'ida_up':
            layer.to(device)
            torch.cuda.empty_cache()
            before_mem = torch.cuda.memory_allocated(device) / 1024**2
            with torch.no_grad():
                layer(y, 0, len(y))
            after_mem = torch.cuda.memory_allocated(device) / 1024**2
            used_memory = after_mem - before_mem

        elif name == 'convGRU':
            feature = y[-1]
            if feature.dim() == 3:
                feature = feature.unsqueeze(0)
            feature = feature.to(device)

            input_channels = feature.shape[1]
            convgru = ConvGRU(
                input_channels=input_channels,
                hidden_channels=[64],
                kernel_size=3,
                step=4,
                effective_step=[0, 1, 2, 3]
            ).to(device)

            torch.cuda.empty_cache()
            before_mem = torch.cuda.memory_allocated(device) / 1024**2
            with torch.no_grad():
                gru_outputs, _ = convgru(feature)
            after_mem = torch.cuda.memory_allocated(device) / 1024**2
            used_memory = after_mem - before_mem

            # measure memory for heads
            for head in model.heads:
                idx = head_map.get(head)
                if idx is not None:
                    head_input = gru_outputs[idx].to(device)
                    head_layer = getattr(model, head).to(device)

                    torch.cuda.empty_cache()
                    before_mem = torch.cuda.memory_allocated(device) / 1024**2
                    with torch.no_grad():
                        out = head_layer(head_input)
                    after_mem = torch.cuda.memory_allocated(device) / 1024**2
                    head_mem = after_mem - before_mem
                    mem_list.append(head_mem)

        else:
            x, used_memory = measure_memory(x, [layer], device=device)

        mem_list.append(used_memory)
        #print(f"Layer ({name}) memory usage: {used_memory:.5f} MB")

    sum_mem = sum(mem_list)

    print(f"Total memory usage across all layers: {sum_mem:.5f} MB")

    return sum_mem

def measure_pruned_layer_memory(pruned_model, input_tensor, device):
    reduced_model = reduce_pruned_model(copy.deepcopy(pruned_model)).to(device)
    reduced_model.eval()

    mem_list = []
    head_to_output_idx = {
        0: ['tracking', 'tracking_hp'],
        1: ['hm', 'wh', 'reg'],
        2: ['hm_hp', 'hp_offset', 'hps', 'hps_uncertainty'],
        3: ['scale', 'scale_uncertainty']
    }
    head_map = {head: idx for idx, heads in head_to_output_idx.items() for head in heads}

    x = input_tensor.to(device)
    for name, layer in extract_layers(reduced_model):
        if name == 'dla_up':
            x, used_memory = measure_memory(x, [layer], device=device)
            y = [x[i].clone() for i in range(len(x))]

        elif name == 'ida_up':
            layer.to(device)
            torch.cuda.empty_cache()
            before_mem = torch.cuda.memory_allocated(device) / 1024**2
            with torch.no_grad():
                layer(y, 0, len(y))
            after_mem = torch.cuda.memory_allocated(device) / 1024**2
            used_memory = after_mem - before_mem

        elif name == 'convGRU':
            feature = y[-1]
            if feature.dim() == 3:
                feature = feature.unsqueeze(0)
            feature = feature.to(device)

            input_channels = feature.shape[1]
            convgru = ConvGRU(
                input_channels=input_channels,
                hidden_channels=[64],
                kernel_size=3,
                step=4,
                effective_step=[0, 1, 2, 3]
            ).to(device)

            torch.cuda.empty_cache()
            before_mem = torch.cuda.memory_allocated(device) / 1024**2
            with torch.no_grad():
                gru_outputs, _ = convgru(feature)
            after_mem = torch.cuda.memory_allocated(device) / 1024**2
            used_memory = after_mem - before_mem

            for head in reduced_model.heads:
                idx = head_map.get(head)
                if idx is not None:
                    head_input = gru_outputs[idx].to(device)
                    head_layer = getattr(reduced_model, head).to(device)

                    torch.cuda.empty_cache()
                    before_mem = torch.cuda.memory_allocated(device) / 1024**2
                    with torch.no_grad():
                        out = head_layer(head_input)
                    after_mem = torch.cuda.memory_allocated(device) / 1024**2
                    head_mem = after_mem - before_mem
                    mem_list.append(head_mem)

        else:
            x, used_memory = measure_memory(x, [layer], device=device)

        mem_list.append(used_memory)

    sum_mem = sum(mem_list)
    print(f"Total memory usage of reduced pruned model: {sum_mem:.5f} MB")
    
    return sum_mem


def custom_memory_loss_function(total_memory, hyperparam, device_condition_memory):
    memory_diff = total_memory - device_condition_memory
    memory_loss = max(0, memory_diff)

    final_loss = hyperparam * memory_loss
    
    return torch.tensor(final_loss, requires_grad=True)