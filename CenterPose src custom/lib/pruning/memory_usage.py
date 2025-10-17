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
import sys


def measure_memory(x, layers, device, pre_img=None, pre_hm=None, pre_hm_hp=None):
    
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
            #x += temp

        elif name == 'base.pre_hm_layer':
            temp, used_memory = measure_memory(x, [layer], device=device, pre_hm=pre_hm)
            #x += temp
        elif name == 'base.pre_hm_hp_layer':
            temp, used_memory = measure_memory(x, [layer], device=device, pre_hm_hp=pre_hm_hp)
            #x += temp

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

        elif name == "hm": # This is a first head layer
            torch.cuda.empty_cache()
            before_mem = torch.cuda.memory_allocated(device) / 1024**2

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

 
            with torch.no_grad():
                gru_outputs, _ = convgru(feature)
            after_mem = torch.cuda.memory_allocated(device) / 1024**2
            used_memory = after_mem - before_mem
            mem_list.append(used_memory)
            if file_stream is not None:
                print(f"Layer ({name}) memory usage: {used_memory:.5f} MB", file =  file_stream) 

            # measure memory for heads
            for head in copy_model.heads:
               #print(f"Head name: {head}")# for debug
                idx = head_map.get(head)
                if idx is not None:

                    head_input = gru_outputs[idx].cpu()
                    head_layer = getattr(copy_model, head).cpu()


                    torch.cuda.empty_cache()
                    before_mem = torch.cuda.memory_allocated(device)/ 1024**2

                    head_input = head_input.to(device)
                    head_layer = head_layer.to(device)


                    with torch.no_grad():
                        out = head_layer(head_input)
          
                    after_mem = torch.cuda.memory_allocated(device)/ 1024**2
                    head_mem = (after_mem - before_mem) 
                    if file_stream is not None:
                        print(f"Head ({head}) memory usage: {head_mem:.5f} MB", file=file_stream) # for debug
                    mem_list.append(head_mem)
 

            
            break
            


        else:
            x, used_memory = measure_memory(x, [layer], device=device)

       
        mem_list.append(used_memory)
        if file_stream is not None:
            print(f"Layer ({name}) memory usage: {used_memory:.5f} MB",file = file_stream)

    sum_mem = sum(mem_list)

    if file_stream is not None:
        print(f"Total memory usage across all layers: {sum_mem:.5f} MB", file = file_stream) # for debug
        print("="*40, file = file_stream)
    return sum_mem

def measure_pruned_layer_memory(model, input_tensor, device, file_stream = None): # this model is pruned model
    if file_stream is not None:
        print(f"============ Reduing model memeory capture start ============", file= file_stream)
    
    reduced_model = reduce_pruned_model(copy.deepcopy(model)).to(device)
    reduced_model.eval()


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
 
    for name, layer in extract_layers(reduced_model):

        if name == 'base.pre_img_layer': 
            temp, used_memory = measure_memory(x, [layer], device=device, pre_img=pre_img)
            #x += temp

        elif name == 'base.pre_hm_layer':
            temp, used_memory = measure_memory(x, [layer], device=device, pre_hm=pre_hm)
            #x += temp
        elif name == 'base.pre_hm_hp_layer':
            temp, used_memory = measure_memory(x, [layer], device=device, pre_hm_hp=pre_hm_hp)
            #x += temp

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

        elif name == "hm": # This is a first head layer
            torch.cuda.empty_cache()
            before_mem = torch.cuda.memory_allocated(device) / 1024**2

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

 
            with torch.no_grad():
                gru_outputs, _ = convgru(feature)
            after_mem = torch.cuda.memory_allocated(device) / 1024**2
            used_memory = after_mem - before_mem
            mem_list.append(used_memory)
            if file_stream is not None:
                print(f"Layer ({name}) memory usage: {used_memory:.5f} MB", file =  file_stream) 

            # measure memory for heads
            for head in reduced_model.heads:
               #print(f"Head name: {head}")# for debug
                idx = head_map.get(head)
                if idx is not None:

                    head_input = gru_outputs[idx].cpu()
                    head_layer = getattr(reduced_model, head).cpu()


                    torch.cuda.empty_cache()
                    before_mem = torch.cuda.memory_allocated(device)/ 1024**2

                    head_input = head_input.to(device)
                    head_layer = head_layer.to(device)


                    with torch.no_grad():
                        out = head_layer(head_input)
          
                    after_mem = torch.cuda.memory_allocated(device)/ 1024**2
                    head_mem = (after_mem - before_mem) 
                    if file_stream is not None:
                        print(f"Head ({head}) memory usage: {head_mem:.5f} MB", file=file_stream) # for debug
                    mem_list.append(head_mem)
            break
            


        else:
            x, used_memory = measure_memory(x, [layer], device=device)

       
        mem_list.append(used_memory)
        if file_stream is not None:
            print(f"Layer ({name}) memory usage: {used_memory:.5f} MB",file = file_stream)

    sum_mem = sum(mem_list)

    if file_stream is not None:
        print(f"Total memory usage across all layers: {sum_mem:.5f} MB", file = file_stream) # for debug
        print("="*40, file = file_stream)
    return sum_mem

def custom_memory_loss_function(total_memory, hyperparam, device_condition_memory):
    memory_diff = total_memory - device_condition_memory
    memory_loss = max(0, memory_diff)

    final_loss = hyperparam * memory_loss
    
    return torch.tensor(final_loss, requires_grad=True)