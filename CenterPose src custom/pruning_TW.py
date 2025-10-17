import torch
import os
from lib.models.model import create_model
import torch.nn as nn
from lib.opts import opts
import time
import copy
from lib.models.networks.convGRU import ConvGRU
from lib.pruning.dlasg_pruning import dlasg_blockwise_pruning, reduce_pruned_model

from lib.datasets.dataset_combined import ObjectPoseDataset

def measure_memory(x, layers, device, pre_img=None, pre_hm=None, pre_hm_hp=None):
    
    if isinstance(x, (tuple, list)):
        x = [item.cpu() for item in x]
    else:
        x = x.cpu()

    for layer in layers:
        layer.cpu()
    torch.cuda.empty_cache()

    before_memory = torch.cuda.memory_allocated(device) / 1024**2
    
    if pre_img is not None or pre_hm is not None or pre_hm_hp is not None:
        if pre_img is not None:
            x = pre_img
        elif pre_hm is not None:
            x = pre_hm
        elif pre_hm_hp is not None:
            x = pre_hm_hp

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

    x = input_tensor["input"].to(device)
    pre_img = input_tensor.get("pre_img", None)
    pre_hm = input_tensor.get("pre_hm", None)
    pre_hm_hp = input_tensor.get("pre_hm_hp", None)
 
    for name, layer in extract_layers(model):

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
            print(f"Layer ({name}) memory usage: {used_memory:.5f} MB")

            # measure memory for heads
            for head in model.heads:
               #print(f"Head name: {head}")# for debug
                idx = head_map.get(head)
                if idx is not None:

                    head_input = gru_outputs[idx].cpu()
                    head_layer = getattr(model, head).cpu()


                    torch.cuda.empty_cache()
                    before_mem = torch.cuda.memory_allocated(device)/ 1024**2

                    head_input = head_input.to(device)
                    head_layer = head_layer.to(device)


                    with torch.no_grad():
                        out = head_layer(head_input)
          
                    after_mem = torch.cuda.memory_allocated(device)/ 1024**2
                    head_mem = (after_mem - before_mem) 

                    print(f"Head ({head}) memory usage: {head_mem:.5f} MB") # for debug
                    mem_list.append(head_mem)
 

            mem_list.append(used_memory)
            
            break
            


        else:
            x, used_memory = measure_memory(x, [layer], device=device)

       
        mem_list.append(used_memory)
        print(f"Layer ({name}) memory usage: {used_memory:.5f} MB")

    sum_mem = sum(mem_list)

    print(f"Total memory usage across all layers: {sum_mem:.5f} MB") # for debug

    return sum_mem




def load_model():
  

    # 옵션 설정
    opt = opts()
    opt = opt.parser.parse_args([])  # 빈 리스트 전달로 argparse 오류 방지

    # Local configuration
    opt.c = 'shoe'
    opt.arch='dla_34'
    opt.obj_scale = True
    opt.obj_scale_weight = 1
    opt.mug = False

    # Training param
    opt.tracking_task = True
    opt.exp_id = f'objectron_{opt.c}_{opt.arch}'
    opt.num_epochs = 15
    opt.val_intervals = 1
    opt.lr_step = '6,10'
    opt.batch_size = 16
    opt.lr = 1.25e-4
    opt.gpus = '0'
    opt.num_workers = 4
    opt.print_iter = 5
    opt.debug = 5
    opt.save_all = True

    # Tracking related
    if opt.tracking_task == True:

        if opt.c == 'chair' or opt.c == 'bike':
            opt.rotate = 15  # degree
        else:
            opt.rotate = 60

        opt.obj_scale_uncertainty = True
        opt.hps_uncertainty = True
        opt.tracking_label_mode = 1
        opt.render_hm_mode = 1
        opt.render_hmhp_mode = 2
        opt.KL_scale_uncertainty = 0.1
        opt.KL_kps_uncertainty = 0.1

        opt.pre_img = True
        opt.pre_hm = True
        opt.tracking = True
        opt.pre_hm_hp = True
        opt.tracking_hp = True

        opt.shift = 0.05
        opt.scale = 0.05

        # For hm
        opt.hm_heat_random = True
        opt.hm_disturb = 0.05
        opt.lost_disturb = 0.2
        opt.fp_disturb = 0.1

        # For hm_hp
        opt.hm_hp_heat_random = True
        opt.hm_hp_disturb = 0.03
        opt.hp_lost_disturb = 0.1
        opt.hp_fp_disturb = 0.05

        opt.max_frame_dist = 3

        # Currently, CenterPose mode does not support symmetrical objects
        if opt.c in ['bottle', 'chair', 'cup']:
            opt.data_generation_mode_ratio = 0
        else:
            opt.data_generation_mode_ratio = 0.3

        print('Running tracking')

        opt.vis_thresh = max(opt.track_thresh, opt.vis_thresh)
        opt.pre_thresh = max(opt.track_thresh, opt.pre_thresh)
        opt.new_thresh = max(opt.track_thresh, opt.new_thresh)
        print('Using tracking threshold for out threshold!', opt.track_thresh)

    # # To continue
    # opt.resume = True
    # opt.load_model = ""

    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
    opt.reg_offset = not opt.not_reg_offset
    opt.reg_bbox = not opt.not_reg_bbox
    opt.hm_hp = not opt.not_hm_hp
    opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

    if opt.head_conv == -1:  # init default head_conv
        opt.head_conv = 256 if 'dla' in opt.arch else 64
    opt.pad = 127 if 'hourglass' in opt.arch else 31
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    if opt.trainval:
        opt.val_intervals = 100000000

    if opt.master_batch_size == -1:
        opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
        slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
        if i < rest_batch_size % (len(opt.gpus) - 1):
            slave_chunk_size += 1
        opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    opt.save_dir = os.path.join(opt.exp_dir, f'{opt.exp_id}_{time_str}')
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)

    Dataset = ObjectPoseDataset
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

   
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)


    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sparsity = 0.5
    model = load_model()
    reduced_model = copy.deepcopy(model)

    dlasg_blockwise_pruning(reduced_model, sparsity,device)
    reduced_model = reduce_pruned_model(reduced_model).to(device)
    reduced_model.eval()

    input_tensor= {
                'input': torch.randn(1, 3, 512, 512, device=device),
                'pre_img':torch.randn(1, 3, 512, 512, device=device),
                'pre_hm': torch.randn(1, 1, 512, 512, device=device),
                'pre_hm_hp': torch.randn(1, 8, 512, 512, device=device)
            }
    model = model.to(device)

    print(f"Reduced model parameters: {count_parameters(reduced_model)}")
    measure_model_memory(model = reduced_model,input_tensor = input_tensor, device=device)
    print("="*30)

    print(f"Original model parameters: {count_parameters(model)}")
    measure_model_memory(model = model,input_tensor = input_tensor, device=device)
    print("="*30)


    


    







    
    


