import torch
import os
import time
import torch.nn as nn
from lib.models.model import create_model, load_model
from lib.opts import opts
from lib.datasets.dataset_combined import ObjectPoseDataset
from lib.models.networks.convGRU import ConvGRU

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

if __name__ == '__main__':
    opt = opts()
    opt = opt.parser.parse_args()

    opt.c = 'shoe'
    opt.arch = 'dlav1_34'
    opt.obj_scale = True
    opt.obj_scale_weight = 1
    opt.mug = False
    opt.exp_id = f'objectron_{opt.c}_{opt.arch}'
    opt.num_epochs = 140
    opt.val_intervals = 5
    opt.lr_step = '90,120'
    opt.batch_size = 16
    opt.lr = 6e-5
    opt.gpus = '0'
    opt.num_workers = 4
    opt.print_iter = 5
    opt.debug = 5
    opt.save_all = True
    opt.head_conv = 256
    opt.tracking_task = True
    opt.heads = {'hm': 1, 'tracking': 2}

    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.test_scales = [1.0]
    opt.fix_res = True
    opt.reg_offset = True
    opt.reg_bbox = True
    opt.hm_hp = True
    opt.reg_hp_offset = True
    opt.pad = 31
    opt.num_stacks = 1
    opt.master_batch_size = opt.batch_size
    opt.chunk_sizes = [opt.master_batch_size]
    opt.seed = 317
    opt.not_cuda_benchmark = False
    opt.test = False
    opt.task = 'objectron'
    opt.device = 'cuda:0'

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    opt.save_dir = os.path.join(opt.exp_dir, f'{opt.exp_id}_{time_str}')
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')

    Dataset = ObjectPoseDataset
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    model_path = "/workspace/MH/CenterPose/src/reduced_model_50.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    print("Creating model...")
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt).to(opt.device)
    print(f"Loading model weights from {model_path}...")
    model = load_model(model, model_path)
    model.eval()

    torch.cuda.empty_cache()

    x = torch.randn(1, 3, 512, 512).to(opt.device)
    mem_list = []

    for name, layer in extract_layers(model):
        if name == 'dla_up':
            x, used_memory = measure_memory(x, [layer], device=opt.device)
            y = [x[i].clone() for i in range(len(x))]  # 안전하게



        elif name == 'ida_up':
            layer.to(opt.device)
            torch.cuda.empty_cache()
            before_mem = torch.cuda.memory_allocated(opt.device) / 1024**2
            with torch.no_grad():
                layer(y, 0, len(y))
            after_mem = torch.cuda.memory_allocated(opt.device) / 1024**2
            used_memory = after_mem - before_mem
            
        elif name == 'convGRU':
            # manual ConvGRU instantiation and usage
            feature = y[-1]
            if feature.dim() == 3:
                feature = feature.unsqueeze(0)

            feature = feature.to(opt.device) 

            input_channels = feature.shape[1]
            convgru = ConvGRU(
                input_channels= input_channels,
                hidden_channels=[64],
                kernel_size=3,
                step=4,
                effective_step=[0, 1, 2, 3]
            ).to(opt.device)

            torch.cuda.empty_cache()
            before_mem = torch.cuda.memory_allocated(opt.device) / 1024**2
            with torch.no_grad():
                gru_outputs, _ = convgru(feature)
            after_mem = torch.cuda.memory_allocated(opt.device) / 1024**2
            used_memory = after_mem - before_mem
            
        elif name in model.heads:
            head_to_output_idx = {
                0: ['tracking', 'tracking_hp'],
                1: ['hm', 'wh', 'reg'],
                2: ['hm_hp', 'hp_offset', 'hps', 'hps_uncertainty'],
                3: ['scale', 'scale_uncertainty']
            }

            # 역방향 매핑: head → output index
            head_map = {head: idx for idx, heads in head_to_output_idx.items() for head in heads}

            for head in model.heads:
                idx = head_map.get(head)
                if idx is not None:
                    head_input = gru_outputs[idx].to(opt.device)
                    head_layer = getattr(model, head).to(opt.device)

                    # 메모리 측정
                    torch.cuda.empty_cache()
                    before_mem = torch.cuda.memory_allocated(opt.device) / 1024**2
                    with torch.no_grad():
                        out = head_layer(head_input)
                    after_mem = torch.cuda.memory_allocated(opt.device) / 1024**2
                    used_memory = after_mem - before_mem

        else:
            x, used_memory = measure_memory(x, [layer], device=opt.device)

        mem_list.append(used_memory)
        print(f"Layer ({name}) memory usage: {used_memory:.5f} MB")

    print(f"Total memory usage across all layers: {sum(mem_list):.5f} MB")
