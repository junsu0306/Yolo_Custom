import torch
import os
from lib.models.model import create_model, load_model
import torch.nn as nn
from lib.opts import opts
from dlasg_pruning import dlasg_blockwise_pruning, get_basicblock_conv_pairs  # 프루닝 함수가 들어있는 모듈
from dlasg_reducing import *
import time
import copy



def dlasg_blockwise_reducing(model, reduced_model):
    

    org_model_module_list = dict(model.named_children())["base"]
    reduced_module_list =  dict(reduced_model.named_children())["base"]
   
    

    for name, module in org_model_module_list.named_modules():
        if module.__class__.__name__ == 'BasicBlock':
            reduce_module = reduced_lookup[name]    
            org_conv_list = []
            org_bn_list = []
            
            org_conv_list.append(module.conv1)
            org_conv_list.append(module.conv2)
            org_bn_list.append(module.bn1)
            org_bn_list.append(module.bn2) 

            prev_name, prev_module = get_prev_module(org_leaf_modules, name+".conv1")
            
            survived_idx_in = get_survived_filter_idx(module.conv1)
            survived_idx_out = get_survived_filter_idx(module.conv2)
            print(survived_idx_in.shape)
            print(survived_idx_out.shape)
              
        
           
            conv_reduce(
                layer= conv,
                reduced_layer= getattr(reduce_module,conv_layer),
                survived_out_channels_idx=survived_idx,
                survived_in_channels_idx=prev_survived_idx,
            )
            bn_reduce(module.bn1, getattr(reduce_module,bn_layer), survived_idx)
            prev_survived_idx = survived_idx

 
      






def load_and_prune_model(model_path, sparsity):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    # 옵션 설정
    opt = opts()
    opt = opt.parser.parse_args([])  # 빈 리스트 전달로 argparse 오류 방지

    # 필요한 속성 초기화
    opt.arch = 'dla_34'  # 기본 모델 구조 설정
    opt.heads = {'hm': 1, 'wh': 2, 'hps': 16, 'hps_uncertainty': 16, 'reg': 2, 'hm_hp': 8, 'hp_offset': 2, 'scale': 3, 'scale_uncertainty': 3, 'tracking': 2, 'tracking_hp': 16}  # heads 기본 설정
    opt.head_conv = 256
    opt.tracking_task = True
    opt.pre_hm= True
    opt.pre_hm_hp= True
    opt.pre_img= True
        
    # 모델 생성
    print("Creating model...")
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

    # 모델 로드
    print(f"Loading model weights from {model_path}...")
    model = load_model(model, model_path)

    # 프루닝 적용
    print(f"Applying pruning with sparsity {sparsity}...")
    pruned_model= dlasg_blockwise_pruning(model, sparsity)

    # Load original model checkpoint to retrieve epoch info
    checkpoint = torch.load(model_path, map_location='cpu')
    epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0

    # Save pruned model state_dict with epoch metadata
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    model_filename = f"pruned_model_{int(sparsity * 100)}_{time_str}.pth"
    torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, model_filename)
    print(f"Pruned model weights saved as {model_filename} (epoch {epoch})")

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model_path = "/workspace/MH/CenterPose/exp/object_pose/objectron_shoe_dla_34_2025-04-22-14-08/shoe_last.pth"  
    sparsity = 0.5
    model = load_and_prune_model(model_path, sparsity)
    reduced_model = copy.deepcopy(model)
    dlasg_blockwise_reducing(model,reduced_model)
    print(f"original model: {count_parameters(model)}")
    print(f"reduced model: {count_parameters(reduced_model)}")
    print(f"reduce rate: {(count_parameters(model) - count_parameters(reduced_model))/count_parameters(model) *100} % ")
    
    
    


