import torch
import os
from lib.models.model import create_model, load_model
from lib.opts import opts
from dlasg_reducing import get_survived_filter_idx, conv_reduce, bn_reduce, copy_layer, bn_copy_layer
import time
from lib.datasets.dataset_combined import ObjectPoseDataset
from lib.pruning.dlasg_pruning import dlasg_blockwise_pruning , reduce_pruned_model

# def reduce_dla_model(model):
#     from torch import nn
#     import copy

#     # 모델 복제
#     reduced_model = copy.deepcopy(model)

#     # 기존 모델 레이어 순회
#     for name, layer in model.named_modules():
#         if isinstance(layer, nn.Conv2d):
#             survived_out = get_survived_filter_idx(layer)
#             survived_in = torch.arange(layer.weight.shape[1])

#             reduced_conv = nn.Conv2d(
#                 in_channels=len(survived_in),
#                 out_channels=len(survived_out),
#                 kernel_size=layer.kernel_size,
#                 stride=layer.stride,
#                 padding=layer.padding,
#                 dilation=layer.dilation,
#                 groups=layer.groups,
#                 bias=(layer.bias is not None)
#             )
#             conv_reduce(layer, reduced_conv, survived_out, survived_in)

#             set_module_by_name(reduced_model, name, reduced_conv)

#         elif isinstance(layer, nn.BatchNorm2d):
#             survived_idx = get_survived_filter_idx(layer)

#             reduced_bn = nn.BatchNorm2d(len(survived_idx))
#             bn_reduce(layer, reduced_bn, survived_idx)

#             set_module_by_name(reduced_model, name, reduced_bn)

#     return reduced_model




# def set_module_by_name(model, name, new_module):
#     names = name.split('.')
#     submod = model
#     for n in names[:-1]:
#         submod = getattr(submod, n)
#     setattr(submod, names[-1], new_module)


def load_and_reduce_model(model_path, sparsity):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    opt = opts()
    opt = opt.parser.parse_args()

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


    Dataset = ObjectPoseDataset
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print("Creating model...")


    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

    print(f"Loading model weights from {model_path}...")


    model = load_model(model, model_path)


    model = model.to("cuda:0")

    dlasg_blockwise_pruning(model, sparsity=0.5, device = 'cuda:0')

    total_zero = 0
    total_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        for p in model.parameters():
            # 정확히 0인 요소 개수
            z = (p == 0).sum().item() 
            total_zero += z

    print(f"Total parameters: {total_params}")
    print(f"zero parameters: {total_zero}")

    print(f"Reducing model with pruning threshold...")



    reduced_model = reduce_pruned_model(model)
   
    checkpoint = torch.load(model_path, map_location='cpu')
    epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0

    model_filename = f"reduced_model_{int(sparsity * 100)}_book.pth"

    #torch.save({'epoch': epoch, 'state_dict': reduced_model.state_dict()}, model_filename)
    torch.save(reduced_model, model_filename)


    print(f"Reduced model weights saved as {model_filename} (epoch {epoch})")

    return reduced_model






if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model_path = "../exp/object_pose/objectron_book_dla_34_2025-09-02-05-43/book_last.pth"
    sparsity = 0.50
    reduced_model = load_and_reduce_model(model_path, sparsity)

    total_zero = 0
    total_params = sum(p.numel() for p in reduced_model.parameters())
    with torch.no_grad():
        for p in reduced_model.parameters():
            # 정확히 0인 요소 개수
            z = (p == 0).sum().item()
            total_zero += z

    print(f"After reduece Total parameters: {total_params}")
    print(f"After reduece zero parameters: {total_zero}")
