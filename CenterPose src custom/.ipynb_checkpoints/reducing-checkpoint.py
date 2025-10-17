import torch
import os
from lib.models.model import create_model, load_model
from lib.opts import opts
from dlasg_reducing import get_survived_filter_idx, conv_reduce, bn_reduce, copy_layer, bn_copy_layer

def reduce_dla_model(model):
    from torch import nn
    import copy

    # 모델 복제
    reduced_model = copy.deepcopy(model)

    # 기존 모델 레이어 순회
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            survived_out = get_survived_filter_idx(layer)
            survived_in = torch.arange(layer.weight.shape[1])

            reduced_conv = nn.Conv2d(
                in_channels=len(survived_in),
                out_channels=len(survived_out),
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=(layer.bias is not None)
            )
            conv_reduce(layer, reduced_conv, survived_out, survived_in)

            set_module_by_name(reduced_model, name, reduced_conv)

        elif isinstance(layer, nn.BatchNorm2d):
            survived_idx = get_survived_filter_idx(layer)

            reduced_bn = nn.BatchNorm2d(len(survived_idx))
            bn_reduce(layer, reduced_bn, survived_idx)

            set_module_by_name(reduced_model, name, reduced_bn)

    return reduced_model

def set_module_by_name(model, name, new_module):
    names = name.split('.')
    submod = model
    for n in names[:-1]:
        submod = getattr(submod, n)
    setattr(submod, names[-1], new_module)


def load_and_reduce_model(model_path, sparsity):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    opt = opts()
    opt = opt.parser.parse_args([])
    opt.arch = 'dla_34'
    opt.heads = {'hm': 1, 'reg': 2, 'wh': 2}
    opt.head_conv = 256
    opt.tracking_task = False

    print("Creating model...")
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

    print(f"Loading model weights from {model_path}...")
    model = load_model(model, model_path)

    print(f"Reducing model with pruning threshold...")
    reduced_model = reduce_dla_model(model)

    checkpoint = torch.load(model_path, map_location='cpu')
    epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0

    model_filename = f"reduced_model_{int(sparsity * 100)}.pth"
    torch.save({'epoch': epoch, 'state_dict': reduced_model.state_dict()}, model_filename)
    print(f"Reduced model weights saved as {model_filename} (epoch {epoch})")

    return reduced_model

if __name__ == "__main__":
    model_path = "/workspace/MH/CenterPose/src/my_pruned_model_50.pth"
    sparsity = 0.50
    reduced_model = load_and_reduce_model(model_path, sparsity)
