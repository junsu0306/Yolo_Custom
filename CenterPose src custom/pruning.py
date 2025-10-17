import torch
import os
from lib.models.model import create_model, load_model
from lib.opts import opts
from dlasg_pruning import dlasg_blockwise_pruning  # 프루닝 함수가 들어있는 모듈

def load_and_prune_model(model_path, sparsity):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    # 옵션 설정
    opt = opts()
    opt = opt.parser.parse_args([])  # 빈 리스트 전달로 argparse 오류 방지

    # 필요한 속성 초기화
    opt.arch = 'dla_34'  # 기본 모델 구조 설정
    opt.heads = {'hm': 1, 'reg': 2, 'wh': 2}  # heads 기본 설정
    opt.head_conv = 256
    opt.tracking_task = False
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
    model_filename = f"my_pruned_model_{int(sparsity * 100)}.pth"
    torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, model_filename)
    print(f"Pruned model weights saved as {model_filename} (epoch {epoch})")


    return model

if __name__ == "__main__":
    model_path = "/workspace/MH/CenterPose/exp/object_pose/objectron_shoe_dla_34_2025-04-22-14-08/shoe_15.pth"
    sparsity = 0.50
    model = load_and_prune_model(model_path, sparsity)

