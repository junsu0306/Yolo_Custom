import torch
import os
from lib.models.model import create_model, load_model
from lib.opts import opts

def check_zeroed_filters_in_model(model):
    """
    전체 Conv2d 레이어의 weight를 검사하여
    완전히 0으로 된 필터 개수 및 pruning 비율 출력
    """
    total_filters = 0
    total_zeroed = 0

    print("🔍 Zeroed filter summary in all Conv2d layers:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weight = module.weight.data
            num_filters = weight.shape[0]
            num_zeroed = (weight.view(num_filters, -1).abs().sum(dim=1) == 0).sum().item()

            print(f"[{name}] Zeroed filters: {num_zeroed} / {num_filters}")

            total_filters += num_filters
            total_zeroed += num_zeroed

    pruning_ratio = total_zeroed / total_filters if total_filters > 0 else 0
    print(f"\n📊 Total filters: {total_filters}")
    print(f"❌ Total zeroed filters: {total_zeroed}")
    print(f"📉 Pruning ratio: {pruning_ratio * 100:.2f}%")

def load_and_prune_model(model_path):
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

    print("\n🔍 Zeroed filter summary in BasicBlocks:")
    check_zeroed_filters_in_model(model)

    return model

if __name__ == "__main__":
    model_path = '/workspace/MH/CenterPose/src/pruned_model_75.pth'
    load_and_prune_model(model_path)