"""
CenterPose Pruning Script

DLA-34 모델에 L2-norm 기반 Structured Pruning을 적용합니다.
출력: 0으로 마스킹된 필터를 가진 pruned 모델 (.pth)
"""

import torch
import os
from lib.models.model import create_model, load_model
from lib.opts import opts
from dlasg_pruning import dlasg_blockwise_pruning


def load_and_prune_model(model_path, sparsity):
    """
    모델을 로드하고 pruning을 적용합니다.

    Args:
        model_path: 사전 학습된 모델 경로
        sparsity: 제거할 필터 비율 (0.0 ~ 1.0)

    Returns:
        pruned_model: Pruning이 적용된 모델
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    # 옵션 설정
    opt = opts()
    opt = opt.parser.parse_args([])

    # 모델 설정
    opt.arch = 'dla_34'
    opt.heads = {'hm': 1, 'reg': 2, 'wh': 2}
    opt.head_conv = 256
    opt.tracking_task = False

    # 모델 생성 및 로드
    print("Creating model...")
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

    print(f"Loading model weights from {model_path}...")
    model = load_model(model, model_path)

    # Pruning 적용
    print(f"Applying pruning with sparsity {sparsity}...")
    pruned_model = dlasg_blockwise_pruning(model, sparsity)

    # 체크포인트에서 epoch 정보 추출
    checkpoint = torch.load(model_path, map_location='cpu')
    epoch = checkpoint.get('epoch', 0)

    # Pruned 모델 저장
    model_filename = f"my_pruned_model_{int(sparsity * 100)}.pth"
    torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, model_filename)
    print(f"Pruned model saved as {model_filename} (epoch {epoch})")

    return model


if __name__ == "__main__":
    # ========== 설정 ==========
    # 본인의 모델 경로로 수정하세요
    model_path = "/workspace/MH/CenterPose/exp/object_pose/objectron_shoe_dla_34_2025-04-22-14-08/shoe_15.pth"
    sparsity = 0.50  # 50% 필터 제거
    # ==========================

    model = load_and_prune_model(model_path, sparsity)

