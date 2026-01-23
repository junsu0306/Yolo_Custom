"""
CenterPose Reducing Script

Pruning된 모델에서 0으로 마스킹된 필터를 물리적으로 제거하여
실제 모델 크기를 줄입니다.

출력: 물리적으로 축소된 모델 (.pth)
"""

import torch
import os
import time
from lib.models.model import create_model, load_model
from lib.opts import opts
from lib.datasets.dataset_combined import ObjectPoseDataset
from lib.pruning.dlasg_pruning import dlasg_blockwise_pruning, reduce_pruned_model


def load_and_reduce_model(model_path, sparsity):
    """
    모델을 로드하고 pruning + reducing을 적용합니다.

    Args:
        model_path: 사전 학습된 모델 경로
        sparsity: 제거할 필터 비율 (0.0 ~ 1.0)

    Returns:
        reduced_model: 물리적으로 축소된 모델
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")

    opt = opts()
    opt = opt.parser.parse_args()

    # ========== 모델 설정 (필요시 수정) ==========
    opt.c = 'shoe'  # 카테고리: shoe, chair, cup, camera, bike, book, bottle, etc.
    opt.arch = 'dla_34'
    opt.obj_scale = True
    opt.obj_scale_weight = 1
    opt.mug = False
    # ============================================

    # 학습 파라미터 설정
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

    # Tracking 관련 설정
    if opt.tracking_task:
        if opt.c in ['chair', 'bike']:
            opt.rotate = 15
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

        # Heatmap 설정
        opt.hm_heat_random = True
        opt.hm_disturb = 0.05
        opt.lost_disturb = 0.2
        opt.fp_disturb = 0.1

        opt.hm_hp_heat_random = True
        opt.hm_hp_disturb = 0.03
        opt.hp_lost_disturb = 0.1
        opt.hp_fp_disturb = 0.05

        opt.max_frame_dist = 3

        # 대칭 객체 처리
        if opt.c in ['bottle', 'chair', 'cup']:
            opt.data_generation_mode_ratio = 0
        else:
            opt.data_generation_mode_ratio = 0.3

        print('Running tracking')
        opt.vis_thresh = max(opt.track_thresh, opt.vis_thresh)
        opt.pre_thresh = max(opt.track_thresh, opt.pre_thresh)
        opt.new_thresh = max(opt.track_thresh, opt.new_thresh)
        print(f'Using tracking threshold: {opt.track_thresh}')

    # 옵션 파싱
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

    if opt.head_conv == -1:
        opt.head_conv = 256 if 'dla' in opt.arch else 64
    opt.pad = 127 if 'hourglass' in opt.arch else 31
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    if opt.trainval:
        opt.val_intervals = 100000000

    if opt.master_batch_size == -1:
        opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = opt.batch_size - opt.master_batch_size
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
        slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
        if i < rest_batch_size % (len(opt.gpus) - 1):
            slave_chunk_size += 1
        opt.chunk_sizes.append(slave_chunk_size)
    print(f'Training chunk_sizes: {opt.chunk_sizes}')

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    opt.save_dir = os.path.join(opt.exp_dir, f'{opt.exp_id}_{time_str}')
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')

    Dataset = ObjectPoseDataset
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    # 모델 생성 및 로드
    print("Creating model...")
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

    print(f"Loading model weights from {model_path}...")
    model = load_model(model, model_path)
    model = model.to("cuda:0")

    # Pruning 적용
    print(f"Applying pruning with sparsity {sparsity}...")
    dlasg_blockwise_pruning(model, sparsity=sparsity, device='cuda:0')

    # Pruning 결과 확인
    total_params = sum(p.numel() for p in model.parameters())
    total_zero = 0
    with torch.no_grad():
        for p in model.parameters():
            total_zero += (p == 0).sum().item()

    print(f"Total parameters: {total_params:,}")
    print(f"Zero parameters: {total_zero:,}")
    print(f"Sparsity: {total_zero / total_params * 100:.1f}%")

    # Reducing 적용
    print("Reducing model (removing zero filters)...")
    reduced_model = reduce_pruned_model(model)

    # 체크포인트에서 epoch 정보 추출
    checkpoint = torch.load(model_path, map_location='cpu')
    epoch = checkpoint.get('epoch', 0)

    # 축소된 모델 저장
    model_filename = f"reduced_model_{int(sparsity * 100)}_{opt.c}.pth"
    torch.save(reduced_model, model_filename)
    print(f"Reduced model saved as {model_filename} (epoch {epoch})")

    return reduced_model


if __name__ == "__main__":
    # ========== 설정 ==========
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # 본인의 모델 경로로 수정하세요
    model_path = "../exp/object_pose/objectron_book_dla_34_2025-09-02-05-43/book_last.pth"
    sparsity = 0.50  # 50% 필터 제거
    # ==========================

    reduced_model = load_and_reduce_model(model_path, sparsity)

    # 최종 결과 확인
    total_params = sum(p.numel() for p in reduced_model.parameters())
    total_zero = 0
    with torch.no_grad():
        for p in reduced_model.parameters():
            total_zero += (p == 0).sum().item()

    print(f"\n===== Final Results =====")
    print(f"Total parameters: {total_params:,}")
    print(f"Zero parameters: {total_zero:,}")
