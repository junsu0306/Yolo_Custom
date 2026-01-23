"""
YOLOv8 Knowledge Distillation Module

Teacher-Student 학습을 위한 Knowledge Distillation 함수들입니다.

주요 함수:
    - distillation_loss: KL Divergence 기반 분류 distillation loss
    - mse_loss: MSE 기반 feature distillation loss
"""

import torch
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    KL Divergence 기반 Knowledge Distillation loss를 계산합니다.

    Soft targets를 사용하여 teacher의 지식을 student에게 전달합니다.
    Temperature가 높을수록 soft probability distribution이 됩니다.

    Args:
        student_logits: Student 모델의 출력 logits
        teacher_logits: Teacher 모델의 출력 logits
        temperature: Softmax temperature (default: 2.0)

    Returns:
        loss: KL Divergence loss (temperature로 스케일링됨)

    Example:
        >>> student_out = student_model(images)
        >>> with torch.no_grad():
        >>>     teacher_out = teacher_model(images)
        >>> kd_loss = distillation_loss(student_out, teacher_out, T=2.0)
    """
    s = student_logits / temperature
    t = teacher_logits / temperature
    return F.kl_div(
        F.log_softmax(s, dim=-1),
        F.softmax(t, dim=-1),
        reduction='batchmean'
    ) * temperature * temperature


def mse_loss(student_feat, teacher_feat):
    """
    MSE 기반 Feature Distillation loss를 계산합니다.

    Intermediate feature maps 또는 bounding box regression에 사용됩니다.

    Args:
        student_feat: Student 모델의 feature map
        teacher_feat: Teacher 모델의 feature map

    Returns:
        loss: MSE loss

    Example:
        >>> # Bounding box regression distillation
        >>> loss_bbox = mse_loss(student_result[:, 0:4, :], teacher_result[:, 0:4, :])
    """
    return F.mse_loss(student_feat, teacher_feat)


def compute_kd_loss(student_output, teacher_output, distill_ratio=0.5, temperature=2.0):
    """
    YOLOv8 전용 Knowledge Distillation loss를 계산합니다.

    Classification과 Bounding Box regression 모두에 대해 distillation을 수행합니다.

    Args:
        student_output: Student 모델의 출력 (shape: [B, C, N])
        teacher_output: Teacher 모델의 출력 (shape: [B, C, N])
        distill_ratio: Total loss에서 KD loss의 비율 (default: 0.5)
        temperature: Softmax temperature (default: 2.0)

    Returns:
        total_kd_loss: Classification + BBox KD loss

    Example:
        >>> student_out = student_model(batch['img'])
        >>> with torch.no_grad():
        >>>     teacher_out = teacher_model(batch['img'])
        >>> kd_loss = compute_kd_loss(student_out[0], teacher_out[0])
        >>> total_loss = task_loss + kd_loss
    """
    # Classification distillation (채널 4 이후가 class logits)
    student_cls = student_output[:, 4:, :]
    teacher_cls = teacher_output[:, 4:, :]

    # Soft targets with temperature
    t_cls = F.softmax(teacher_cls, dim=1)
    s_cls = F.log_softmax(student_cls, dim=1)
    loss_cls = F.kl_div(s_cls, t_cls, reduction='batchmean')

    # Bounding box regression distillation (채널 0-4가 bbox)
    loss_bbox = F.mse_loss(student_output[:, 0:4, :], teacher_output[:, 0:4, :])

    # Total KD loss
    total_kd_loss = (loss_cls + loss_bbox) * distill_ratio

    return total_kd_loss
