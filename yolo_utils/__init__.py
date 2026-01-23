"""
YOLOv8 Compression Utilities

YOLOv8 모델 압축을 위한 유틸리티 모듈입니다.

주요 기능:
    - Structured Pruning (L2-norm 기반)
    - Channel Reducing (물리적 채널 제거)
    - Knowledge Distillation (Teacher-Student)

사용 예시:
    >>> from yolo_utils import yolov8_pruning, yolov8_reducing
    >>> from ultralytics import YOLO
    >>>
    >>> # Pruning 적용
    >>> model = YOLO('yolov8n.pt')
    >>> yolov8_pruning(model.model.model, sparsity=0.3)
    >>>
    >>> # Reducing 적용 (학습 후)
    >>> reduced_model = YOLO('yolov8n.yaml')
    >>> yolov8_reducing(model.model.model, reduced_model.model.model)
"""

from .pruning import (
    yolov8_pruning,
    get_filter_pruning_idx,
    filter_pruning,
    bn_pruning,
    get_filter_pruning_sparsity,
)

from .reducing import (
    yolov8_reducing,
    get_survived_filter_idx,
    conv_reduce,
    bn_reduce,
    fc_reduce,
    copy_layer,
    bn_copy_layer,
)

from .kd import (
    distillation_loss,
    mse_loss,
    compute_kd_loss,
)

from .memory import (
    measure_memory,
    extract_layers,
    model_memory_usage,
    model_memory_usage_with_reducing,
    custom_memory_loss_function,
)

__all__ = [
    # Pruning
    'yolov8_pruning',
    'get_filter_pruning_idx',
    'filter_pruning',
    'bn_pruning',
    'get_filter_pruning_sparsity',
    # Reducing
    'yolov8_reducing',
    'get_survived_filter_idx',
    'conv_reduce',
    'bn_reduce',
    'fc_reduce',
    'copy_layer',
    'bn_copy_layer',
    # Knowledge Distillation
    'distillation_loss',
    'mse_loss',
    'compute_kd_loss',
    # Memory
    'measure_memory',
    'extract_layers',
    'model_memory_usage',
    'model_memory_usage_with_reducing',
    'custom_memory_loss_function',
]

__version__ = '1.0.0'
