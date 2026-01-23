"""
CenterPose Compression Utilities

CenterPose (DLA-34) 모델 압축을 위한 유틸리티 모듈입니다.

주요 기능:
    - Structured Pruning (L2-norm 기반, BasicBlock 단위)
    - Channel Reducing (물리적 채널 제거)

사용 예시:
    >>> from centerpose_utils import dlasg_blockwise_pruning, reduce_pruned_model
    >>>
    >>> # Pruning 적용
    >>> model = load_model(...)
    >>> pruned_model = dlasg_blockwise_pruning(model, sparsity=0.3)
    >>>
    >>> # Reducing 적용
    >>> reduced_model = reduce_pruned_model(pruned_model)
"""

from .pruning import (
    dlasg_blockwise_pruning,
    filter_pruning,
    bn_pruning,
    get_filter_norms,
    get_pruning_indices,
)

from .reducing import (
    reduce_pruned_model,
    get_survived_filter_idx,
    conv_reduce,
    bn_reduce,
    copy_layer,
    bn_copy_layer,
)

from .memory import (
    measure_memory,
    extract_layers,
    measure_model_memory,
    measure_pruned_layer_memory,
    custom_memory_loss_function,
)

__all__ = [
    # Pruning
    'dlasg_blockwise_pruning',
    'filter_pruning',
    'bn_pruning',
    'get_filter_norms',
    'get_pruning_indices',
    # Reducing
    'reduce_pruned_model',
    'get_survived_filter_idx',
    'conv_reduce',
    'bn_reduce',
    'copy_layer',
    'bn_copy_layer',
    # Memory
    'measure_memory',
    'extract_layers',
    'measure_model_memory',
    'measure_pruned_layer_memory',
    'custom_memory_loss_function',
]

__version__ = '1.0.0'
