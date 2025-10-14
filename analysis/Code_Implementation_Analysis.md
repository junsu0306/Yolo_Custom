# YOLOv8 압축 구현 - 코드 비교 분석

## 목차

1. [수정된 파일 목록](#1-수정된-파일-목록)
2. [실행 흐름](#2-실행-흐름)
3. [코드 비교: 설정 파일](#3-코드-비교-설정-파일)
4. [코드 비교: Trainer 초기화](#4-코드-비교-trainer-초기화)
5. [코드 비교: Training Loop](#5-코드-비교-training-loop)
6. [추가된 압축 모듈](#6-추가된-압축-모듈)
7. [전체 실행 순서](#7-전체-실행-순서)

---

# 1. 수정된 파일 목록

## GitHub 원본에 없는 파일 (새로 추가)

```
ultralytics_custom/
├── engine/
│   ├── compression.py                              # NEW
│   ├── compression_src/                            # NEW
│   │   ├── pruning/
│   │   │   ├── common.py                          # NEW
│   │   │   └── resnet.py                          # NEW
│   │   ├── reducing/
│   │   │   ├── common.py                          # NEW
│   │   │   └── resnet.py                          # NEW
│   │   └── models/
│   │       └── resnet.py                          # NEW
│   ├── model_bak.py                                # NEW
│   └── trainer_bak.py                              # NEW
└── utils/
    └── memory_usage_MH.py                          # NEW
```

## 수정된 파일

```
ultralytics_custom/
├── cfg/
│   ├── default.yaml                    # MODIFIED (Lines 16-19)
│   └── __init__.py                     # MODIFIED (Lines 112, 138)
└── engine/
    └── trainer.py                      # MODIFIED (Lines 11-18, 67-75, 264-272, 429-492)
```

---

# 2. 실행 흐름

```
사용자: yolo train model=yolov8n.pt data=coco.yaml pruning_ratio=0.3
    ↓
[1] 설정 로드 (cfg/default.yaml)
    - pruning_ratio: 0.3
    - distill_ratio: 설정값
    - mem_usg: 목표 메모리
    ↓
[2] Trainer 초기화 (_setup_train)
    - Teacher 모델 로드
    - Teacher 파라미터 고정
    - Student 모델(압축 대상) 로드
    ↓
[3] Training Loop 시작 (_do_train)
    For each batch:
        ↓
    [3-1] Student Forward (distillation=True)
        - Student 모델 forward
        - Loss + Feature 반환
        ↓
    [3-2] Teacher Forward (no_grad)
        - Teacher 모델 forward
        - Soft label 생성
        ↓
    [3-3] Loss 계산
        - Task Loss (detection)
        - KD Loss (classification + bbox)
        - Memory Loss
        ↓
    [3-4] Backward
        - Gradient 계산
        ↓
    [3-5] Optimizer Step
        - Weight 업데이트
        ↓
    [3-6] ★ Pruning ★
        - yolov8_pruning() 호출
        - L2 norm 기반 필터 선택
        - 선택된 필터 가중치 = 0
        ↓
    다음 batch
    ↓
[4] Epoch 종료 후 Validation
    ↓
[5] 학습 완료 후 (선택)
    - yolov8_reducing() 호출
    - 0인 필터 물리적 제거
    - 작은 모델 생성
```

---

# 3. 코드 비교: 설정 파일

## 3.1 cfg/default.yaml

### GitHub 원본
```yaml
# ultralytics_github/cfg/default.yaml (Lines 13-22)

batch: 16
imgsz: 640

save: True
save_period: -1
cache: False
device:
workers: 8
```

### Custom 버전
```yaml
# ultralytics_custom/cfg/default.yaml (Lines 13-24)

batch: 16
imgsz: 640

############################################
pruning_ratio: 0.0 # (float) pruning ratio
mem_usg: 0.0 # (float) model's mem usage
############################################

save: True
save_period: -1
cache: False
device:
workers: 8
```

**변경사항**:
- Lines 16-19: `pruning_ratio`, `mem_usg` 파라미터 추가
- 기본값 0.0 = 압축 비활성화 (backward compatibility)

---

## 3.2 cfg/__init__.py

### GitHub 원본
```python
# ultralytics_github/cfg/__init__.py

VALID_KEYS = {
    "task",
    "mode",
    "model",
    "data",
    "epochs",
    "batch",
    "imgsz",
    # ... 다른 키들
}

NUMERIC_KEYS = {
    "epochs",
    "batch",
    "imgsz",
    # ... 다른 키들
}
```

### Custom 버전
```python
# ultralytics_custom/cfg/__init__.py

VALID_KEYS = {
    "task",
    "mode",
    "model",
    "data",
    "epochs",
    "batch",
    "imgsz",
    "mem_usg",        # ← 추가
    "pruning_ratio",  # ← 추가
    # ... 다른 키들
}

NUMERIC_KEYS = {
    "epochs",
    "batch",
    "imgsz",
    "pruning_ratio",  # ← 추가
    "mem_usg",        # ← 추가
    # ... 다른 키들
}
```

**변경사항**:
- `VALID_KEYS`에 새 파라미터 등록
- `NUMERIC_KEYS`에 숫자형 타입 지정

---

# 4. 코드 비교: Trainer 초기화

## 4.1 Import 추가

**파일**: `engine/trainer.py`

### GitHub 원본 (Lines 1-25)
```python
# ultralytics_github/engine/trainer.py

# Ultralytics 🚀 AGPL-3.0 License
"""
Train a model on a dataset.
"""

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
```

### Custom 버전 (Lines 1-30)
```python
# ultralytics_custom/engine/trainer.py

# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.
"""

import numpy as np
#########################################
############### SM Insert ###############
from model_compression.funcs4 import *
#########################################
#########################################
from ultralytics_custom.models.yolo.model import YOLO
from ultralytics_custom.utils.memory_usage_MH import *
#########################################
import gc
import math
import os
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # ← 추가

from torch import distributed as dist
from torch import nn, optim

from ultralytics_custom.cfg import get_cfg, get_save_dir
from ultralytics_custom.data.utils import check_cls_dataset, check_det_dataset
```

**추가된 Import**:
```python
from model_compression.funcs4 import *           # 외부 압축 라이브러리
from ultralytics_custom.models.yolo.model import YOLO  # Teacher 모델용
from ultralytics_custom.utils.memory_usage_MH import *  # 메모리 측정
import torch.nn.functional as F                  # KD loss용
```

---

## 4.2 Loss 함수 추가

### GitHub 원본
```python
# 없음 - 바로 BaseTrainer 클래스 정의
```

### Custom 버전 (Lines 67-75)
```python
# ultralytics_custom/engine/trainer.py

def distillation_loss(student_logits, teacher_logits, T=2.0):
    s = student_logits / T
    t = teacher_logits / T
    return F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction='batchmean') * T * T


def MSE_loss(student_feat, teacher_feat):
    """Calculate MSE loss between student and teacher features"""
    return F.mse_loss(student_feat, teacher_feat)
```

**추가된 함수**:
- `distillation_loss()`: KL Divergence 계산
- `MSE_loss()`: Feature MSE 계산
- 주의: 정의만 되어있고 실제로는 사용 안 함 (Training loop에서 직접 F.kl_div 사용)

---

## 4.3 Teacher 모델 초기화

**파일**: `engine/trainer.py`
**메서드**: `BaseTrainer._setup_train()`

### GitHub 원본
```python
# ultralytics_github/engine/trainer.py (대략 Line 250 근처)

def _setup_train(self, world_size):
    """Builds dataloaders and optimizer on correct rank process."""

    # 바로 pretrain routine 시작
    self.run_callbacks("on_pretrain_routine_start")
    ckpt = self.setup_model()
    self.model = self.model.to(self.device)
    self.set_model_attributes()

    # Freeze layers
    freeze_list = (
        self.args.freeze
        if isinstance(self.args.freeze, list)
        else range(self.args.freeze)
        if isinstance(self.args.freeze, int)
        else []
    )
    # ... 나머지 초기화
```

### Custom 버전 (Lines 260-280)
```python
# ultralytics_custom/engine/trainer.py

def _setup_train(self, world_size):
    """Builds dataloaders and optimizer on correct rank process."""
##############################################################
    # Model

    # STEP 1: Teacher 모델 정의 (deepcopy or yolov8x.pt 로드)
    #self.teacher_model = deepcopy(self.model).eval().to(self.device)
    self.teacher_model = YOLO('/workspace/TW/YOLO/runs/detect/train_YOLOv8x/weights/best.pt').to(self.device)
    self.teacher_model.args = self.args
    for p in self.teacher_model.parameters():
        p.requires_grad = False

    self.teacher_model.half()




    self.run_callbacks("on_pretrain_routine_start")
    ckpt = self.setup_model()
    self.model = self.model.to(self.device)
    self.set_model_attributes()
##############################################################

    # Freeze layers
    freeze_list = (
        self.args.freeze
        if isinstance(self.args.freeze, list)
        else range(self.args.freeze)
        if isinstance(self.args.freeze, int)
        else []
    )
    # ... 나머지 초기화
```

**추가된 코드 (Lines 264-272)**:
```python
# 주석 처리된 옵션 1: Self-distillation
#self.teacher_model = deepcopy(self.model).eval().to(self.device)

# 실제 사용하는 옵션 2: 사전 학습된 큰 모델
self.teacher_model = YOLO('/workspace/TW/YOLO/runs/detect/train_YOLOv8x/weights/best.pt').to(self.device)
self.teacher_model.args = self.args

# Teacher 파라미터 고정 (학습 안 함)
for p in self.teacher_model.parameters():
    p.requires_grad = False

# FP16 변환 (메모리 절약)
self.teacher_model.half()
```

**실행 순서**:
1. Student 모델 초기화 전에 Teacher 먼저 로드
2. Teacher를 eval mode로 설정 (`.eval()` 는 YOLO 초기화에 포함)
3. Teacher 파라미터를 `requires_grad=False`로 고정
4. FP16으로 변환하여 메모리 절약
5. 이후 Student 모델 초기화

---

# 5. 코드 비교: Training Loop

**파일**: `engine/trainer.py`
**메서드**: `BaseTrainer._do_train()`

## 5.1 Forward Pass

### GitHub 원본
```python
# ultralytics_github/engine/trainer.py (대략 Line 420 근처)

# Forward
with autocast(self.amp):
    self.loss, self.loss_items = self.model(batch)

    if RANK != -1:
        self.loss *= world_size
    self.tloss = (
        (self.tloss * i + self.loss_items) / (i + 1)
        if self.tloss is not None else self.loss_items
    )
```

### Custom 버전 (Lines 429-468)
```python
# ultralytics_custom/engine/trainer.py

# Forward
with autocast(self.amp):
    # 메모리 추적
    #self.args.mem_usg = model_memory_usage_with_reducing(dummy_input, self.model, device = self.device)
    self.args.mem_usg = 100

    hyperparam = 0.5
    dummy_input = torch.zeros(1, 3, 640, 640).half().to(self.device)
    device_condition_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    # Student forward (distillation mode)
    self.loss, self.loss_items, feats = self.model(batch, distillation=True)

    # Teacher forward
    with torch.no_grad():
        teacher_out = self.teacher_model.model(batch['img'])  # inference

        if self.epoch > 0:  # Skip first epoch
            # Extract features
            stu_result = feats[0]
            teacher_result = teacher_out[0]

            # Classification loss (KL Divergence)
            s_cls = F.log_softmax(stu_result[:, 4:, :], dim=1)
            t_cls = F.softmax(teacher_result[:, 4:, :], dim=1)

            loss_cls = F.kl_div(s_cls, t_cls, reduction='batchmean')

            # Bbox regression loss (MSE)
            loss_bbox = F.mse_loss(stu_result[:, 0:4, :], teacher_result[:, 0:4, :])

            # Total KD loss
            total_kd_loss = (loss_cls + loss_bbox) * self.args.distill_ratio

            # Memory loss
            self.loss += total_kd_loss
            self.loss += max(0, self.args.mem_usg - device_condition_mem) * hyperparam

    if RANK != -1:
        self.loss *= world_size
    self.tloss = (
        (self.tloss * i + self.loss_items) / (i + 1)
        if self.tloss is not None else self.loss_items
    )
```

**변경사항 상세**:

#### Step 1: 메모리 측정 (Lines 429-433)
```python
#self.args.mem_usg = model_memory_usage_with_reducing(dummy_input, self.model, device = self.device)
self.args.mem_usg = 100  # 하드코딩된 목표 메모리 (MB)

hyperparam = 0.5  # 메모리 loss 가중치
dummy_input = torch.zeros(1, 3, 640, 640).half().to(self.device)
device_condition_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
```

**작동**:
- 현재 GPU 메모리 사용량을 `device_condition_mem`에 저장
- 목표 메모리 `self.args.mem_usg = 100` (MB)
- 초과 시 loss에 패널티 추가

#### Step 2: Student Forward (Line 436)
```python
self.loss, self.loss_items, feats = self.model(batch, distillation=True)
```

**GitHub 원본과 차이**:
```python
# 원본
self.loss, self.loss_items = self.model(batch)

# Custom
self.loss, self.loss_items, feats = self.model(batch, distillation=True)
```

**변경점**:
- `distillation=True` 인자 추가
- `feats` 추가 반환 (중간 feature map)
- 주의: 이를 위해 `model.py`도 수정 필요 (코드에는 없음, 가정)

#### Step 3: Teacher Forward (Lines 439-441)
```python
with torch.no_grad():
    teacher_out = self.teacher_model.model(batch['img'])  # inference
```

**작동**:
- `torch.no_grad()`: Teacher는 gradient 계산 안 함
- `teacher_out`: Teacher의 출력 (logits)

#### Step 4: Epoch Gating (Line 443)
```python
if self.epoch > 0:  # Skip first epoch
```

**이유**:
- 첫 epoch에는 distillation 안 함
- 초기 모델이 불안정하여 Teacher와 차이가 너무 큼

#### Step 5: Feature 추출 (Lines 445-448)
```python
# Extract features
stu_result = feats[0]          # Student logits [batch, channels, anchors]
teacher_result = teacher_out[0]  # Teacher logits [batch, channels, anchors]
```

**YOLO 출력 형식**:
```
output shape: [batch, 84, 8400]
              ↑      ↑   ↑
           batch  channels anchors

channels:
  [0:4]   : bbox (x, y, w, h)
  [4:84]  : class probabilities (80 classes)
```

#### Step 6: Classification KD Loss (Lines 450-454)
```python
# Classification loss (KL Divergence)
s_cls = F.log_softmax(stu_result[:, 4:, :], dim=1)  # Student: log(P)
t_cls = F.softmax(teacher_result[:, 4:, :], dim=1)  # Teacher: P

loss_cls = F.kl_div(s_cls, t_cls, reduction='batchmean')
```

**작동**:
- `stu_result[:, 4:, :]`: Student의 class 확률 (80 classes)
- `teacher_result[:, 4:, :]`: Teacher의 class 확률
- `F.kl_div()`: KL Divergence 계산
  - `s_cls`: log probabilities (student)
  - `t_cls`: probabilities (teacher)

#### Step 7: Bbox Regression Loss (Line 456)
```python
# Bbox regression loss (MSE)
loss_bbox = F.mse_loss(stu_result[:, 0:4, :], teacher_result[:, 0:4, :])
```

**작동**:
- `stu_result[:, 0:4, :]`: Student의 bbox (x, y, w, h)
- `teacher_result[:, 0:4, :]`: Teacher의 bbox
- MSE: Mean Squared Error

#### Step 8: Total KD Loss (Line 458)
```python
# Total KD loss
total_kd_loss = (loss_cls + loss_bbox) * self.args.distill_ratio
```

**작동**:
- `self.args.distill_ratio`: 설정 파일에서 가져옴
- Classification loss + Bbox loss를 합쳐서 가중치 적용

#### Step 9: Loss 통합 (Lines 463-468)
```python
# Add to total loss
self.loss += total_kd_loss
self.loss += max(0, self.args.mem_usg - device_condition_mem) * hyperparam
```

**최종 Loss**:
```python
total_loss = task_loss + kd_loss + memory_loss

여기서:
- task_loss: self.loss (원래 YOLO detection loss)
- kd_loss: total_kd_loss
- memory_loss: max(0, target - current) * 0.5
```

---

## 5.2 Backward & Optimize

### GitHub 원본
```python
# ultralytics_github/engine/trainer.py

# Backward
self.scaler.scale(self.loss).backward()

# Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
if ni - last_opt_step >= self.accumulate:
    self.optimizer_step()
    last_opt_step = ni
```

### Custom 버전 (Lines 476-492)
```python
# ultralytics_custom/engine/trainer.py

# Backward
self.scaler.scale(self.loss).backward()

# Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
if ni - last_opt_step >= self.accumulate:
    self.optimizer_step()
    ########################################################################
    ##################SM Insert ############################################
    if self.args.pruning_ratio == 0:
        continue
    else:
        yolov8_pruning(self.model.model, self.args.pruning_ratio, device = self.device)# Prune YOLOv8n

    ########################################################################

    last_opt_step = ni
```

**추가된 코드 (Lines 484-487)**:
```python
if self.args.pruning_ratio == 0:
    continue
else:
    yolov8_pruning(self.model.model, self.args.pruning_ratio, device = self.device)
```

**작동 순서**:
1. Gradient accumulation 완료 체크 (`ni - last_opt_step >= self.accumulate`)
2. Optimizer step 실행 (weight update)
3. **★ Pruning 실행 ★** (매 optimizer step 후)
   - `pruning_ratio == 0`이면 skip (압축 비활성화)
   - 아니면 `yolov8_pruning()` 호출
4. 다음 iteration

---

# 6. 추가된 압축 모듈

## 6.1 compression.py - YOLOv8 Pruning

**파일**: `ultralytics_custom/engine/compression.py`

```python
from compression_src.pruning.common import *
from compression_src.reducing.common import *

import torch
from math import *

def yolov8_pruning(model, sparsity):
    # Create a list of (backbone) blocks from the model
    block_list = [model[i] for i in range(23)]

    # Iterate through each block in the model
    for i, block in enumerate(block_list):
        # Check if the block is a Conv layer
        if type(block).__name__ == 'Conv':
            # Get the indices of filters to be pruned based on the specified sparsity
            pruning_idx = get_filter_pruning_idx(layer=block.conv, sparsity=sparsity)
            # Perform filter pruning on the convolutional layer using the calculated indices
            filter_pruning(layer=block.conv, pruning_idx=pruning_idx)
            # Prune the corresponding batch normalization layer using the same indices
            bn_pruning(block.bn, pruning_idx=pruning_idx)

        # Check if the block is a C2f or SPPF layer
        if type(block).__name__ in ['C2f', 'SPPF']:
            # Get the indices of filters to be pruned based on the specified sparsity
            pruning_idx = get_filter_pruning_idx(layer=block.cv2.conv, sparsity=sparsity)
            # Perform filter pruning on the convolutional layer using the calculated indices
            filter_pruning(layer=block.cv2.conv, pruning_idx=pruning_idx)
            # Prune the corresponding batch normalization layer using the same indices
            bn_pruning(block.cv2.bn, pruning_idx=pruning_idx)

        # Check if the block is a Detect layer
        if type(block).__name__ == 'Detect':
            # Iterate through each set of convolutional layers in the Detect layer
            for i in range(3):
                for j in range(2):
                    # Get the indices of filters to be pruned in cv2 convolutional layers
                    pruning_idx = get_filter_pruning_idx(layer=block.cv2[i][j].conv, sparsity=sparsity)
                    # Perform filter pruning on the cv2 convolutional layers using the calculated indices
                    filter_pruning(layer=block.cv2[i][j].conv, pruning_idx=pruning_idx)
                    # Prune the corresponding batch normalization layers using the same indices
                    bn_pruning(block.cv2[i][j].bn, pruning_idx=pruning_idx)

                    # Get the indices of filters to be pruned in cv3 convolutional layers
                    pruning_idx = get_filter_pruning_idx(layer=block.cv3[i][j].conv, sparsity=sparsity)
                    # Perform filter pruning on the cv3 convolutional layers using the calculated indices
                    filter_pruning(layer=block.cv3[i][j].conv, pruning_idx=pruning_idx)
                    # Prune the corresponding batch normalization layers using the same indices
                    bn_pruning(block.cv3[i][j].bn, pruning_idx=pruning_idx)
```

**작동 순서**:
1. YOLOv8 모델에서 레이어 0-22 추출 (Backbone + Neck)
2. 각 블록 타입에 따라 처리:
   - **Conv**: `block.conv`, `block.bn` 프루닝
   - **C2f/SPPF**: `block.cv2.conv`, `block.cv2.bn` 프루닝
   - **Detect**: `cv2[i][j]`, `cv3[i][j]` 모두 프루닝 (3 scales × 2 layers)

---

## 6.2 pruning/common.py - 프루닝 함수

**파일**: `ultralytics_custom/engine/compression_src/pruning/common.py`

### get_filter_pruning_idx()

```python
def get_filter_pruning_idx(layer, sparsity):
    with torch.no_grad():
        weight = layer.weight                              # [out_ch, in_ch, h, w]
        num_filters = weight.shape[0]                      # 출력 채널 수
        num_pruning_filters = int(num_filters * sparsity)  # 제거할 개수

        # 각 필터의 L2 norm 계산
        filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)

        # L2 norm이 가장 작은 필터들의 인덱스 반환
        _, pruning_idx = torch.topk(filter_norms, num_pruning_filters, largest=False)
    return pruning_idx
```

**작동**:
1. Layer의 weight 가져오기 `[256, 128, 3, 3]`
2. 각 필터를 1D로 펼치기 `[256, 1152]`
3. L2 norm 계산: `||filter_i||_2 = sqrt(Σ w^2)`
4. `torch.topk(..., largest=False)`: 가장 작은 norm의 필터 선택
5. 제거할 필터의 인덱스 반환

**예시**:
```python
# 입력
weight.shape = [256, 128, 3, 3]
sparsity = 0.3

# 계산
num_pruning_filters = int(256 * 0.3) = 76
filter_norms = [0.8, 0.3, 1.2, ..., 0.5]  # 256개

# topk(76, largest=False)
pruning_idx = [1, 17, 89, ...]  # norm이 작은 76개 인덱스
```

### filter_pruning()

```python
def filter_pruning(layer, pruning_idx):
    weight = layer.weight
    with torch.no_grad():
        weight[pruning_idx, :, :, :] = 0.0
```

**작동**:
- 선택된 필터의 모든 가중치를 0으로 설정
- In-place 수정 (원본 모델 직접 변경)

**예시**:
```python
# Before
weight[1] = [[0.5, -0.3], [0.8, 0.2]]  # 필터 1
weight[17] = [[0.1, 0.05], [0.02, 0.0]]  # 필터 17

# After filter_pruning([1, 17])
weight[1] = [[0.0, 0.0], [0.0, 0.0]]  # 모두 0
weight[17] = [[0.0, 0.0], [0.0, 0.0]]  # 모두 0
```

### bn_pruning()

```python
def bn_pruning(layer, pruning_idx):
    weight = layer.weight        # γ (scale)
    bias = layer.bias            # β (shift)
    mean = layer.running_mean    # μ
    var = layer.running_var      # σ²

    with torch.no_grad():
        weight[pruning_idx] = 0.0   # γ = 0
        bias[pruning_idx] = 0.0     # β = 0
        mean[pruning_idx] = 0.0     # μ = 0
        var[pruning_idx] = 1.0      # σ² = 1
```

**작동**:
- Batch Normalization 파라미터 조정
- 프루닝된 필터에 대응하는 BN 파라미터 설정

**결과**:
```
BN(x) = γ * ((x - μ) / sqrt(σ²)) + β
      = 0 * ((x - 0) / sqrt(1)) + 0
      = 0
```
→ Conv 출력이 0이면 BN 출력도 0

---

## 6.3 yolov8_reducing() - 모델 축소

**파일**: `ultralytics_custom/engine/compression.py`

```python
def yolov8_reducing(model, reduced_model):
    # Get the indices of filters that survived the first Conv layer
    survived_idx = get_survived_filter_idx(model[0].conv)

    # Reduce the first Conv layer using the survived indices
    conv_reduce(
        layer=model[0].conv,
        reduced_layer=reduced_model[0].conv,
        survived_out_channels_idx=survived_idx,
        survived_in_channels_idx=torch.arange(3),  # RGB input channels
    )

    # Reduce the corresponding Batch Normalization layer using the same indices
    bn_reduce(model[0].bn, reduced_model[0].bn, survived_idx)

    # Initialize previous survived indices for use in subsequent layers
    prev_survived_idx = survived_idx

    # Create lists of blocks from the original model and the reduced model
    block_list = [model[i] for i in range(1, 23)]
    reduced_block_list = [reduced_model[i] for i in range(1, 23)]

    # Iterate through each block and reduce it
    for i, (block, reduced_block) in enumerate(zip(block_list, reduced_block_list)):

        # Check if the current block is a Conv layer
        if type(block).__name__ == 'Conv':

            # Get the indices of filters that survived in the current Conv layer
            survived_idx = get_survived_filter_idx(block.conv)

            # Reduce the current Conv layer using the survived indices
            conv_reduce(
                layer=block.conv,
                reduced_layer=reduced_block.conv,
                survived_out_channels_idx=survived_idx,
                survived_in_channels_idx=prev_survived_idx,
            )

            # Reduce the corresponding Batch Normalization layer using the same indices
            bn_reduce(block.bn, reduced_block.bn, survived_idx)

        # Check if the current block is a C2f layer
        elif type(block).__name__ in ['C2f', 'SPPF']:

            # Handle prev_survived_idx separately for layers where the input is created by concatenating outputs from multiple layers
            if i in [11, 14, 17, 20]:
                # Define n1 and n2 as the layers whose outputs are concatenated
                n1, n2 = (8, 5) if i == 11 else (11, 3) if i == 14 else (15, 11) if i == 17 else (18, 8) if i == 20 else (None, None)

                # Get the survived indices from the first layer
                p_surv_idx_1 = get_survived_filter_idx(block_list[n1].conv if i in [17, 20] else block_list[n1].cv2.conv)

                # Determine the number of output channels from the first layer (n1) to adjust the indices accordingly.
                p_surv_idx_2p = block_list[n1].conv.out_channels if i in [17, 20] else block_list[n1].cv2.conv.out_channels

                # Get the survived indices from the second layer
                p_surv_idx_2 = get_survived_filter_idx(block_list[n2].cv2.conv) + p_surv_idx_2p

                # Concatenate the indices from the first and second layers to create prev_survived_idx (input channels for the current block)
                prev_survived_idx = torch.concat([p_surv_idx_1, p_surv_idx_2])

            # Reduce cv1, cv2 layers ...
            # (나머지 코드 생략)
```

**작동 순서**:
1. 첫 Conv 레이어 축소 (RGB 입력은 항상 3 채널)
2. 레이어 1-22 순회하면서:
   - **Conv**: survived indices 추적하여 축소
   - **C2f/SPPF**:
     - Concatenation 레이어면 (11, 14, 17, 20) 특별 처리
     - 두 입력의 survived indices를 concat
   - **Detect**: 3개 스케일 각각 처리

---

## 6.4 reducing/common.py - 축소 함수

### get_survived_filter_idx()

```python
def get_survived_filter_idx(layer):
    weight = layer.weight
    num_filters = weight.shape[0]
    filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)
    survived_filter_idx = torch.where(filter_norms != 0)[0]
    return survived_filter_idx
```

**작동**:
- L2 norm 계산
- norm != 0인 필터만 선택 (pruning에서 0으로 만든 필터 제외)

**예시**:
```python
filter_norms = [0.8, 0.0, 1.2, 0.0, 0.9, 0.5]
                 ↓
survived_idx = [0, 2, 4, 5]  # 0이 아닌 필터
```

### conv_reduce()

```python
def conv_reduce(
    layer, reduced_layer, survived_out_channels_idx, survived_in_channels_idx
):

    # set reduced_layer's in_channels and out_channels
    reduced_layer.in_channels = len(survived_in_channels_idx)
    reduced_layer.out_channels = len(survived_out_channels_idx)

    # in case of depthwise conv, the number of groups and out_channels have to be same
    if reduced_layer.groups != 1:
        reduced_layer.groups = reduced_layer.out_channels

    # re-defining the weight parameter of reduced_layer
    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(
            reduced_layer.out_channels,
            reduced_layer.in_channels,
            reduced_layer.kernel_size[0],
            reduced_layer.kernel_size[1],
        ),
        requires_grad=True,
    )

    weight = layer.weight
    reduced_weight = reduced_layer.weight

    # copy the survived weights to the reduced layer
    with torch.no_grad():
        reduced_weight.copy_(
            weight[survived_out_channels_idx, :, :, :][
                :, survived_in_channels_idx, :, :
            ]
        )
```

**작동 순서**:
1. 새 레이어의 채널 수 설정
2. Depthwise conv면 groups 조정
3. 새로운 가중치 텐서 생성 (작은 크기)
4. 살아남은 가중치만 복사

**예시**:
```python
# 원본
weight.shape = [256, 128, 3, 3]
survived_out = [0, 2, 4, ..., 250]  # 206개
survived_in = [0, 1, 3, ..., 127]   # 115개

# 새 레이어
reduced_weight.shape = [206, 115, 3, 3]

# 복사
reduced_weight[:,:,:,:] = weight[survived_out][:, survived_in, :, :]
```

---

# 7. 전체 실행 순서

## 7.1 학습 시작

```bash
yolo train model=yolov8n.pt data=coco.yaml pruning_ratio=0.3 distill_ratio=0.5
```

**실행 흐름**:

```
[1] 설정 파일 로드
    ├─ cfg/default.yaml 읽기
    ├─ pruning_ratio: 0.3
    ├─ distill_ratio: 0.5 (가정)
    └─ mem_usg: 100.0 (가정)

[2] Trainer 생성 및 초기화
    ├─ BaseTrainer.__init__()
    └─ _setup_train() 호출
        ├─ [2-1] Teacher 모델 로드
        │   ├─ YOLO('...yolov8x.../best.pt')
        │   ├─ teacher_model.eval()
        │   ├─ requires_grad = False
        │   └─ teacher_model.half()
        │
        ├─ [2-2] Student 모델 로드
        │   ├─ setup_model()
        │   └─ model.to(device)
        │
        ├─ [2-3] Dataloader 생성
        └─ [2-4] Optimizer 생성

[3] Training Loop 시작 (_do_train)
    └─ For epoch in range(epochs):
        └─ For batch_idx, batch in enumerate(train_loader):

            [3-1] Forward Pass
                ├─ autocast 활성화
                ├─ 메모리 측정
                │   └─ device_condition_mem = torch.cuda.max_memory_allocated()
                │
                ├─ Student Forward
                │   └─ loss, loss_items, feats = model(batch, distillation=True)
                │
                ├─ Teacher Forward (if epoch > 0)
                │   ├─ with torch.no_grad():
                │   │   └─ teacher_out = teacher_model.model(batch['img'])
                │   │
                │   ├─ Feature 추출
                │   │   ├─ stu_result = feats[0]
                │   │   └─ teacher_result = teacher_out[0]
                │   │
                │   ├─ Classification KD Loss
                │   │   ├─ s_cls = F.log_softmax(stu_result[:, 4:, :])
                │   │   ├─ t_cls = F.softmax(teacher_result[:, 4:, :])
                │   │   └─ loss_cls = F.kl_div(s_cls, t_cls)
                │   │
                │   ├─ Bbox Regression Loss
                │   │   └─ loss_bbox = F.mse_loss(stu[:, 0:4], tea[:, 0:4])
                │   │
                │   ├─ Total KD Loss
                │   │   └─ kd_loss = (loss_cls + loss_bbox) * distill_ratio
                │   │
                │   └─ Memory Loss
                │       └─ mem_loss = max(0, target - current) * 0.5
                │
                └─ Loss 통합
                    └─ total_loss = task_loss + kd_loss + mem_loss

            [3-2] Backward Pass
                └─ scaler.scale(loss).backward()

            [3-3] Optimizer Step (if accumulation done)
                ├─ optimizer_step()
                │
                └─ [3-4] ★ Pruning ★ (if pruning_ratio > 0)
                    └─ yolov8_pruning(model, sparsity=0.3)
                        ├─ For each block (0-22):
                        │   ├─ Conv:
                        │   │   ├─ pruning_idx = get_filter_pruning_idx(block.conv, 0.3)
                        │   │   │   ├─ filter_norms = torch.norm(weight)
                        │   │   │   └─ topk(norm, largest=False)
                        │   │   ├─ filter_pruning(block.conv, pruning_idx)
                        │   │   │   └─ weight[pruning_idx] = 0.0
                        │   │   └─ bn_pruning(block.bn, pruning_idx)
                        │   │       ├─ weight[pruning_idx] = 0.0
                        │   │       ├─ bias[pruning_idx] = 0.0
                        │   │       ├─ mean[pruning_idx] = 0.0
                        │   │       └─ var[pruning_idx] = 1.0
                        │   │
                        │   ├─ C2f/SPPF:
                        │   │   └─ (cv2에 대해 동일 작업)
                        │   │
                        │   └─ Detect:
                        │       └─ For 3 scales × 2 Conv:
                        │           └─ (cv2, cv3에 대해 동일 작업)

            다음 batch

        [4] Epoch 종료
            └─ Validation
                └─ metrics = model.val()

[5] Training 종료

[6] (선택) Reducing 실행
    └─ yolov8_reducing(model, reduced_model)
        ├─ For each layer:
        │   ├─ survived_idx = get_survived_filter_idx(layer)
        │   │   └─ torch.where(filter_norms != 0)
        │   │
        │   ├─ conv_reduce(layer, reduced_layer, survived_idx)
        │   │   ├─ 새 레이어 생성 (작은 크기)
        │   │   └─ 살아남은 가중치만 복사
        │   │
        │   └─ bn_reduce(bn, reduced_bn, survived_idx)
        │
        ├─ Concatenation 레이어 (11, 14, 17, 20) 특별 처리
        │   ├─ 두 입력의 survived_idx 가져오기
        │   ├─ offset 추가 (두 번째 입력)
        │   └─ concat
        │
        └─ 축소된 모델 반환

[7] 모델 저장
    └─ reduced_model.save('compressed.pt')
```

---

## 7.2 각 Iteration에서의 Loss 계산

```
매 Training Iteration:

┌─────────────────────────────────────┐
│ Student Forward                     │
│ ├─ loss (detection loss)            │
│ └─ feats (중간 feature)              │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Teacher Forward (no_grad)           │
│ └─ teacher_out (soft label)         │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Loss Calculation                    │
│                                     │
│ task_loss = loss (from student)     │
│                                     │
│ if epoch > 0:                       │
│   cls_loss = KL_Div(stu, tea)       │
│   bbox_loss = MSE(stu, tea)         │
│   kd_loss = (cls + bbox) * 0.5      │
│                                     │
│   mem_loss = max(0, 100 - cur) * 0.5│
│                                     │
│ total = task + kd + mem             │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Backward                            │
│ └─ total_loss.backward()            │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Optimizer Step                      │
│ └─ weight update                    │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ ★ Pruning ★                         │
│ └─ if pruning_ratio > 0:            │
│     ├─ get_filter_pruning_idx()     │
│     ├─ filter_pruning()             │
│     └─ bn_pruning()                 │
└─────────────────────────────────────┘
```

---

## 7.3 Pruning 세부 실행

```
yolov8_pruning(model, sparsity=0.3) 호출:

YOLOv8 구조:
├─ [0] Conv         ← 프루닝
├─ [1] Conv         ← 프루닝
├─ [2] C2f          ← cv2 프루닝
├─ [3] Conv         ← 프루닝
├─ [4] C2f          ← cv2 프루닝
├─ ...
├─ [21] C2f         ← cv2 프루닝
├─ [22] Conv        ← 프루닝
└─ [23] Detect      ← cv2, cv3 프루닝 (3 scales × 2 layers)

각 레이어에서:
┌──────────────────────────────────────────┐
│ 1. get_filter_pruning_idx(layer, 0.3)    │
│    ├─ weight.shape = [256, 128, 3, 3]    │
│    ├─ filter_norms = [0.8, 0.3, 1.2, ...]│
│    ├─ num_pruning = int(256 * 0.3) = 76  │
│    └─ return [1, 17, 89, ...] (76개)     │
└──────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────┐
│ 2. filter_pruning(layer, [1, 17, ...])   │
│    └─ weight[[1,17,...], :, :, :] = 0.0  │
└──────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────┐
│ 3. bn_pruning(bn, [1, 17, ...])          │
│    ├─ weight[[1,17,...]] = 0.0           │
│    ├─ bias[[1,17,...]] = 0.0             │
│    ├─ mean[[1,17,...]] = 0.0             │
│    └─ var[[1,17,...]] = 1.0              │
└──────────────────────────────────────────┘

결과:
- 76개 필터가 0이 됨
- 모델 크기는 동일 (아직 제거 안 함)
- 하지만 해당 필터는 출력이 0
```

---

## 7.4 Reducing 세부 실행 (학습 후)

```
yolov8_reducing(model, reduced_model) 호출:

레이어 0 (첫 Conv):
┌──────────────────────────────────────────┐
│ 1. get_survived_filter_idx(model[0].conv)│
│    ├─ filter_norms = [0.8, 0, 1.2, 0, ...│
│    └─ survived = [0, 2, 4, 5, ...]  (206) │
└──────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────┐
│ 2. conv_reduce()                          │
│    ├─ 원본: [256, 3, 3, 3]               │
│    ├─ 축소: [206, 3, 3, 3]               │
│    └─ 복사: weight[survived, :, :, :]    │
└──────────────────────────────────────────┘

레이어 1-22:
For each layer:
    ┌──────────────────────────────────────┐
    │ 1. survived_out = get_survived(layer)│
    │    └─ 현재 레이어의 출력 필터 중 0 아닌것│
    │                                      │
    │ 2. survived_in = prev_survived_idx   │
    │    └─ 이전 레이어의 출력 (현재 입력)  │
    │                                      │
    │ 3. 특별 처리: Concatenation 레이어   │
    │    if i in [11, 14, 17, 20]:        │
    │      ├─ input1 = layer 8 (또는 11...)│
    │      ├─ input2 = layer 5 (또는 3...) │
    │      ├─ offset = input1 채널 수       │
    │      ├─ surv1 = [0,2,5,...]          │
    │      ├─ surv2 = [1,3,4,...] + offset │
    │      └─ survived_in = concat(surv1,2)│
    │                                      │
    │ 4. conv_reduce()                     │
    │    ├─ 원본: [512, 256, 3, 3]         │
    │    ├─ 축소: [410, 206, 3, 3]         │
    │    └─ 복사                           │
    │                                      │
    │ 5. prev_survived = survived_out      │
    └──────────────────────────────────────┘

결과:
- 실제로 작은 모델 생성
- [256,512,256,...] → [206,410,218,...]
- 메모리 절약, 속도 향상
```

---

## 요약

### GitHub 원본 → Custom 변경사항

1. **설정 파일** (`cfg/`):
   - `pruning_ratio`, `mem_usg` 파라미터 추가

2. **Trainer 초기화** (`trainer.py:264-272`):
   - Teacher 모델 로드 및 고정

3. **Training Loop** (`trainer.py:429-492`):
   - Student forward → `distillation=True` 모드
   - Teacher forward → `no_grad()`
   - KD loss 계산 (classification + bbox)
   - Memory loss 계산
   - **Optimizer step 후 Pruning 실행**

4. **새 모듈** (`compression.py`, `compression_src/`):
   - `yolov8_pruning()`: L2 norm 기반 필터 선택 및 0으로 설정
   - `yolov8_reducing()`: 0인 필터 물리적 제거

### 실행 순서

```
설정 로드 → Teacher 초기화 → Training Loop
  └─ (매 iteration)
      Forward → Loss(task+kd+mem) → Backward → Optimize → ★Pruning★
  └─ (학습 후)
      Reducing → 작은 모델 생성
```
