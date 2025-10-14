# 기존 코드에 압축 기능 통합 - 상세 가이드

## 개요

이 문서는 원본 `ultralytics_github` 코드에 AI 모델 압축 기능을 어떻게 통합했는지 상세히 설명합니다.

### 통합 전략

1. **새로운 모듈 추가**: `compression.py` 및 `compression_src/` 디렉토리
2. **기존 코드 수정**: `trainer.py`, `default.yaml`, `__init__.py`
3. **추가 기능**: Knowledge Distillation, Memory-aware Training

---

## 1. 설정 파일 수정 (Configuration)

### 파일: `cfg/default.yaml`

**위치**: Lines 16-19

#### 원본 (ultralytics_github)
```yaml
# Ultralytics 🚀 AGPL-3.0 License
task: detect
mode: train
# Train settings
model:
data:
epochs: 100
...
batch: 16
imgsz: 640
save: True
```

#### 수정 후 (ultralytics_custom)
```yaml
# Ultralytics 🚀 AGPL-3.0 License
task: detect
mode: train
# Train settings
model:
data:
epochs: 100
...
batch: 16
imgsz: 640

############################################
pruning_ratio: 0.0 # (float) pruning ratio
mem_usg: 0.0 # (float) model's mem usage
############################################

save: True
```

**추가된 파라미터**:
- `pruning_ratio`: 프루닝 비율 (0.0 = 비활성화, 0.3 = 30% 프루닝)
- `mem_usg`: 메모리 사용량 목표 (MB 단위)

**목적**:
- 사용자가 YAML 파일로 압축 설정 제어
- 기본값 0.0으로 기존 동작 유지 (backward compatibility)

---

## 2. Config 모듈 수정

### 파일: `cfg/__init__.py`

원본 코드에 새 파라미터를 등록해야 합니다.

#### 추가 위치
```python
# ultralytics_custom/cfg/__init__.py

# Line 112 근처
VALID_KEYS = {
    ...
    "mem_usg",      # ← 추가
    ...
}

# Line 138 근처
NUMERIC_KEYS = {
    ...
    "pruning_ratio",  # ← 추가
    ...
}
```

**목적**:
- 새 파라미터를 유효한 설정으로 인식
- 타입 검증 (숫자형)

---

## 3. Trainer 수정 - Part 1: Import 추가

### 파일: `engine/trainer.py`

#### 원본 (ultralytics_github) - Lines 1-60
```python
# Ultralytics 🚀 AGPL-3.0 License
"""
Train a model on a dataset.
...
"""

import gc
import math
import os
...
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
...
```

#### 수정 후 (ultralytics_custom) - Lines 1-75
```python
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.
...
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
...
import torch
import torch.nn.functional as F  # ← F 추가 (distillation용)

from torch import distributed as dist
from torch import nn, optim

from ultralytics_custom.cfg import get_cfg, get_save_dir
...
```

**추가된 Import**:
1. `from model_compression.funcs4 import *`
   - 외부 압축 라이브러리 (프루닝 함수)
   - `yolov8_pruning()` 등이 포함되어 있을 것으로 추정

2. `from ultralytics_custom.models.yolo.model import YOLO`
   - Teacher 모델 로딩용

3. `from ultralytics_custom.utils.memory_usage_MH import *`
   - 메모리 사용량 측정 함수

4. `import torch.nn.functional as F`
   - KL Divergence, MSE Loss 등 distillation용

**주의**: `model_compression.funcs4`는 외부 패키지로 보입니다.

---

## 4. Trainer 수정 - Part 2: Loss 함수 추가

### 위치: Lines 67-75

#### 추가된 코드
```python
def distillation_loss(student_logits, teacher_logits, T=2.0):
    """
    Knowledge Distillation loss using KL Divergence

    Args:
        student_logits: Student model output (before softmax)
        teacher_logits: Teacher model output (before softmax)
        T: Temperature for softening probability distributions
    """
    s = student_logits / T
    t = teacher_logits / T
    return F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction='batchmean') * T * T


def MSE_loss(student_feat, teacher_feat):
    """Calculate MSE loss between student and teacher features"""
    return F.mse_loss(student_feat, teacher_feat)
```

**목적**:
- **Knowledge Distillation**: Teacher 모델의 지식을 Student에게 전달
- **Temperature Scaling**: T > 1로 soft probability 생성
- **Feature Matching**: 중간 feature map MSE loss

**배경 지식 - Knowledge Distillation**:
```
일반 학습:    Student → Ground Truth (hard label)
Distillation: Student → Teacher Output (soft label) + Ground Truth

장점:
- Teacher의 암묵적 지식 학습
- 압축 모델의 정확도 향상
- Generalization 개선
```

---

## 5. Trainer 수정 - Part 3: Teacher 모델 초기화

### 위치: `_setup_train()` 메서드 - Lines 264-272

#### 원본 (ultralytics_github)
```python
def _setup_train(self, world_size):
    """Builds dataloaders and optimizer on correct rank process."""

    # 바로 pretrain routine 시작
    self.run_callbacks("on_pretrain_routine_start")
    ckpt = self.setup_model()
    self.model = self.model.to(self.device)
    ...
```

#### 수정 후 (ultralytics_custom)
```python
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

    self.teacher_model.half()  # FP16으로 메모리 절약


    self.run_callbacks("on_pretrain_routine_start")
    ckpt = self.setup_model()
    self.model = self.model.to(self.device)
    self.set_model_attributes()
##############################################################
    ...
```

**핵심 변경사항**:

#### A. Teacher 모델 로딩
```python
# 옵션 1: 현재 모델 복사 (주석 처리됨)
self.teacher_model = deepcopy(self.model).eval().to(self.device)

# 옵션 2: 사전 학습된 큰 모델 사용 (실제 사용)
self.teacher_model = YOLO('/workspace/.../best.pt').to(self.device)
```

**선택 기준**:
- **옵션 1**: Pruning 전 모델 자체를 teacher로 (self-distillation)
- **옵션 2**: 더 큰 모델을 teacher로 (YOLOv8x → YOLOv8n 압축)

#### B. Teacher를 고정
```python
for p in self.teacher_model.parameters():
    p.requires_grad = False
```
- Gradient 계산 비활성화
- Teacher는 학습하지 않음 (추론만)

#### C. FP16 변환
```python
self.teacher_model.half()
```
- 메모리 사용량 절반으로 감소
- 추론 속도 향상

---

## 6. Trainer 수정 - Part 4: Training Loop 수정

### 위치: `_do_train()` 메서드 - Lines 429-468

Training loop의 핵심 forward/backward 부분에 압축 로직이 통합되었습니다.

#### 원본 (ultralytics_github)
```python
# Forward
with autocast(self.amp):
    self.loss, self.loss_items = self.model(batch)

    if RANK != -1:
        self.loss *= world_size
    self.tloss = (
        (self.tloss * i + self.loss_items) / (i + 1)
        if self.tloss is not None else self.loss_items
    )

# Backward
self.scaler.scale(self.loss).backward()

# Optimize
if ni - last_opt_step >= self.accumulate:
    self.optimizer_step()
    last_opt_step = ni
```

#### 수정 후 (ultralytics_custom) - Lines 429-492

```python
# Forward
with autocast(self.amp):
    #self.args.mem_usg = model_memory_usage_with_reducing(dummy_input, self.model, device = self.device)
    self.args.mem_usg = 100  # MB (하드코딩됨)

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

            # Memory constraint loss
            self.loss += total_kd_loss
            self.loss += max(0, self.args.mem_usg - device_condition_mem) * hyperparam

    if RANK != -1:
        self.loss *= world_size
    self.tloss = (
        (self.tloss * i + self.loss_items) / (i + 1)
        if self.tloss is not None else self.loss_items
    )

# Backward
self.scaler.scale(self.loss).backward()

# Optimize
if ni - last_opt_step >= self.accumulate:
    self.optimizer_step()
    ########################################################################
    ##################SM Insert ############################################
    if self.args.pruning_ratio == 0:
        continue
    else:
        yolov8_pruning(self.model.model, self.args.pruning_ratio, device=self.device)
    ########################################################################

    last_opt_step = ni
```

---

### 상세 분석

#### A. 메모리 사용량 측정 (Lines 429-433)

```python
#self.args.mem_usg = model_memory_usage_with_reducing(dummy_input, self.model, device = self.device)
self.args.mem_usg = 100  # 목표 메모리 (MB)

hyperparam = 0.5  # 메모리 loss 가중치
dummy_input = torch.zeros(1, 3, 640, 640).half().to(self.device)
device_condition_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
```

**목적**:
- 현재 GPU 메모리 사용량 추적
- 목표 메모리를 초과하면 loss에 패널티 추가

**메모리 constraint loss**:
```python
mem_loss = max(0, target_mem - current_mem) * weight
```
- `current_mem < target_mem`: loss = 0 (문제없음)
- `current_mem > target_mem`: loss > 0 (패널티)

#### B. Distillation Loss (Lines 436-458)

**1. Student Forward (distillation mode)**
```python
self.loss, self.loss_items, feats = self.model(batch, distillation=True)
```
- `distillation=True`: 중간 feature도 반환하도록 모델 수정 필요
- `feats`: Student의 중간 출력

**2. Teacher Forward (no gradient)**
```python
with torch.no_grad():
    teacher_out = self.teacher_model.model(batch['img'])
```
- `torch.no_grad()`: Teacher는 gradient 계산 안 함 (메모리 절약)

**3. Feature Extraction**
```python
if self.epoch > 0:  # 첫 epoch은 skip
    stu_result = feats[0]        # Student logits
    teacher_result = teacher_out[0]  # Teacher logits
```

**왜 첫 epoch skip?**
- 초기 모델이 불안정
- Teacher와 차이가 너무 커서 학습에 방해

**4. Classification Loss (KL Divergence)**
```python
s_cls = F.log_softmax(stu_result[:, 4:, :], dim=1)  # log(P(student))
t_cls = F.softmax(teacher_result[:, 4:, :], dim=1)  # P(teacher)
loss_cls = F.kl_div(s_cls, t_cls, reduction='batchmean')
```

**YOLO 출력 형식**:
```
result: [batch, channels, anchors]
  [:, 0:4, :]: bbox coordinates (x, y, w, h)
  [:, 4:, :]: class probabilities
```

**KL Divergence**:
```
D_KL(P||Q) = Σ P(x) log(P(x) / Q(x))

여기서:
- P: Teacher distribution (soft label)
- Q: Student distribution
```

**5. Bbox Regression Loss (MSE)**
```python
loss_bbox = F.mse_loss(stu_result[:, 0:4, :], teacher_result[:, 0:4, :])
```

**MSE Loss**:
```
MSE = (1/N) Σ (y_pred - y_true)^2
```

**6. Total KD Loss**
```python
total_kd_loss = (loss_cls + loss_bbox) * self.args.distill_ratio
```

- `distill_ratio`: Distillation loss의 가중치
- 총 loss = Task loss + (KD loss * distill_ratio) + Memory loss

**7. Memory Loss**
```python
self.loss += max(0, self.args.mem_usg - device_condition_mem) * hyperparam
```

#### C. Pruning 실행 (Lines 484-487)

**Optimizer step 직후 pruning**:
```python
if ni - last_opt_step >= self.accumulate:
    self.optimizer_step()
    ########################################################################
    ##################SM Insert ############################################
    if self.args.pruning_ratio == 0:
        continue
    else:
        yolov8_pruning(self.model.model, self.args.pruning_ratio, device=self.device)
    ########################################################################
```

**프루닝 시점**:
- **매 gradient update 후**
- Accumulated gradients 적용 후
- Weight update 완료 후

**왜 이 시점?**
1. 가중치가 최신 상태
2. 다음 forward 전에 불필요한 필터 제거
3. 즉시 메모리 효율 개선

**프루닝 플로우**:
```
Training iteration:
1. Forward pass
2. Backward pass
3. Accumulate gradients
4. Optimizer step (weight update)
5. Pruning ← 이 시점
6. Next iteration
```

**동적 프루닝 (Dynamic Pruning)**:
- 학습 중 지속적으로 프루닝
- 중요도가 낮아진 필터를 실시간으로 제거
- Fine-tuning과 pruning을 동시에 진행

---

## 7. 전체 Loss 구성

### Final Loss Function

```python
total_loss = task_loss + kd_loss + memory_loss

여기서:
- task_loss: 원래 YOLO loss (detection loss)
- kd_loss = (cls_loss + bbox_loss) * distill_ratio
  - cls_loss: KL divergence between teacher/student classifications
  - bbox_loss: MSE between teacher/student bboxes
- memory_loss = max(0, target_mem - current_mem) * hyperparam
```

### Loss 가중치 밸런싱

```python
# 예시 값들
task_loss ≈ 1.0
distill_ratio ≈ 0.5    # KD loss weight
hyperparam ≈ 0.5       # Memory loss weight

→ kd_loss ≈ 0.5
→ memory_loss ≈ 0.0~0.5 (상황에 따라)

total_loss ≈ 1.0 + 0.5 + 0.0~0.5 = 1.5~2.0
```

---

## 8. 코드 통합 패턴 요약

### 통합 체크리스트

#### ✅ 1. 설정 추가
- [ ] `cfg/default.yaml`에 새 파라미터 추가
- [ ] `cfg/__init__.py`에 파라미터 등록

#### ✅ 2. Import 추가
- [ ] 압축 함수 import
- [ ] Distillation용 F import
- [ ] 메모리 측정 함수 import

#### ✅ 3. 초기화 수정
- [ ] Teacher 모델 로딩
- [ ] Teacher를 eval mode로 고정
- [ ] FP16 변환 (선택)

#### ✅ 4. Training Loop 수정
- [ ] Student forward를 distillation mode로
- [ ] Teacher forward 추가 (no_grad)
- [ ] KD loss 계산
- [ ] Memory loss 계산
- [ ] Pruning 호출

#### ✅ 5. 압축 모듈 추가
- [ ] `compression.py` 생성
- [ ] `compression_src/` 디렉토리 생성
- [ ] Pruning/Reducing 함수 구현

---

## 9. 사용 예제

### YAML 설정 파일
```yaml
# config.yaml
task: detect
mode: train
model: yolov8n.pt
data: coco8.yaml
epochs: 100
batch: 16
imgsz: 640

# Compression settings
pruning_ratio: 0.3      # 30% pruning
mem_usg: 100.0          # Target 100MB
distill_ratio: 0.5      # 50% KD loss weight
```

### Python 코드
```python
from ultralytics_custom import YOLO

# 모델 로드
model = YOLO('yolov8n.pt')

# 학습 (자동으로 pruning + distillation)
model.train(
    data='coco8.yaml',
    epochs=100,
    pruning_ratio=0.3,
    distill_ratio=0.5,
    mem_usg=100.0
)

# 압축된 모델 저장
model.save('yolov8n_compressed.pt')
```

### 터미널 명령어
```bash
yolo train \
  model=yolov8n.pt \
  data=coco8.yaml \
  epochs=100 \
  pruning_ratio=0.3 \
  distill_ratio=0.5 \
  mem_usg=100.0
```

---

## 10. 주의사항 및 권장사항

### ⚠️ 주의사항

1. **Teacher 모델 경로**
   ```python
   # 하드코딩된 경로 수정 필요
   self.teacher_model = YOLO('/workspace/TW/YOLO/runs/detect/train_YOLOv8x/weights/best.pt')
   ```
   → 설정 파일이나 인자로 받도록 수정 권장

2. **외부 의존성**
   ```python
   from model_compression.funcs4 import *
   ```
   → `model_compression` 패키지가 필요 (별도 설치?)

3. **메모리 측정**
   ```python
   self.args.mem_usg = 100  # 하드코딩
   ```
   → 실제 측정 함수 활성화 고려

4. **Distillation mode**
   - 모델이 `distillation=True` 인자를 지원해야 함
   - `model.py`도 수정 필요

### 💡 권장사항

1. **점진적 압축**
   ```python
   # Epoch별로 pruning ratio 증가
   if epoch < 30:
       pruning_ratio = 0.0
   elif epoch < 60:
       pruning_ratio = 0.1
   else:
       pruning_ratio = 0.3
   ```

2. **Loss 가중치 조정**
   ```python
   # 학습 초반: task loss에 집중
   # 학습 후반: distillation loss 증가
   distill_ratio = min(0.5, epoch / 100)
   ```

3. **Validation 주기 증가**
   - Pruning으로 인한 불안정성
   - 자주 검증하여 성능 모니터링

---

## 11. 다음 단계

이 통합 가이드를 참고하여:

1. **자신의 코드에 적용**
   - 단계별로 천천히 통합
   - 각 단계마다 테스트

2. **하이퍼파라미터 튜닝**
   - `pruning_ratio`, `distill_ratio` 실험
   - 최적의 조합 찾기

3. **성능 평가**
   - 압축률 vs 정확도
   - 추론 속도 측정
   - 메모리 사용량 확인

4. **고급 기능 추가**
   - Layer-wise pruning
   - Quantization 결합
   - Progressive pruning

---

## 참고: 전체 플로우

```
┌─────────────────────────────────────────────────────────┐
│                    Training Start                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  1. Setup Phase (_setup_train)                          │
│     - Load teacher model                                │
│     - Freeze teacher parameters                         │
│     - Convert to FP16                                   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  2. Training Loop (_do_train)                           │
│     ┌───────────────────────────────────────────┐      │
│     │  For each batch:                          │      │
│     │                                            │      │
│     │  A. Forward Pass                          │      │
│     │     - Student: model(batch, distil=True)  │      │
│     │     - Teacher: teacher_model(batch)       │      │
│     │                                            │      │
│     │  B. Loss Calculation                      │      │
│     │     - Task loss (detection)               │      │
│     │     - KD loss (cls + bbox)                │      │
│     │     - Memory loss                         │      │
│     │                                            │      │
│     │  C. Backward Pass                         │      │
│     │     - Compute gradients                   │      │
│     │                                            │      │
│     │  D. Optimization                          │      │
│     │     - Update weights                      │      │
│     │     - **Pruning** ← 여기!                 │      │
│     │                                            │      │
│     └───────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  3. Validation & Checkpointing                          │
│     - Evaluate compressed model                         │
│     - Save best checkpoint                              │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  4. Final Reducing (after training)                     │
│     - yolov8_reducing()                                 │
│     - Create smaller model                              │
│     - Save compressed .pt file                          │
└─────────────────────────────────────────────────────────┘
```

---

이 가이드를 통해 기존 ultralytics 코드에 AI 모델 압축 기능을 정확히 어떻게 통합했는지 이해할 수 있습니다!
