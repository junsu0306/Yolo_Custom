# YOLOv8 Compression

YOLOv8 모델에 대한 **Knowledge Distillation + Structured Pruning** 기반 압축 구현입니다.

---

## 빠른 시작

**권장**: `yolo_utils` 모듈을 사용하면 간편하게 압축을 적용할 수 있습니다.

### 환경 설정

```bash
pip install ultralytics torch torchvision
```

### 압축 실행

```python
from yolo_utils import yolov8_pruning, yolov8_reducing
from ultralytics import YOLO
import torch

# 1. 모델 로드
model = YOLO('yolov8n.pt')

# 2. Pruning (30% 필터 제거)
yolov8_pruning(model.model.model, sparsity=0.3)

# 3. 학습 (선택)
# model.train(data='coco8.yaml', epochs=100)

# 4. Reducing (물리적 채널 제거)
reduced_model = YOLO('yolov8n.yaml')
yolov8_reducing(model.model.model, reduced_model.model.model)

# 5. 저장
torch.save(reduced_model.model.state_dict(), 'compressed.pt')
```

---

## 개요

| 기법 | 설명 |
|------|------|
| **Pruning** | L2-norm 기반 Structured Filter Pruning |
| **Distillation** | Teacher(YOLOv8x) → Student(YOLOv8n) 지식 전이 |
| **Memory Loss** | 메모리 사용량 기반 추가 손실 함수 |

### 모델 크기

```
YOLOv8n (nano)   →  3.2M params   ← Student
YOLOv8x (extra)  →  68.2M params  ← Teacher
```

---

## 파일 구조

```
ultralytics_custom/
├── engine/
│   ├── trainer.py           # KD 통합 학습 로직
│   ├── compression.py       # Pruning/Reducing 함수
│   └── compression_src/     # 범용 압축 모듈
│       ├── pruning/common.py
│       └── reducing/common.py
├── cfg/default.yaml         # 압축 파라미터 (pruning_ratio, mem_usg)
└── utils/memory_usage_MH.py # 메모리 측정
```

---

## 원본 Ultralytics와 통합

### Step 1: 환경 설정

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics && pip install -e .
```

### Step 2: 커스텀 파일 복사

```bash
cp ultralytics_custom/engine/trainer.py ultralytics/ultralytics/engine/
cp ultralytics_custom/engine/compression.py ultralytics/ultralytics/engine/
cp -r ultralytics_custom/engine/compression_src ultralytics/ultralytics/engine/
cp ultralytics_custom/cfg/default.yaml ultralytics/ultralytics/cfg/
```

### Step 3: 압축 학습 실행

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='coco8.yaml',
    epochs=100,
    pruning_ratio=0.3,  # 30% pruning
    mem_usg=100.0       # 목표 메모리 (MB)
)
```

---

## Knowledge Distillation

```
Teacher (YOLOv8x, 68.2M) ──┐
                          │ Knowledge Transfer
Student (YOLOv8n, 3.2M) ──┘

KD Loss = Classification Loss (KL-div) + BBox Loss (MSE)
```

### Loss 계산

```python
# Classification: KL Divergence
loss_cls = F.kl_div(
    F.log_softmax(student[:, 4:, :], dim=1),
    F.softmax(teacher[:, 4:, :], dim=1)
)

# Bounding Box: MSE
loss_bbox = F.mse_loss(student[:, 0:4, :], teacher[:, 0:4, :])

# Total KD Loss
kd_loss = (loss_cls + loss_bbox) * distill_ratio
```

---

## Training Flow

```
[1] Setup
    ├── Teacher 모델 로드 (frozen, FP16)
    └── Student 모델 로드

[2] Training Loop
    ├── Forward: Student & Teacher
    ├── Loss: Task + KD + Memory
    ├── Backward & Optimize
    └── Pruning (매 step)

[3] Post-training
    └── Reducing (물리적 채널 제거)
```

---

## 주의사항

1. **Teacher 모델**: KD 사용 시 사전 학습된 Teacher 필요 (YOLOv8x 권장)
2. **GPU 메모리**: Teacher + Student 동시 로드로 최소 8GB 권장
3. **Reducing**: 학습 완료 후 1회만 실행, Fine-tuning 권장

---

## 관련 파일

- [yolo_utils/](../yolo_utils/) - 간편 사용 유틸리티
- [메인 README](../README.md) - 전체 압축 과정 설명
