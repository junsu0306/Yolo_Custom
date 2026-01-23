# YOLO Compression (ultralytics_custom)

YOLOv8 모델에 대한 **Knowledge Distillation + Structured Pruning** 기반 Training-time 압축 구현입니다.

---

## 개요

### 핵심 기술

| 기법 | 설명 |
|------|------|
| **Pruning** | L2-norm 기반 One-shot Structured Filter Pruning |
| **Distillation** | Teacher(YOLOv8x) → Student(YOLOv8n) 지식 전이 |
| **Memory Loss** | 메모리 사용량 기반 추가 손실 함수 |

### 모델 크기 비교

```
YOLOv8n (nano)   →  3.2M params   ← Student (학습 대상)
YOLOv8s (small)  →  11.2M params
YOLOv8m (medium) →  25.9M params
YOLOv8l (large)  →  43.7M params
YOLOv8x (extra)  →  68.2M params  ← Teacher (고정)
```

---

## 수정 및 추가된 파일 목록

```
ultralytics_custom/
├── cfg/
│   ├── default.yaml                 # 압축 관련 파라미터 추가
│   │   └── pruning_ratio, mem_usg 추가
│   └── __init__.py                  # 파라미터 등록
│
├── engine/
│   ├── trainer.py                   # 핵심 학습 로직 수정
│   │   ├── [Lines 11-18]   Import 추가 (compression, distillation 관련)
│   │   ├── [Lines 67-75]   Loss 함수 정의 (distillation_loss, MSE_loss)
│   │   ├── [Lines 264-272] Teacher 모델 로드 및 Freeze
│   │   └── [Lines 429-492] Training loop 수정 (KD + Pruning)
│   │
│   ├── compression.py               # YOLO 전용 압축 함수
│   │   ├── yolov8_pruning()        # L2 norm 기반 필터 마스킹
│   │   └── yolov8_reducing()       # 0인 필터 물리적 제거
│   │
│   └── compression_src/             # 범용 압축 모듈
│       ├── pruning/common.py
│       │   ├── get_filter_pruning_idx()
│       │   ├── filter_pruning()
│       │   └── bn_pruning()
│       └── reducing/common.py
│           ├── get_survived_filter_idx()
│           ├── conv_reduce()
│           └── bn_reduce()
│
└── utils/
    └── memory_usage_MH.py           # GPU 메모리 측정 유틸리티
```

---

## Knowledge Distillation 구조

### Teacher-Student 구성

```
┌─────────────────────────────────────────────────────────┐
│  Teacher Model: YOLOv8x (68.2M params)                 │
│  - 사전 학습된 대형 모델                                 │
│  - requires_grad = False (학습 안함)                    │
│  - FP16 변환으로 메모리 절약                             │
└─────────────────────────────────────────────────────────┘
                           ↓ Knowledge Transfer
┌─────────────────────────────────────────────────────────┐
│  Student Model: YOLOv8n (3.2M params)                  │
│  - 경량 모델                                            │
│  - Teacher 출력을 모방하도록 학습                        │
│  - Pruning 적용 대상                                    │
└─────────────────────────────────────────────────────────┘
```

### Distillation Loss 구성

```python
# 1. Classification KD Loss (KL Divergence)
s_cls = F.log_softmax(student_result[:, 4:, :], dim=1)
t_cls = F.softmax(teacher_result[:, 4:, :], dim=1)
loss_cls = F.kl_div(s_cls, t_cls, reduction='batchmean')

# 2. Bounding Box Regression Loss (MSE)
loss_bbox = F.mse_loss(student_result[:, 0:4, :], teacher_result[:, 0:4, :])

# 3. Total KD Loss
total_kd_loss = (loss_cls + loss_bbox) * distill_ratio
```

---

## 기능별 구현 코드

### 1. Structured Pruning

**목적**: Conv 필터의 중요도를 L2 norm으로 측정하여 불필요한 필터 제거

**핵심 파일**: `compression.py`, `compression_src/pruning/common.py`

```python
def yolov8_pruning(model, sparsity, device='cuda'):
    """YOLOv8 모델의 Conv, C2f, SPPF, Detect 블록에 pruning 적용"""
    for idx, block in enumerate(model.model):
        if hasattr(block, 'conv'):
            # L2 norm 계산 및 pruning index 선정
            pruning_idx = get_filter_pruning_idx(block.conv, sparsity)

            # 필터 마스킹 (weight = 0)
            filter_pruning(block.conv, pruning_idx)
            bn_pruning(block.bn, pruning_idx)
```

**Pruning 메커니즘**:
```python
def filter_pruning(conv, pruning_idx):
    """Conv weight를 0으로 마스킹"""
    conv.weight.data[pruning_idx, :, :, :] = 0.0

def bn_pruning(bn, pruning_idx):
    """BatchNorm 파라미터 마스킹"""
    bn.weight.data[pruning_idx] = 0.0
    bn.bias.data[pruning_idx] = 0.0
    bn.running_mean.data[pruning_idx] = 0.0
    bn.running_var.data[pruning_idx] = 1.0  # var=1로 설정 (나눗셈 방지)
```

**특징**:
- L2 Norm 기반 필터 선택 (작은 값 = 덜 중요)
- Dynamic Pruning (매 optimizer step 후 실행)
- BN 파라미터 동시 마스킹

### 2. Channel Reduction

**목적**: 마스킹된(0인) 필터를 물리적으로 제거하여 실제 모델 크기 감소

**핵심 파일**: `compression.py`, `compression_src/reducing/common.py`

```python
def yolov8_reducing(model, reduced_model):
    """0인 필터를 물리적으로 제거한 축소 모델 생성"""
    for idx, (block, reduced_block) in enumerate(zip(model.model, reduced_model.model)):
        # 살아남은 필터 인덱스 찾기
        survived_idx = get_survived_filter_idx(block.conv)

        # Conv/BN 레이어 축소
        conv_reduce(block.conv, reduced_block.conv, survived_idx)
        bn_reduce(block.bn, reduced_block.bn, survived_idx)
```

**특징**:
- Skip Connection 추적 (Concat 레이어 처리)
- Detect 헤드의 복잡한 구조 처리
- C2f 블록의 Bottleneck 처리

### 3. Memory-aware Loss

**목적**: GPU 메모리 사용량을 loss에 반영하여 메모리 효율적인 모델 학습

```python
# 현재 메모리 사용량 측정
device_condition_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB

# 목표 메모리 초과 시 패널티 부여
memory_loss = max(0, target_mem - device_condition_mem) * hyperparam
total_loss += memory_loss
```

---

## 전체 Training Flow

```
┌─────────────────────────────────────────────────────────┐
│                    Training Start                        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  1. Setup Phase (_setup_train)                          │
│     ┌─────────────────────────────────────────┐        │
│     │  A. Teacher 모델 로드                    │        │
│     │     - YOLO('yolov8x/best.pt')           │        │
│     │     - requires_grad = False             │        │
│     │     - model.half() (FP16)               │        │
│     │                                          │        │
│     │  B. Student 모델 로드                    │        │
│     │     - YOLO('yolov8n.pt')                │        │
│     │     - 학습 대상                          │        │
│     └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  2. Training Loop (_do_train)                           │
│     ┌─────────────────────────────────────────┐        │
│     │  For each batch:                         │        │
│     │                                          │        │
│     │  A. Forward Pass                         │        │
│     │     - Student: model(batch, distil=True) │        │
│     │     - Teacher: teacher_model(batch)      │        │
│     │                                          │        │
│     │  B. Loss Calculation                     │        │
│     │     - Task Loss (Detection)              │        │
│     │     - KD Loss (KL-div + MSE)             │        │
│     │     - Memory Loss                        │        │
│     │                                          │        │
│     │  C. Backward & Optimize                  │        │
│     │     - scaler.scale(loss).backward()      │        │
│     │     - optimizer.step()                   │        │
│     │                                          │        │
│     │  D. Pruning (매 step 후)                 │        │
│     │     - yolov8_pruning(model, ratio)       │        │
│     └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  3. Post-training Reducing (선택)                       │
│     - yolov8_reducing(model, reduced_model)            │
│     - 물리적으로 0인 필터 제거                           │
│     - compressed.pt 저장                                │
└─────────────────────────────────────────────────────────┘
```

---

## 상세 Training Flow

```
[1] 설정 파일 로드
    ├─ cfg/default.yaml
    ├─ pruning_ratio: 0.3 (30% 필터 제거)
    ├─ distill_ratio: 0.5 (KD loss 가중치)
    └─ mem_usg: 100.0 (목표 메모리 MB)

[2] Trainer 초기화 (_setup_train)
    ├─ Teacher 모델 로드
    │   ├─ YOLO('yolov8x/best.pt').to(device)
    │   ├─ requires_grad = False
    │   └─ model.half()
    │
    └─ Student 모델 로드
        ├─ setup_model()
        └─ model.to(device)

[3] Training Loop
    └─ For epoch in range(epochs):
        └─ For batch in train_loader:

            [3-1] Forward Pass
                ├─ Student Forward
                │   └─ loss, loss_items, feats = model(batch, distillation=True)
                │
                └─ Teacher Forward (no_grad)
                    └─ teacher_out = teacher_model.model(batch['img'])

            [3-2] Loss 계산
                ├─ Classification KD Loss
                │   ├─ s_cls = F.log_softmax(stu[:, 4:, :])
                │   ├─ t_cls = F.softmax(tea[:, 4:, :])
                │   └─ loss_cls = F.kl_div(s_cls, t_cls)
                │
                ├─ Bbox Regression Loss
                │   └─ loss_bbox = F.mse_loss(stu[:, 0:4], tea[:, 0:4])
                │
                ├─ Total KD Loss
                │   └─ kd_loss = (loss_cls + loss_bbox) * distill_ratio
                │
                └─ Memory Loss
                    └─ mem_loss = max(0, target - current) * 0.5

            [3-3] Backward & Optimize
                ├─ total_loss = task_loss + kd_loss + mem_loss
                ├─ scaler.scale(total_loss).backward()
                └─ optimizer.step()

            [3-4] Pruning (if pruning_ratio > 0)
                └─ yolov8_pruning(model, sparsity)
                    ├─ Conv 블록: filter_pruning + bn_pruning
                    ├─ C2f 블록: cv2에 대해 동일 작업
                    ├─ SPPF 블록: cv2에 대해 동일 작업
                    └─ Detect 블록: cv2, cv3에 대해 동일 작업

[4] Training 종료 후 Reducing (선택)
    └─ yolov8_reducing(model, reduced_model)
        ├─ survived_idx = get_survived_filter_idx(layer)
        ├─ conv_reduce(layer, reduced_layer, survived_idx)
        ├─ bn_reduce(bn, reduced_bn, survived_idx)
        └─ Concatenation 레이어 특별 처리 (11, 14, 17, 20)

[5] 모델 저장
    └─ reduced_model.save('compressed.pt')
```

---

## 설정 파라미터

### cfg/default.yaml

```yaml
# 압축 관련 파라미터
pruning_ratio: 0.3    # 30% 필터 제거
mem_usg: 100.0        # 목표 메모리 (MB)

# 기존 학습 파라미터
model: yolov8n.pt     # Student 모델
epochs: 100
batch: 16
imgsz: 640
```

---

## 실행 방법

### 1. 압축 학습 실행

```bash
cd ultralytics_custom
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(
    data='coco8.yaml',
    epochs=100,
    pruning_ratio=0.3,
    mem_usg=100.0
)
"
```

### 2. 학습 후 Reducing

```python
from engine.compression import yolov8_reducing

# Pruned 모델 로드
model = YOLO('runs/detect/train/weights/best.pt')

# Reducing 실행
reduced_model = yolov8_reducing(model.model, create_reduced_model())

# 저장
torch.save(reduced_model.state_dict(), 'compressed_yolov8n.pt')
```

---

## 압축 결과 예상

| 항목 | Original | Compressed | 감소율 |
|------|----------|------------|--------|
| Parameters | 3.2M | 2.0-2.2M | 30-35% |
| Model Size | 6.3 MB | 4.0-4.5 MB | 30-35% |
| Memory (Inference) | ~150 MB | ~100 MB | 30-35% |
| mAP50 | 37.3 | 35-36 | 3-5% 하락 |

---

## YOLO vs CenterPose 압축 비교

| 항목 | YOLO Compression | CenterPose Compression |
|------|------------------|------------------------|
| **백본** | CSPDarknet (YOLOv8) | DLA-34 |
| **타겟 모듈** | Conv, C2f, SPPF, Detect | BasicBlock |
| **Pruning 방식** | Dynamic (매 step) | One-shot |
| **Distillation** | Yes (YOLOv8x → v8n) | No |
| **Training 필요** | Yes | No |
| **복잡도** | High | Low |
| **성능 유지** | 더 좋음 (KD 효과) | Fine-tuning 필요 |

---

## 실제 실행 가이드 (Step-by-Step)

이 섹션에서는 **처음부터 끝까지** YOLO 모델 압축을 진행하고 실제로 사용하는 전체 과정을 설명합니다.

### 사전 요구사항

```bash
# Python 3.8+ 및 PyTorch 1.10+ 필요
pip install torch torchvision
pip install opencv-python numpy pillow
```

### Step 1: 원본 Ultralytics 레포지토리 클론

```bash
# Ultralytics 원본 레포지토리 클론
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics

# 의존성 설치
pip install -e .
```

### Step 2: Custom 압축 파일 적용

```bash
# Custom 파일들을 Ultralytics에 복사
# (Yolo_Custom 프로젝트 루트에서 실행)

# 방법 1: 전체 ultralytics_custom 폴더 사용 (권장)
# ultralytics_custom을 별도로 사용하고 import 경로만 수정

# 방법 2: 개별 파일 복사
cp ultralytics_custom/engine/trainer.py ultralytics/ultralytics/engine/
cp ultralytics_custom/engine/compression.py ultralytics/ultralytics/engine/
cp -r ultralytics_custom/engine/compression_src ultralytics/ultralytics/engine/
cp ultralytics_custom/cfg/default.yaml ultralytics/ultralytics/cfg/
cp ultralytics_custom/utils/memory_usage_MH.py ultralytics/ultralytics/utils/
```

### Step 3: 데이터셋 준비

```bash
# COCO 데이터셋 다운로드 (또는 커스텀 데이터셋)
# Ultralytics는 자동으로 coco8.yaml 샘플 데이터를 다운로드

# 커스텀 데이터셋 사용 시 data.yaml 파일 생성
# data.yaml 예시:
# path: /path/to/dataset
# train: images/train
# val: images/val
# names:
#   0: class1
#   1: class2
```

### Step 4: 압축 학습 실행 (Pruning + Knowledge Distillation)

```python
# train_compressed.py
from ultralytics import YOLO

# Student 모델 로드
model = YOLO('yolov8n.pt')

# 압축 학습 실행
# - pruning_ratio: 제거할 필터 비율 (0.3 = 30%)
# - mem_usg: 목표 메모리 사용량 (MB)
results = model.train(
    data='coco8.yaml',      # 데이터셋
    epochs=100,             # 학습 에폭
    imgsz=640,              # 이미지 크기
    batch=16,               # 배치 크기
    pruning_ratio=0.3,      # 30% pruning
    mem_usg=100.0,          # 메모리 제약 (MB)
    device=0                # GPU 번호
)
```

**실행:**
```bash
python train_compressed.py
```

**또는 CLI로 실행:**
```bash
yolo detect train \
    model=yolov8n.pt \
    data=coco8.yaml \
    epochs=100 \
    pruning_ratio=0.3 \
    mem_usg=100.0
```

### Step 5: 학습 후 Reducing (물리적 채널 제거)

학습이 완료되면 Pruning된 모델(0으로 마스킹된 필터)을 물리적으로 축소합니다.

```python
# reduce_model.py
import torch
from ultralytics import YOLO
from ultralytics.engine.compression import yolov8_pruning, yolov8_reducing

# 1. 학습된 모델 로드
model = YOLO('runs/detect/train/weights/best.pt')

# 2. 새로운 빈 모델 생성 (reducing 대상)
reduced_model = YOLO('yolov8n.yaml')  # 구조만 로드

# 3. Reducing 실행
yolov8_reducing(
    model=model.model.model,           # 원본 (pruned) 모델
    reduced_model=reduced_model.model.model  # 축소될 모델
)

# 4. 저장
torch.save(reduced_model.model.state_dict(), 'compressed_yolov8n.pt')
print("Compressed model saved!")

# 5. 파라미터 수 비교
original_params = sum(p.numel() for p in model.model.parameters())
reduced_params = sum(p.numel() for p in reduced_model.model.parameters())
print(f"Original: {original_params:,} params")
print(f"Reduced:  {reduced_params:,} params")
print(f"Reduction: {(1 - reduced_params/original_params)*100:.1f}%")
```

### Step 6: 압축된 모델로 추론 실행

```python
# inference.py
from ultralytics import YOLO

# 압축된 모델 로드
model = YOLO('runs/detect/train/weights/best.pt')

# 이미지 추론
results = model.predict(
    source='path/to/image.jpg',
    save=True,
    conf=0.25
)

# 결과 확인
for result in results:
    print(result.boxes)  # 바운딩 박스
    print(result.names)  # 클래스 이름
```

**CLI로 추론:**
```bash
# 이미지
yolo detect predict model=runs/detect/train/weights/best.pt source=image.jpg

# 비디오
yolo detect predict model=runs/detect/train/weights/best.pt source=video.mp4

# 웹캠
yolo detect predict model=runs/detect/train/weights/best.pt source=0

# 폴더 내 모든 이미지
yolo detect predict model=runs/detect/train/weights/best.pt source=images/
```

### Step 7: 모델 내보내기 (Export)

```python
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

# ONNX 내보내기
model.export(format='onnx')

# TensorRT 내보내기 (NVIDIA GPU)
model.export(format='engine')

# CoreML 내보내기 (Apple)
model.export(format='coreml')

# TFLite 내보내기 (모바일)
model.export(format='tflite')
```

### Step 8: 성능 평가

```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('runs/detect/train/weights/best.pt')

# Validation 실행
metrics = model.val(data='coco8.yaml')

print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.p:.3f}")
print(f"Recall: {metrics.box.r:.3f}")
```

---

## 전체 실행 흐름 요약

```
[1] 환경 준비
    └── Ultralytics 원본 클론 + pip install -e .
            ↓
[2] Custom 파일 적용
    └── engine/trainer.py, compression.py, cfg/default.yaml 복사
            ↓
[3] 데이터셋 준비
    └── COCO 또는 커스텀 데이터셋
            ↓
[4] 압축 학습 실행
    └── model.train(pruning_ratio=0.3, mem_usg=100.0)
    └── 출력: runs/detect/train/weights/best.pt
            ↓
[5] (선택) Reducing 실행
    └── yolov8_reducing(model, reduced_model)
    └── 출력: compressed_yolov8n.pt
            ↓
[6] 추론 실행
    └── model.predict(source='image.jpg')
            ↓
[7] 모델 내보내기
    └── model.export(format='onnx')
            ↓
[8] 성능 평가
    └── model.val(data='coco8.yaml')
```

---

## 주요 파라미터 설명

### cfg/default.yaml 압축 파라미터

```yaml
pruning_ratio: 0.3    # Pruning 비율 (0.0 = 비활성화)
                      # 0.3 = 30% 필터 제거
                      # 권장: 0.2 ~ 0.5

mem_usg: 100.0        # 목표 메모리 사용량 (MB)
                      # 0.0 = 메모리 제약 비활성화
                      # 초과 시 Memory Loss 추가
```

### Knowledge Distillation 설정

현재 `trainer.py`에서 하드코딩되어 있습니다:
- **Teacher**: `yolov8x/best.pt` (사전 학습 필요)
- **Student**: 학습 대상 모델

Teacher 모델 경로 수정 시 `trainer.py` Line 264-272 참조:
```python
# Teacher 모델 로드 예시
self.teacher_model = YOLO('path/to/teacher/best.pt')
for param in self.teacher_model.model.parameters():
    param.requires_grad = False
self.teacher_model.model.half()  # FP16 변환
```

---

## 주의사항

### 1. Teacher 모델 준비
- Knowledge Distillation 사용 시 Teacher 모델 사전 학습 필요
- YOLOv8x 권장 (더 큰 모델 = 더 좋은 Teacher)

### 2. GPU 메모리
- Teacher + Student 동시 로드로 메모리 사용량 증가
- 최소 8GB GPU 메모리 권장
- 메모리 부족 시 `batch` 크기 줄이기

### 3. 학습 시간
- Pruning + Distillation으로 일반 학습 대비 시간 증가
- 매 step마다 Teacher forward pass 추가

### 4. Reducing 주의점
- Reducing은 학습 완료 후 1회만 실행
- Concatenation 레이어 (indices: 11, 14, 17, 20) 특별 처리 필요
- Reducing 후 추가 Fine-tuning 권장

### 5. 성능 저하 대응
- pruning_ratio를 낮춰서 (0.2~0.3) 재시도
- 학습 에폭 증가
- Learning rate 조정
