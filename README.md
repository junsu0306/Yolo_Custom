# On-Device AI Model Compression

**IITP (ETRI Project) - On_Device_AI**

딥러닝 모델의 온디바이스 배포를 위한 **Structured Pruning** 및 **Knowledge Distillation** 기반 모델 압축 프로젝트입니다.

---

## 레포지토리 구조

```
On_Device_AI/
├── notebooks/                   # Colab 데모 노트북
│   ├── yolo_compression_demo.ipynb
│   └── centerpose_compression_demo.ipynb
│
├── yolo_utils/                  # YOLOv8 압축 핵심 코드
│   ├── pruning.py               # L2-norm Pruning
│   ├── reducing.py              # Channel Reducing
│   ├── kd.py                    # Knowledge Distillation
│   └── memory.py                # Memory Measurement
│
├── centerpose_utils/            # CenterPose 압축 핵심 코드
│   ├── pruning.py               # BasicBlock Pruning
│   ├── reducing.py              # Channel Reducing
│   └── memory.py                # Memory Measurement
│
├── ultralytics_custom/          # YOLO 원본 통합용 (trainer 등)
├── CenterPose src custom/       # CenterPose 원본 통합용
└── analysis/                    # 기술 분석 문서
```

---

## 핵심 압축 기법

본 프로젝트에서 사용하는 모델 압축의 핵심 과정입니다.

### 1. L2-Norm 기반 Structured Pruning

**원리**: 각 Conv 필터의 L2 norm을 계산하여, norm이 작은 (덜 중요한) 필터를 제거합니다.

```python
# yolo_utils/pruning.py, centerpose_utils/pruning.py

def get_filter_pruning_idx(layer, sparsity):
    """L2 norm 기준으로 pruning할 필터 인덱스를 계산"""
    with torch.no_grad():
        weight = layer.weight
        num_filters = weight.shape[0]
        num_pruning_filters = int(num_filters * sparsity)

        # 각 필터의 L2 norm 계산 (필터를 flatten 후 norm)
        filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)

        # 가장 작은 norm을 가진 필터 선택
        _, pruning_idx = torch.topk(filter_norms, num_pruning_filters, largest=False)

    return pruning_idx


def filter_pruning(layer, pruning_idx):
    """선택된 필터를 0으로 마스킹"""
    with torch.no_grad():
        layer.weight[pruning_idx, :, :, :] = 0.0


def bn_pruning(layer, pruning_idx):
    """BatchNorm의 해당 채널도 마스킹"""
    with torch.no_grad():
        layer.weight[pruning_idx] = 0.0
        layer.bias[pruning_idx] = 0.0
        layer.running_mean[pruning_idx] = 0.0
        layer.running_var[pruning_idx] = 1.0  # var=1로 설정하여 나눗셈 오류 방지
```

**Pruning 과정 시각화**:
```
Original Conv Filter (32 filters)
┌─────────────────────────────────────────┐
│ F1  F2  F3  F4  F5  ... F30 F31 F32     │  ← 모든 필터
│ 0.8 0.1 0.9 0.2 0.7 ... 0.3 0.6 0.05   │  ← L2 norm
└─────────────────────────────────────────┘
                    ↓ sparsity=0.3 (30% 제거)
┌─────────────────────────────────────────┐
│ F1  [0] F3  [0] F5  ... [0] F31 [0]     │  ← 작은 norm 필터 = 0
│ 0.8  0  0.9  0  0.7 ...  0  0.6  0      │
└─────────────────────────────────────────┘
```

---

### 2. Channel Reducing (물리적 채널 제거)

**원리**: Pruning 후 0으로 마스킹된 필터를 실제로 제거하여 모델 크기를 줄입니다.

```python
# yolo_utils/reducing.py, centerpose_utils/reducing.py

def get_survived_filter_idx(layer):
    """살아남은 필터 인덱스를 반환 (norm != 0)"""
    weight = layer.weight
    num_filters = weight.shape[0]
    filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)
    survived_filter_idx = torch.where(filter_norms != 0)[0]
    return survived_filter_idx


def conv_reduce(layer, reduced_layer, survived_out_idx, survived_in_idx):
    """Conv 레이어를 물리적으로 축소"""
    # 새로운 채널 수 설정
    reduced_layer.in_channels = len(survived_in_idx)
    reduced_layer.out_channels = len(survived_out_idx)

    # 새로운 weight 텐서 생성
    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(
            len(survived_out_idx),
            len(survived_in_idx),
            layer.kernel_size[0],
            layer.kernel_size[1],
        ),
        requires_grad=True,
    )

    # 살아남은 가중치만 복사
    with torch.no_grad():
        reduced_layer.weight.copy_(
            layer.weight[survived_out_idx, :, :, :][:, survived_in_idx, :, :]
        )
```

**Reducing 과정 시각화**:
```
Pruned Model (32 filters, 10개가 0)
┌─────────────────────────────────────────┐
│ F1  [0] F3  [0] F5  F6  [0] F8  ...     │  Conv: [32, 64, 3, 3]
└─────────────────────────────────────────┘
                    ↓ Reducing
┌─────────────────────────────────────────┐
│ F1  F3  F5  F6  F8  ...                 │  Conv: [22, 64, 3, 3]
└─────────────────────────────────────────┘

파라미터 감소: 32×64×3×3 → 22×64×3×3 (31% 감소)
```

---

### 3. Knowledge Distillation (YOLO 전용)

**원리**: 큰 Teacher 모델의 출력을 작은 Student 모델이 모방하도록 학습합니다.

```python
# yolo_utils/kd.py

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """KL Divergence 기반 Knowledge Distillation loss"""
    # Temperature로 softmax를 smooth하게 만듦
    s = student_logits / temperature
    t = teacher_logits / temperature

    return F.kl_div(
        F.log_softmax(s, dim=-1),
        F.softmax(t, dim=-1),
        reduction='batchmean'
    ) * temperature * temperature


def compute_kd_loss(student_output, teacher_output, distill_ratio=0.5):
    """YOLOv8 전용 KD loss (Classification + BBox)"""
    # Classification distillation (채널 4 이후 = class logits)
    student_cls = student_output[:, 4:, :]
    teacher_cls = teacher_output[:, 4:, :]

    t_cls = F.softmax(teacher_cls, dim=1)
    s_cls = F.log_softmax(student_cls, dim=1)
    loss_cls = F.kl_div(s_cls, t_cls, reduction='batchmean')

    # Bounding box regression distillation (채널 0-4 = bbox)
    loss_bbox = F.mse_loss(student_output[:, 0:4, :], teacher_output[:, 0:4, :])

    return (loss_cls + loss_bbox) * distill_ratio
```

**KD 구조**:
```
┌─────────────────────────────────────────┐
│  Teacher: YOLOv8x (68.2M params)        │
│  - 사전 학습된 대형 모델                  │
│  - Frozen (학습 안함)                    │
└─────────────────────────────────────────┘
                    ↓ Soft Labels
┌─────────────────────────────────────────┐
│  Student: YOLOv8n (3.2M params)         │
│  - 경량 모델 (학습 대상)                  │
│  - Teacher 출력을 모방                   │
└─────────────────────────────────────────┘

Total Loss = Task Loss + KD Loss + Memory Loss
```

---

### 4. Memory-Aware Training

**원리**: 학습 중 메모리 사용량을 측정하고, 목표치 초과 시 페널티를 부여합니다.

```python
# yolo_utils/memory.py, centerpose_utils/memory.py

def measure_memory(x, layers, device):
    """레이어 실행 시 GPU 메모리 사용량 측정"""
    # CPU로 이동 후 캐시 비우기
    for layer in layers:
        layer.cpu()
    torch.cuda.empty_cache()

    before_memory = torch.cuda.memory_allocated(device) / 1024**2  # MB

    # GPU에서 실행
    with torch.no_grad():
        x = x.to(device)
        for layer in layers:
            layer.to(device)
            x = layer(x)

    after_memory = torch.cuda.memory_allocated(device) / 1024**2

    return x, after_memory - before_memory


def custom_memory_loss_function(memory, hyperparam, target_memory):
    """메모리 제약 기반 손실 함수"""
    memory_diff = memory - target_memory
    memory_loss = max(0, memory_diff)  # 목표 초과분만 페널티

    return torch.tensor(hyperparam * memory_loss, requires_grad=True)
```

---

## 전체 압축 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                    1. Model Loading                          │
│         사전 학습된 모델 로드 (YOLOv8n, DLA-34 등)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    2. Pruning                                │
│    ┌─────────────────────────────────────────────┐          │
│    │  A. 각 Conv 필터의 L2 norm 계산              │          │
│    │  B. 하위 sparsity% 필터 선택                 │          │
│    │  C. 선택된 필터를 0으로 마스킹               │          │
│    │  D. 해당 BatchNorm 채널도 마스킹             │          │
│    └─────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              3. Training (선택, YOLO의 경우)                 │
│    ┌─────────────────────────────────────────────┐          │
│    │  A. Forward: Student & Teacher              │          │
│    │  B. Loss = Task + KD + Memory               │          │
│    │  C. Backward & Optimize                     │          │
│    │  D. 매 step Pruning 재적용                   │          │
│    └─────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    4. Reducing                               │
│    ┌─────────────────────────────────────────────┐          │
│    │  A. 살아남은 필터 인덱스 추출 (norm != 0)    │          │
│    │  B. 새로운 축소된 레이어 생성                │          │
│    │  C. 살아남은 가중치만 복사                   │          │
│    │  D. 연결된 레이어 채널도 조정                │          │
│    └─────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    5. Save & Deploy                          │
│              압축된 모델 저장 및 배포                         │
│         Original: 3.2M → Compressed: 2.1M (35% 감소)         │
└─────────────────────────────────────────────────────────────┘
```

---

## 빠른 시작

### YOLOv8 압축

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/yolo_compression_demo.ipynb)

```python
from yolo_utils import yolov8_pruning, yolov8_reducing
from ultralytics import YOLO
import torch

# 1. 모델 로드
model = YOLO('yolov8n.pt')
print(f"Original: {sum(p.numel() for p in model.model.parameters()):,} params")

# 2. Pruning (30% 필터 제거)
yolov8_pruning(model.model.model, sparsity=0.3)

# 3. Reducing (물리적 채널 제거)
reduced_model = YOLO('yolov8n.yaml')
yolov8_reducing(model.model.model, reduced_model.model.model)
print(f"Reduced: {sum(p.numel() for p in reduced_model.model.parameters()):,} params")

# 4. 저장
torch.save(reduced_model.model.state_dict(), 'compressed_yolov8n.pt')
```

### CenterPose 압축

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/centerpose_compression_demo.ipynb)

```python
from centerpose_utils import dlasg_blockwise_pruning, reduce_pruned_model
import torch

# 1. 모델 로드
model = load_model(...)  # CenterPose 환경 필요

# 2. Pruning (30% 필터 제거)
pruned_model = dlasg_blockwise_pruning(model, sparsity=0.3)

# 3. Reducing (물리적 채널 제거)
reduced_model = reduce_pruned_model(pruned_model)

# 4. 저장
torch.save(reduced_model, 'compressed_centerpose.pth')
```

---

