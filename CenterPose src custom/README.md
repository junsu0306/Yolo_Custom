# CenterPose Compression

DLA-34 기반 CenterPose 6D Pose Estimation 모델에 대한 **Structured Pruning** 구현입니다.

---

## 빠른 시작

**권장**: `centerpose_utils` 모듈을 사용하면 간편하게 압축을 적용할 수 있습니다.

### 환경 설정

```bash
# CenterPose 원본 레포 클론
git clone https://github.com/NVlabs/CenterPose.git
cd CenterPose && pip install -r requirements.txt

# DCNv2 빌드 (필요한 경우)
cd src/lib/models/networks/DCNv2
python setup.py build develop
```

### 압축 실행

```python
from centerpose_utils import dlasg_blockwise_pruning, reduce_pruned_model
import torch

# CenterPose 환경 필요
from lib.models.model import create_model, load_model

# 1. 모델 로드
model = create_model('dla_34', heads, head_conv)
model = load_model(model, 'pretrained.pth')

# 2. Pruning (30% 필터 제거)
pruned_model = dlasg_blockwise_pruning(model, sparsity=0.3)

# 3. Reducing (물리적 채널 제거)
reduced_model = reduce_pruned_model(pruned_model)

# 4. 저장
torch.save(reduced_model, 'compressed.pth')
```

---

## 개요

| 기법 | 설명 |
|------|------|
| **Pruning** | L2-norm 기반 BasicBlock 단위 Structured Pruning |
| **Reducing** | 물리적 채널 제거를 통한 실제 모델 경량화 |
| **Memory Loss** | 학습 중 메모리 사용량 기반 최적화 |

### 모델 구조

```
CenterPose (DLA-34)
├── Base Network: DLA-34 (~20M params)
│   └── BasicBlock × 여러 개 (Pruning 대상)
├── Upsampling: DLA-Up, IDA-Up
└── Detection Heads: hm, wh, reg, hps, scale
```

---

## 파일 구조

```
CenterPose src custom/
├── pruning.py               # Pruning 실행 스크립트
├── reducing.py              # Reducing 실행 스크립트
├── reduced_demo.py          # 압축 모델 Demo
└── lib/pruning/
    ├── dlasg_pruning.py     # Pruning 핵심 구현
    └── memory_usage.py      # 메모리 측정
```

---

## 원본 CenterPose와 통합

### Step 1: 환경 설정

```bash
git clone https://github.com/NVlabs/CenterPose.git
cd CenterPose && pip install -r requirements.txt

# DCNv2 빌드 (필요한 경우)
cd src/lib/models/networks/DCNv2
python setup.py build develop
```

### Step 2: 커스텀 파일 복사

```bash
cp "CenterPose src custom/lib/pruning/dlasg_pruning.py" CenterPose/src/lib/pruning/
cp "CenterPose src custom/lib/pruning/memory_usage.py" CenterPose/src/lib/pruning/
cp "CenterPose src custom/pruning.py" CenterPose/src/
cp "CenterPose src custom/reducing.py" CenterPose/src/
```

### Step 3: Pruning 실행

```bash
cd CenterPose/src

# pruning.py 수정 후 실행
python pruning.py
```

### Step 4: Reducing 실행

```bash
python reducing.py
```

---

## Compression Flow

```
[1] Model Load
    └── DLA-34 사전 학습 모델 로드

[2] Pruning (dlasg_blockwise_pruning)
    ├── BasicBlock 순회
    ├── Conv1 필터 L2 norm 계산
    ├── 하위 sparsity% 필터 선택
    └── 필터 0으로 마스킹

[3] Reducing (reduce_pruned_model)
    ├── 0이 아닌 필터 인덱스 추출
    ├── Conv1 출력 채널 축소
    ├── Conv2 입력 채널 축소
    └── BatchNorm 채널 축소

[4] Save
    └── 압축된 모델 저장
```

---

## Pruning 상세

### BasicBlock 구조

```
Input → Conv1 → BN1 → ReLU → Conv2 → BN2 → Output
         ↑ Pruning 대상
```

### L2-norm 기반 필터 선택

```python
# 각 필터의 L2 norm 계산
filter_norms = torch.norm(conv.weight.view(num_filters, -1), dim=1)

# 가장 작은 norm을 가진 필터 선택
_, pruning_idx = torch.topk(filter_norms, num_pruning, largest=False)

# 선택된 필터를 0으로 마스킹
conv.weight[pruning_idx] = 0
bn.weight[pruning_idx] = 0
```

---

## 주요 함수

```python
from centerpose_utils import (
    # Pruning
    dlasg_blockwise_pruning,  # BasicBlock 단위 pruning
    filter_pruning,           # Conv 필터 마스킹
    bn_pruning,               # BatchNorm 마스킹
    get_filter_norms,         # 필터 L2 norm 계산

    # Reducing
    reduce_pruned_model,      # 물리적 채널 제거
    get_survived_filter_idx,  # 살아남은 필터 인덱스

    # Memory
    measure_model_memory,     # 메모리 측정
    custom_memory_loss_function,  # 메모리 손실 함수
)
```

---

## 실행 예시

### Pruning

```python
model_path = "path/to/pretrained.pth"
sparsity = 0.50  # 50% 필터 제거

model = load_and_prune_model(model_path, sparsity)
# 출력: my_pruned_model_50.pth
```

### Reducing

```python
reduced_model = load_and_reduce_model(model_path, sparsity)
# 출력: reduced_model_50_shoe.pth
```

### 추론

```bash
python reduced_demo.py \
    --load_model reduced_model_50.pth \
    --demo images/test.jpg \
    --arch dla_34 \
    --c shoe
```

---

## 주의사항

1. **Fine-tuning 권장**: Pruning 후 성능 저하 시 Fine-tuning 필요
2. **Sparsity 조정**: 성능-크기 트레이드오프에 따라 0.3~0.5 권장
3. **카테고리별 설정**: shoe, chair, cup 등 카테고리에 맞게 설정

---

## 관련 파일

- [centerpose_utils/](../centerpose_utils/) - 간편 사용 유틸리티
- [메인 README](../README.md) - 전체 압축 과정 설명
- [CenterPose 원본](https://github.com/NVlabs/CenterPose)
