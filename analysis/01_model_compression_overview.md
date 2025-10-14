# YOLOv8 AI 모델 압축 기법 분석

## 개요

이 문서는 `ultralytics_custom`에 구현된 YOLOv8 신경망 모델 압축 기법을 분석합니다.

### 압축 목표
- **모델 크기 축소**: 메모리 사용량 감소
- **연산량 감소**: 추론 속도 향상
- **정확도 유지**: 최소한의 성능 손실

### 압축 방식
**Structured Pruning + Channel Reduction** 조합
1. **Phase 1: Pruning** - 불필요한 필터를 0으로 만듦 (가지치기)
2. **Phase 2: Reducing** - 0이 된 필터를 물리적으로 제거하여 작은 모델 생성

---

## 핵심 파일 구조

```
ultralytics_custom/engine/
├── compression.py                     # YOLOv8 전용 압축 오케스트레이션
└── compression_src/
    ├── pruning/
    │   ├── common.py                  # 범용 프루닝 함수
    │   └── resnet.py                  # ResNet 전용 프루닝
    ├── reducing/
    │   ├── common.py                  # 범용 축소 함수
    │   └── resnet.py                  # ResNet 전용 축소
    └── models/
        └── resnet.py                  # ResNet 모델 정의
```

---

## Phase 1: Pruning (가지치기)

### 원리: Magnitude-based Filter Pruning

중요하지 않은 필터를 식별하여 가중치를 0으로 만듭니다.

### 알고리즘

#### 1. 프루닝할 필터 선택 (`get_filter_pruning_idx`)

**위치**: `compression_src/pruning/common.py:33-40`

```python
def get_filter_pruning_idx(layer, sparsity):
    with torch.no_grad():
        weight = layer.weight                              # Conv2d weights: [out_ch, in_ch, h, w]
        num_filters = weight.shape[0]                      # 출력 채널 수 (필터 개수)
        num_pruning_filters = int(num_filters * sparsity)  # 제거할 필터 개수

        # 각 필터의 L2 norm 계산
        filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)

        # L2 norm이 가장 작은 필터들의 인덱스 반환
        _, pruning_idx = torch.topk(filter_norms, num_pruning_filters, largest=False)
    return pruning_idx
```

**핵심 아이디어**:
- **L2 Norm**: 각 필터의 가중치 크기를 측정
- **작은 norm = 덜 중요한 필터**: 출력에 미치는 영향이 작음
- **sparsity**: 제거할 비율 (예: 0.3 = 30% 제거)

**수식**:
```
filter_norm_i = ||W_i||_2 = sqrt(Σ w_ij^2)

여기서:
- W_i: i번째 필터의 모든 가중치
- w_ij: 필터의 개별 가중치
```

#### 2. 필터 프루닝 실행 (`filter_pruning`)

**위치**: `compression_src/pruning/common.py:15-18`

```python
def filter_pruning(layer, pruning_idx):
    weight = layer.weight
    with torch.no_grad():
        # 선택된 필터의 모든 가중치를 0으로 설정
        weight[pruning_idx, :, :, :] = 0.0
```

**효과**:
- 필터가 0이 되면 해당 출력 채널은 0만 생성
- 모델 구조는 유지되지만 해당 필터는 사실상 비활성화

#### 3. Batch Normalization 프루닝 (`bn_pruning`)

**위치**: `compression_src/pruning/common.py:21-30`

```python
def bn_pruning(layer, pruning_idx):
    weight = layer.weight        # γ (scale)
    bias = layer.bias            # β (shift)
    mean = layer.running_mean    # μ (running mean)
    var = layer.running_var      # σ² (running variance)

    with torch.no_grad():
        weight[pruning_idx] = 0.0   # γ = 0
        bias[pruning_idx] = 0.0     # β = 0
        mean[pruning_idx] = 0.0     # μ = 0
        var[pruning_idx] = 1.0      # σ² = 1
```

**배경 지식 - Batch Normalization**:
```
BN(x) = γ * ((x - μ) / sqrt(σ² + ε)) + β

프루닝 후:
- γ = 0, β = 0, μ = 0, σ² = 1
→ BN(x) = 0 * ((x - 0) / 1) + 0 = 0
```

**중요**: Conv 레이어와 함께 프루닝해야 일관성 유지

---

## YOLOv8 Pruning 구현

### YOLOv8 아키텍처 이해

```
[0] Conv          ─┐
[1-22] Backbone   │ 23개 레이어
    ├─ Conv       │
    ├─ C2f        │
    └─ SPPF       ┘
[23] Detect (Head)
    ├─ cv2 (bbox regression)
    └─ cv3 (classification)
```

### Pruning 실행 (`yolov8_pruning`)

**위치**: `compression.py:7-48`

#### 1. Backbone Pruning (레이어 0-22)

```python
def yolov8_pruning(model, sparsity):
    # Backbone 블록 생성 (0-22번 레이어)
    block_list = [model[i] for i in range(23)]

    for i, block in enumerate(block_list):
        # Conv 레이어 프루닝
        if type(block).__name__ == 'Conv':
            pruning_idx = get_filter_pruning_idx(layer=block.conv, sparsity=sparsity)
            filter_pruning(layer=block.conv, pruning_idx=pruning_idx)
            bn_pruning(block.bn, pruning_idx=pruning_idx)
```

**적용 대상**:
- `Conv`: block.conv + block.bn
- `C2f`: block.cv2.conv + block.cv2.bn
- `SPPF`: block.cv2.conv + block.cv2.bn

#### 2. Detect Head Pruning

```python
        if type(block).__name__ == 'Detect':
            # 3개 스케일 (large, medium, small)
            for i in range(3):
                # 각 스케일마다 2개 Conv 레이어
                for j in range(2):
                    # cv2: bbox regression branch
                    pruning_idx = get_filter_pruning_idx(layer=block.cv2[i][j].conv, sparsity=sparsity)
                    filter_pruning(layer=block.cv2[i][j].conv, pruning_idx=pruning_idx)
                    bn_pruning(block.cv2[i][j].bn, pruning_idx=pruning_idx)

                    # cv3: classification branch
                    pruning_idx = get_filter_pruning_idx(layer=block.cv3[i][j].conv, sparsity=sparsity)
                    filter_pruning(layer=block.cv3[i][j].conv, pruning_idx=pruning_idx)
                    bn_pruning(block.cv3[i][j].bn, pruning_idx=pruning_idx)
```

**Detect 구조**:
```
cv2[0][0-1]: Large object detection bbox
cv2[1][0-1]: Medium object detection bbox
cv2[2][0-1]: Small object detection bbox

cv3[0][0-1]: Large object classification
cv3[1][0-1]: Medium object classification
cv3[2][0-1]: Small object classification
```

---

## 왜 이 방식인가?

### Structured Pruning의 장점

1. **하드웨어 친화적**
   - Unstructured pruning (개별 가중치 제거)는 sparse matrix 연산 필요
   - Structured pruning (필터 단위 제거)는 일반 GPU/CPU에서 효율적

2. **실제 속도 향상**
   - 필터 제거 → 채널 수 감소 → FLOPs 감소
   - Dense matrix 연산 유지 → 하드웨어 최적화 활용

3. **구현 간단**
   - 복잡한 sparse tensor 라이브러리 불필요
   - PyTorch 기본 연산만으로 구현 가능

### Magnitude-based Pruning의 한계

1. **최적이 아닐 수 있음**
   - Norm이 작아도 중요한 필터일 수 있음
   - Feature importance를 완전히 반영하지 못함

2. **Layer-wise pruning**
   - 각 레이어를 독립적으로 프루닝
   - Global optimal은 아님

3. **Fine-tuning 필수**
   - Pruning 후 정확도 회복을 위한 재학습 필요

---

## 다음 단계

Phase 2: Reducing에서는 프루닝된 필터를 물리적으로 제거하여 실제로 작은 모델을 생성합니다.

→ `02_model_reducing_analysis.md` 참조
