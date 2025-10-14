# Phase 2: Model Reducing (모델 축소)

## 개요

Pruning 단계에서 필터를 0으로 만들었지만, 모델 크기는 변하지 않았습니다.
Reducing 단계에서는 0이 된 필터를 **물리적으로 제거**하여 실제로 작은 모델을 생성합니다.

### 목표
- 프루닝된 필터를 모델에서 완전히 제거
- 채널 수 감소 → 메모리 사용량 감소
- 실제 추론 속도 향상

---

## Phase 2의 핵심 개념

### Before Reducing (Pruning 후)
```
Conv Layer:
  weight: [256, 128, 3, 3]  (256 filters, but 50 are zeros)
  실제 메모리: 256 * 128 * 3 * 3 = 294,912 parameters
  유효 필터: 206개 (256 - 50)
```

### After Reducing
```
Conv Layer:
  weight: [206, 128, 3, 3]  (only survived 206 filters)
  실제 메모리: 206 * 128 * 3 * 3 = 237,312 parameters
  메모리 절약: ~19.5%
```

---

## 핵심 함수 분석

### 1. 살아남은 필터 식별 (`get_survived_filter_idx`)

**위치**: `compression_src/reducing/common.py:4-9`

```python
def get_survived_filter_idx(layer):
    weight = layer.weight                              # [out_ch, in_ch, h, w]
    num_filters = weight.shape[0]

    # 각 필터의 L2 norm 계산
    filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)

    # norm이 0이 아닌 필터의 인덱스 반환
    survived_filter_idx = torch.where(filter_norms != 0)[0]
    return survived_filter_idx
```

**로직**:
1. Pruning 단계에서 0으로 만든 필터는 norm = 0
2. `torch.where(filter_norms != 0)`: non-zero norm을 가진 필터만 선택
3. 이 필터들만 새 모델로 복사됨

**예시**:
```python
# Pruning 후 필터 norm
filter_norms = [0.5, 0.0, 1.2, 0.0, 0.8, 0.3]
                 ↓
survived_idx = [0, 2, 4, 5]  # 인덱스 1, 3은 제외됨
```

---

### 2. Convolution Layer 축소 (`conv_reduce`)

**위치**: `compression_src/reducing/common.py:12-56`

```python
def conv_reduce(
    layer, reduced_layer, survived_out_channels_idx, survived_in_channels_idx
):
    # 1. 축소된 레이어의 채널 수 설정
    reduced_layer.in_channels = len(survived_in_channels_idx)
    reduced_layer.out_channels = len(survived_out_channels_idx)

    # 2. Depthwise convolution 처리
    if reduced_layer.groups != 1:
        reduced_layer.groups = reduced_layer.out_channels

    # 3. 새로운 가중치 텐서 생성
    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(
            reduced_layer.out_channels,     # 축소된 출력 채널
            reduced_layer.in_channels,      # 축소된 입력 채널
            reduced_layer.kernel_size[0],   # 커널 크기는 유지
            reduced_layer.kernel_size[1],
        ),
        requires_grad=True,
    )

    # 4. 살아남은 가중치만 복사
    weight = layer.weight                   # [256, 128, 3, 3]
    reduced_weight = reduced_layer.weight   # [206, 115, 3, 3]

    with torch.no_grad():
        # 출력 채널 선택 → 입력 채널 선택
        reduced_weight.copy_(
            weight[survived_out_channels_idx, :, :, :][
                :, survived_in_channels_idx, :, :
            ]
        )
```

#### 핵심 포인트

**A. 출력/입력 채널 동시 처리**
```python
# 원본: [256 out, 128 in, 3, 3]
# Step 1: 출력 채널 선택 → [206 out, 128 in, 3, 3]
weight[survived_out_channels_idx, :, :, :]

# Step 2: 입력 채널 선택 → [206 out, 115 in, 3, 3]
[..., survived_in_channels_idx, :, :]
```

**B. Depthwise Convolution 처리**
```python
if reduced_layer.groups != 1:
    reduced_layer.groups = reduced_layer.out_channels
```

**Depthwise Conv 특성**:
- 각 입력 채널마다 하나의 필터
- `groups = in_channels = out_channels`
- 채널이 줄어들면 groups도 함께 조정

**C. 단언문 (Assertion)**
```python
assert len(survived_out_channels_idx) == reduced_weight.shape[0]
assert len(survived_in_channels_idx) == reduced_weight.shape[1]
```
- 차원 불일치 방지
- 디버깅 시 오류 조기 발견

---

### 3. Batch Normalization 축소 (`bn_reduce`)

**위치**: `compression_src/reducing/common.py:107-154`

```python
def bn_reduce(layer, reduced_layer, survived_features_idx):
    # 1. Feature 개수 설정
    reduced_layer.num_features = len(survived_features_idx)

    # 2. 학습 가능한 파라미터 재생성
    reduced_layer.weight = torch.nn.Parameter(
        data=torch.zeros(reduced_layer.num_features),
        requires_grad=True,
    )
    reduced_layer.bias = torch.nn.Parameter(
        data=torch.zeros(reduced_layer.num_features),
        requires_grad=True,
    )

    # 3. Running statistics 재생성
    reduced_layer.running_mean = torch.zeros(reduced_layer.num_features)
    reduced_layer.running_var = torch.zeros(reduced_layer.num_features)

    # 4. 살아남은 feature의 파라미터만 복사
    with torch.no_grad():
        reduced_layer.weight.copy_(layer.weight[survived_features_idx])
        reduced_layer.bias.copy_(layer.bias[survived_features_idx])
        reduced_layer.running_mean.copy_(layer.running_mean[survived_features_idx])
        reduced_layer.running_var.copy_(layer.running_var[survived_features_idx])
```

**Batch Norm 파라미터**:
- `weight` (γ): Scale parameter [C]
- `bias` (β): Shift parameter [C]
- `running_mean` (μ): Running average of mean [C]
- `running_var` (σ²): Running average of variance [C]

**축소 로직**:
- Conv의 출력 채널이 256 → 206으로 줄면
- BN의 feature도 256 → 206으로 줄어야 함
- 각 채널에 대응하는 파라미터만 복사

---

## YOLOv8 Reducing 구현

### 전체 구조

**위치**: `compression.py:51-175`

```python
def yolov8_reducing(model, reduced_model):
    # Phase 1: 첫 번째 Conv 레이어 (RGB 입력)
    # Phase 2: Backbone 블록들 (1-22)
    # Phase 3: Detect Head
```

---

### Phase 1: 첫 번째 Conv 레이어

```python
# 첫 Conv 레이어 축소
survived_idx = get_survived_filter_idx(model[0].conv)

conv_reduce(
    layer=model[0].conv,
    reduced_layer=reduced_model[0].conv,
    survived_out_channels_idx=survived_idx,
    survived_in_channels_idx=torch.arange(3),  # RGB 입력은 항상 3채널
)

bn_reduce(model[0].bn, reduced_model[0].bn, survived_idx)
prev_survived_idx = survived_idx
```

**특이사항**:
- 입력은 항상 RGB 3채널 → `torch.arange(3)`
- 출력 채널만 프루닝됨

---

### Phase 2: Backbone 블록 축소

```python
block_list = [model[i] for i in range(1, 23)]
reduced_block_list = [reduced_model[i] for i in range(1, 23)]

for i, (block, reduced_block) in enumerate(zip(block_list, reduced_block_list)):
    if type(block).__name__ == 'Conv':
        # Conv 블록 처리
        ...
    elif type(block).__name__ in ['C2f', 'SPPF']:
        # C2f/SPPF 블록 처리
        ...
```

#### A. Conv 블록 축소

```python
if type(block).__name__ == 'Conv':
    survived_idx = get_survived_filter_idx(block.conv)

    conv_reduce(
        layer=block.conv,
        reduced_layer=reduced_block.conv,
        survived_out_channels_idx=survived_idx,
        survived_in_channels_idx=prev_survived_idx,  # 이전 레이어 출력
    )

    bn_reduce(block.bn, reduced_block.bn, survived_idx)
    prev_survived_idx = survived_idx  # 다음 레이어를 위해 저장
```

**연쇄 반응 (Cascade)**:
```
Layer 1: 256 filters → 206 survived
         ↓
Layer 2 입력: 206 channels (이전 레이어 출력)
Layer 2: 512 filters → 410 survived
         ↓
Layer 3 입력: 410 channels
...
```

#### B. C2f/SPPF 블록 축소 - Concatenation 처리

**YOLOv8의 핵심: Skip Connection**

```
Layer 8 output ─┐
                ├─ concat ─→ Layer 11 (C2f)
Layer 5 output ─┘
```

**문제**: 두 레이어의 출력을 concat하면?
- Layer 8: 206 channels survived
- Layer 5: 115 channels survived
- Layer 11 입력: 206 + 115 = 321 channels

**해결책**: `compression.py:97-111`

```python
if i in [11, 14, 17, 20]:  # Concat이 있는 레이어
    # 연결되는 두 레이어 정의
    n1, n2 = (8, 5) if i == 11 else \
             (11, 3) if i == 14 else \
             (15, 11) if i == 17 else \
             (18, 8) if i == 20 else (None, None)

    # 첫 번째 레이어의 survived indices
    p_surv_idx_1 = get_survived_filter_idx(
        block_list[n1].conv if i in [17, 20] else block_list[n1].cv2.conv
    )

    # 두 번째 레이어의 출력 채널 수 (offset으로 사용)
    p_surv_idx_2p = block_list[n1].conv.out_channels if i in [17, 20] \
                    else block_list[n1].cv2.conv.out_channels

    # 두 번째 레이어의 survived indices (offset 추가)
    p_surv_idx_2 = get_survived_filter_idx(block_list[n2].cv2.conv) + p_surv_idx_2p

    # 두 인덱스를 concat
    prev_survived_idx = torch.concat([p_surv_idx_1, p_surv_idx_2])
```

**예시**:
```python
# Layer 8 survived: [0, 2, 3, 5, 7]
# Layer 5 survived: [1, 3, 4]

# Layer 8 출력 채널 수: 256
# Layer 5 indices with offset: [1, 3, 4] + 256 = [257, 259, 260]

# Concat result: [0, 2, 3, 5, 7, 257, 259, 260]
```

**C2f 블록 내부 축소**:

```python
# cv1 레이어 축소 (입력 처리)
survived_idx = torch.arange(block.cv1.conv.out_channels)  # cv1은 프루닝 안됨
conv_reduce(
    layer=block.cv1.conv,
    reduced_layer=reduced_block.cv1.conv,
    survived_out_channels_idx=survived_idx,
    survived_in_channels_idx=prev_survived_idx,  # concat된 입력
)

# cv2 레이어 축소 (출력 생성)
prev_survived_idx = torch.arange(block.cv2.conv.in_channels)
survived_idx = get_survived_filter_idx(block.cv2.conv)

conv_reduce(
    layer=block.cv2.conv,
    reduced_layer=reduced_block.cv2.conv,
    survived_out_channels_idx=survived_idx,
    survived_in_channels_idx=prev_survived_idx,
)

bn_reduce(block.cv2.bn, reduced_block.cv2.bn, survived_idx)
```

**C2f 구조**:
```
Input → cv1 (channel expansion) → [Bottlenecks] → cv2 (프루닝됨) → Output
```

---

### Phase 3: Detect Head 축소

**Detect 구조**:
```
P3 (small)  ─→ cv2[0], cv3[0]  (작은 객체)
P4 (medium) ─→ cv2[1], cv3[1]  (중간 객체)
P5 (large)  ─→ cv2[2], cv3[2]  (큰 객체)
```

**구현**: `compression.py:138-172`

```python
if type(block).__name__ == 'Detect':
    prev_indices = [14, 17, 20]  # P3, P4, P5 feature map 생성 레이어
    cv_layers = [0, 1, 2]

    for idx, prev_idx in zip(cv_layers, prev_indices):
        # cv2 (bbox), cv3 (class) 처리
        for sub_layer, reduced_sub_layer in zip([block.cv2, block.cv3],
                                                  [reduced_block.cv2, reduced_block.cv3]):
            # 이전 레이어에서 survived indices 가져오기
            prev_survived_idx = get_survived_filter_idx(block_list[prev_idx].cv2.conv)

            # cv2[idx][0], cv2[idx][1] 또는 cv3[idx][0], cv3[idx][1]
            for i in range(2):
                survived_idx = get_survived_filter_idx(sub_layer[idx][i].conv)
                conv_reduce(
                    layer=sub_layer[idx][i].conv,
                    reduced_layer=reduced_sub_layer[idx][i].conv,
                    survived_out_channels_idx=survived_idx,
                    survived_in_channels_idx=prev_survived_idx,
                )
                bn_reduce(sub_layer[idx][i].bn, reduced_sub_layer[idx][i].bn, survived_idx)
                prev_survived_idx = survived_idx  # 다음 sub-layer 입력

            # 마지막 Conv2d 레이어 (출력 레이어)
            survived_idx = get_survived_filter_idx(sub_layer[idx][2])
            conv_reduce(
                layer=sub_layer[idx][2],
                reduced_layer=reduced_sub_layer[idx][2],
                survived_out_channels_idx=survived_idx,
                survived_in_channels_idx=prev_survived_idx,
            )
```

**특이사항**:
- Detect의 최종 출력 채널은 고정됨
  - cv2: `4 * reg_max` (bbox regression)
  - cv3: `num_classes` (classification)
- 중간 레이어만 프루닝, 최종 출력은 유지

**주석 참조** (`compression.py:164-165`):
```python
# For the Conv2d layer, only the corresponding input channels are reduced
# The output channels of Conv2d are pre-defined by self.reg_max and self.nc,
# so reducing them incorrectly may cause errors
```

---

## 전체 프로세스 요약

### 1. Pruning
```python
model = YOLO('yolov8n.pt')
yolov8_pruning(model.model, sparsity=0.3)  # 30% 필터 제거
# 모델 크기는 동일, 많은 가중치가 0
```

### 2. Fine-tuning (필요시)
```python
# Pruning으로 인한 정확도 손실 회복
model.train(data='coco.yaml', epochs=10)
```

### 3. Reducing
```python
reduced_model = YOLO('yolov8n.pt')  # 새 모델 생성
yolov8_reducing(model.model, reduced_model.model)
# 실제로 작은 모델, 메모리 절약
```

### 4. 결과
```
Original:  [256, 512, 256, 512, ...]  channels
Pruned:    [256, 512, 256, 512, ...]  (many zeros)
Reduced:   [206, 410, 218, 435, ...]  channels (물리적으로 축소)

메모리:    100% → 100% → 75%
속도:      100% → ~105% → 130%  (추정치)
정확도:    100% → ~97% → ~97%   (fine-tuning 후)
```

---

## Architecture-Aware 압축의 중요성

### 일반 모델 vs YOLO

**일반적인 Sequential 모델**:
```
Conv1 → Conv2 → Conv3 → ...
```
- 간단: 각 레이어 순차적으로 축소

**YOLO (Skip Connections)**:
```
    ┌─→ Conv8 ─┐
Conv1           ├─concat─→ C2f11 → ...
    └─→ Conv5 ─┘
```
- 복잡: Concat으로 인한 채널 추적 필요

### 이 코드의 해결책

1. **명시적 concatenation 처리**
   - 어떤 레이어들이 concat되는지 하드코딩
   - `if i in [11, 14, 17, 20]`

2. **인덱스 offset 계산**
   - Concat 시 두 번째 입력의 인덱스에 첫 번째 출력 채널 수 추가
   - `p_surv_idx_2 = get_survived_filter_idx(...) + p_surv_idx_2p`

3. **Detect head 특별 처리**
   - 3개 스케일, 각각 다른 feature map에서 입력
   - `prev_indices = [14, 17, 20]`

---

## 장점과 한계

### 장점

1. **실제 압축 달성**
   - 메모리 사용량 실제 감소
   - 추론 속도 실제 향상

2. **Hardware-friendly**
   - Dense tensor 유지
   - GPU 최적화 활용 가능

3. **모듈화**
   - `pruning/` 과 `reducing/` 분리
   - 다른 모델에도 적용 가능 (ResNet 구현 존재)

### 한계

1. **Architecture-specific**
   - YOLOv8 구조에 강하게 의존
   - Concat 레이어를 수동으로 지정
   - 새로운 YOLO 버전에는 수정 필요

2. **Magnitude-based의 한계**
   - Feature importance를 완전히 반영 못함
   - 더 정교한 방법 (Gradient-based, Hessian-based) 가능

3. **Fine-tuning 필수**
   - Pruning 후 정확도 손실 불가피
   - 재학습 시간 필요

---

## 다음 읽을 내용

- `03_usage_examples.md`: 실제 사용 예제
- `04_advanced_topics.md`: 고급 주제 (sparsity 선택, 성능 평가)
