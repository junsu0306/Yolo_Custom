# CenterPose Compression

DLA-34 기반 CenterPose 6D Pose Estimation 모델에 대한 **One-shot Structured Pruning** 구현입니다.

---

## 개요

### 핵심 기술

| 기법 | 설명 |
|------|------|
| **Pruning** | L2-norm 기반 One-shot Structured Pruning |
| **타겟 모듈** | DLA-34의 BasicBlock 단위 Blockwise Pruning |
| **Reducing** | 물리적 채널 제거를 통한 실제 모델 경량화 |
| **Memory Loss** | 학습 중 메모리 사용량 기반 최적화 |

### 모델 구조

```
CenterPose (DLA-34 기반)
├── Base Network: DLA-34 (~20M params)
│   └── BasicBlock × 여러 개 (Pruning 대상)
├── Upsampling: DLA-Up, IDA-Up
├── ConvGRU: Temporal processing
└── Detection Heads: hm, wh, reg, hps, scale 등
```

---

## 수정 및 추가된 파일 목록

```
CenterPose src custom/
├── pruning.py                       # 메인 Pruning 실행 스크립트
├── reducing.py                      # 메인 Reducing 실행 스크립트
├── pruning_TW.py                    # 메모리 측정 포함 Pruning
├── dlasg_pruning.py                 # DLA 모델 전용 Pruning 함수
├── dlasg_reducing.py                # DLA 모델 전용 Reducing 함수
├── memory_usage.py                  # 레이어별 메모리 측정 도구
├── reduced_demo.py                  # 압축 모델 Demo 실행
├── memmory_comparison.txt           # 메모리 비교 결과
│
└── lib/
    ├── trains/
    │   └── base_trainer.py          # 학습 루프에 압축 통합
    │       ├── dlasg_blockwise_pruning() 호출
    │       ├── measure_model_memory()
    │       ├── measure_pruned_layer_memory()
    │       └── custom_memory_loss_function()
    │
    └── pruning/
        ├── dlasg_pruning.py         # Pruning 핵심 구현
        │   ├── filter_pruning()
        │   ├── bn_pruning()
        │   ├── get_filter_norms()
        │   ├── get_pruning_indices()
        │   ├── dlasg_blockwise_pruning()
        │   └── reduce_pruned_model()
        │
        └── memory_usage.py          # 메모리 프로파일링
            ├── measure_memory()
            ├── extract_layers()
            ├── measure_model_memory()
            ├── measure_pruned_layer_memory()
            └── custom_memory_loss_function()
```

---

## Pruning 알고리즘

### 1. L2 Norm 기반 필터 선택

**목적**: Conv 필터의 중요도를 L2 norm으로 측정하여 Pruning 대상 선정

**핵심 함수**: `lib/pruning/dlasg_pruning.py`

```python
def get_filter_norms(layer, inf=99999):
    """L2 Norm 계산 (최대값 보호)"""
    # 각 필터의 L2 norm 계산
    filter_norms = torch.norm(
        layer.weight.view(layer.weight.shape[0], -1),
        dim=1
    )

    # 최대 norm 필터는 Pruning 방지 (inf로 설정)
    max_idx = torch.argmax(filter_norms)
    filter_norms[max_idx] = inf

    return filter_norms
```

### 2. Global Sparsity 기반 Pruning Index 계산

```python
def get_pruning_indices(filter_norms, sparsity):
    """전체 레이어를 고려한 Global Pruning"""
    # 모든 레이어의 norm 합치기
    all_norms = torch.cat(filter_norms)

    # 전체에서 sparsity 비율만큼 선택
    num_pruning_filters = int(all_norms.numel() * sparsity)

    # 가장 작은 norm을 가진 필터들 선택
    _, global_pruning_idx = torch.topk(
        all_norms,
        num_pruning_filters,
        largest=False  # 작은 값부터
    )

    # 각 레이어별 local index로 변환
    return convert_to_local_indices(global_pruning_idx)
```

**특징**:
- **Global Sparsity**: 전체 레이어를 통합 고려
- **최대 Norm 보호**: 가장 중요한 필터는 항상 보존
- **Top-k Smallest**: 가장 작은 norm 필터 선택

---

## BasicBlock Blockwise Pruning

### DLA-34 BasicBlock 구조

```
BasicBlock
├── conv1: 3×3 Conv (Pruning 대상)
├── bn1: BatchNorm
├── relu
├── conv2: 3×3 Conv (입력 채널 조정)
└── bn2: BatchNorm
    ↓
(+ residual connection)
```

### Blockwise Pruning 구현

**핵심 함수**: `dlasg_blockwise_pruning()` in `lib/pruning/dlasg_pruning.py`

```python
def dlasg_blockwise_pruning(model, sparsity, device='cpu'):
    """DLA-34의 BasicBlock 단위로 구조화된 Pruning 적용"""

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'BasicBlock':
            conv1, bn1 = module.conv1, module.bn1
            conv2, bn2 = module.conv2, module.bn2

            # Step 1: Conv1 출력 채널의 L2 norm 계산
            norms1 = get_filter_norms(conv1)

            # Step 2: Global sparsity 기준 Pruning index 계산
            prune_idx1 = get_pruning_indices([norms1], sparsity)[0]

            # Step 3: Conv1 필터 마스킹 (weight = 0)
            filter_pruning(conv1, prune_idx1)

            # Step 4: BN1 파라미터 마스킹
            bn_pruning(bn1, prune_idx1)
```

### Pruning 메커니즘 상세

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

---

## Channel Reduction (물리적 제거)

### 목적

마스킹된(0인) 채널을 물리적으로 제거하여 실제 모델 크기와 메모리 사용량 감소

### 구현

**핵심 함수**: `reduce_pruned_model()` in `lib/pruning/dlasg_pruning.py`

```python
def reduce_pruned_model(model):
    """0인 필터를 물리적으로 제거한 축소 모델 생성"""

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'BasicBlock':
            conv1, bn1 = module.conv1, module.bn1
            conv2, bn2 = module.conv2, module.bn2

            # Step 1: 살아있는 채널 찾기 (norm != 0)
            keep = torch.where(
                conv1.weight.view(conv1.weight.shape[0], -1).abs().sum(1) != 0
            )[0]

            # Step 2: Conv1 축소 (출력 채널 감소)
            new_conv1 = nn.Conv2d(
                in_channels=conv1.in_channels,
                out_channels=keep.numel(),  # 축소된 크기!
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=conv1.bias is not None
            )
            new_conv1.weight.data = conv1.weight[keep].clone()

            # Step 3: Conv2 축소 (입력 채널 감소)
            new_conv2 = nn.Conv2d(
                in_channels=keep.numel(),  # Conv1 출력에 맞춤!
                out_channels=conv2.out_channels,
                kernel_size=conv2.kernel_size,
                stride=conv2.stride,
                padding=conv2.padding,
                bias=conv2.bias is not None
            )
            new_conv2.weight.data = conv2.weight[:, keep].clone()

            # Step 4: BN1 축소
            new_bn1 = nn.BatchNorm2d(keep.numel())
            new_bn1.weight.data = bn1.weight[keep].clone()
            new_bn1.bias.data = bn1.bias[keep].clone()
            new_bn1.running_mean.data = bn1.running_mean[keep].clone()
            new_bn1.running_var.data = bn1.running_var[keep].clone()

            # Step 5: 모듈 교체
            _assign_module(model, f"{name}.conv1", new_conv1)
            _assign_module(model, f"{name}.conv2", new_conv2)
            _assign_module(model, f"{name}.bn1", new_bn1)
```

### Reducing 전후 비교

```
Before Reducing:
├── conv1: [64, 32, 3, 3]  → 18,432 params (50% are zeros)
├── bn1:   [64]            → 256 params
├── conv2: [64, 64, 3, 3]  → 36,864 params
└── Total: 55,552 params

After Reducing (50% sparsity):
├── conv1: [32, 32, 3, 3]  → 9,216 params (실제 사용)
├── bn1:   [32]            → 128 params
├── conv2: [64, 32, 3, 3]  → 18,432 params
└── Total: 27,776 params (50% 감소)
```

---

## 메모리 측정

### 목적

레이어별 GPU 메모리 사용량을 프로파일링하여 최적화 대상 식별

### 구현

**핵심 함수**: `memory_usage.py`

```python
def measure_memory(x, layers, device):
    """레이어별 메모리 사용량 측정"""

    # GPU 메모리 초기화
    x.cpu()
    for layer in layers:
        layer.cpu()
    torch.cuda.empty_cache()

    before_memory = torch.cuda.memory_allocated(device) / 1024**2  # MB

    # Forward pass
    with torch.no_grad():
        x = x.to(device)
        for layer in layers:
            layer.to(device)
            x = layer(x)

    after_memory = torch.cuda.memory_allocated(device) / 1024**2  # MB

    return x, after_memory - before_memory
```

### 측정 대상

| 모듈 | 설명 |
|------|------|
| **DLA Base Layers** | level0 ~ level5 |
| **Upsampling** | dla_up, ida_up |
| **ConvGRU** | 4-step sequential GRU |
| **Detection Heads** | hm, wh, reg, hps, scale 등 |

---

## Training-time 압축 통합

### Base Trainer 수정

**핵심 파일**: `lib/trains/base_trainer.py`

```python
# Training phase에서 backprop 후 압축 적용
if phase == 'train':
    self.optimizer.zero_grad()
    loss.backward()

    # 1. 모델 추출 (DataParallel 고려)
    if isinstance(model_with_loss, torch.nn.DataParallel):
        model_for_pruning = model_with_loss.module.model.to(opt.device)
    else:
        model_for_pruning = model_with_loss.model.to(opt.device)

    # 2. Blockwise Pruning 수행
    dlasg_blockwise_pruning(
        model_for_pruning,
        sparsity=0.5,  # 50% 필터 제거
        device=opt.device
    )

    # 3. Pruning 전 메모리 측정 (최초 1회)
    if self.file_stream is not None:
        _ = measure_model_memory(
            model_for_pruning,
            self.dummy_input,
            opt.device,
            self.file_stream
        )

    # 4. Reducing 후 메모리 측정
    mem_usg = measure_pruned_layer_memory(
        model_for_pruning,
        self.dummy_input,
        opt.device,
        self.file_stream
    )

    # 5. 메모리 기반 Loss 추가
    hyperparam = 1.0
    device_condition_memory = 1.0  # MB 단위
    loss += custom_memory_loss_function(
        mem_usg,
        hyperparam,
        device_condition_memory
    )

    # 6. Gradient clipping 및 optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
    self.optimizer.step()
```

### 특징

- **Training-time Pruning**: 매 iteration마다 자동 실행
- **메모리 제약 Loss**: 메모리 사용량을 loss에 반영
- **DataParallel 호환**: 멀티 GPU 환경 지원
- **Gradient Clipping**: 안정적 학습 보장

---

## 전체 Compression Flow

```
┌─────────────────────────────────────────────────────────┐
│                  Compression Start                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  1. Model Loading                                       │
│     - Load pretrained DLA-34 model                      │
│     - Create model with opts configuration              │
│     - Load checkpoint (.pth file)                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  2. Pruning Phase                                       │
│     ┌─────────────────────────────────────────┐        │
│     │  For each BasicBlock:                   │        │
│     │                                         │        │
│     │  A. Analyze Conv1 Filters               │        │
│     │     - Calculate L2 norms                │        │
│     │     - Protect max norm filter (inf)     │        │
│     │                                         │        │
│     │  B. Select Pruning Targets              │        │
│     │     - Global sparsity threshold         │        │
│     │     - Top-k smallest norms              │        │
│     │                                         │        │
│     │  C. Apply Pruning                       │        │
│     │     - conv1.weight[idx] = 0.0           │        │
│     │     - bn1 parameters = 0.0              │        │
│     └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  3. Reducing Phase                                      │
│     ┌─────────────────────────────────────────┐        │
│     │  For each BasicBlock:                   │        │
│     │                                         │        │
│     │  A. Identify Survived Channels          │        │
│     │     - keep = where(abs(weight).sum != 0)│        │
│     │                                         │        │
│     │  B. Create Reduced Conv1                │        │
│     │     - new out_channels = len(keep)      │        │
│     │                                         │        │
│     │  C. Create Reduced Conv2                │        │
│     │     - new in_channels = len(keep)       │        │
│     │                                         │        │
│     │  D. Create Reduced BN1                  │        │
│     └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  4. Save & Verify                                       │
│     - Save reduced model (.pth)                         │
│     - Verify parameter reduction                        │
│     - Measure memory usage                              │
└─────────────────────────────────────────────────────────┘
```

---

## 상세 Compression Flow

```
[1] 모델 로드
    ├─ opt 설정
    │   ├─ arch = 'dla_34'
    │   ├─ tracking_task = True
    │   └─ heads 설정 (hm, reg, wh, hps, scale 등)
    │
    └─ load_model(model, checkpoint_path)

[2] Pruning 실행
    └─ dlasg_blockwise_pruning(model, sparsity=0.5)
        │
        ├─ [2-1] BasicBlock 순회
        │
        ├─ [2-2] Conv1 필터 norm 계산
        │   ├─ norms = L2_norm(conv1.weight)
        │   └─ norms[max_idx] = inf  # 보호
        │
        ├─ [2-3] Global pruning index 계산
        │   └─ top-k smallest norms
        │
        ├─ [2-4] Conv1 Pruning
        │   └─ conv1.weight[prune_idx] = 0.0
        │
        └─ [2-5] BN1 Pruning
            ├─ bn1.weight[prune_idx] = 0.0
            ├─ bn1.bias[prune_idx] = 0.0
            └─ bn1.running_var[prune_idx] = 1.0

[3] Reducing 실행
    └─ reduce_pruned_model(model)
        │
        ├─ [3-1] 살아있는 채널 찾기
        │   └─ keep = where(weight != 0)
        │
        ├─ [3-2] Conv1 축소
        │   └─ out_channels = len(keep)
        │
        ├─ [3-3] Conv2 축소
        │   └─ in_channels = len(keep)
        │
        └─ [3-4] BN1 축소

[4] 모델 저장
    └─ torch.save(reduced_model, 'reduced_model_50.pth')
```

---

## 실행 방법

### 1. Pruning만 수행

```bash
cd "CenterPose src custom"
python pruning.py
# Output: my_pruned_model_50.pth (0으로 마스킹된 상태)
```

### 2. Pruning + Reducing

```bash
python reducing.py
# Output: reduced_model_50.pth (물리적으로 축소된 모델)
```

### 3. 메모리 측정 포함 Pruning

```bash
python pruning_TW.py
# Output: 레이어별 메모리 사용량 출력
```

### 4. 압축 모델 Demo 실행

```bash
python reduced_demo.py --load_model ../models/reduced_model_50.pth
# Output: 압축 모델 기반 실시간 추론
```

### 5. Training-time 자동 압축

```bash
python main.py --task objectpose --exp_id compression_exp
# 매 iteration마다 자동 pruning + 메모리 최적화
```

---

## 압축 결과

### Parameter Count

```
Original Model:     ~20M parameters
Pruned Model (50%): ~20M parameters (50%가 0)
Reduced Model:      ~12-14M parameters (물리적 제거)
Reduction Ratio:    30-40%
```

### Memory Usage

```
Original Model:  ~150-200 MB (inference)
Reduced Model:   ~100-130 MB (inference)
Memory Reduction: 30-40%
```

---

## YOLO vs CenterPose 압축 비교

| 항목 | YOLO Compression | CenterPose Compression |
|------|------------------|------------------------|
| **백본** | CSPDarknet (YOLOv8) | DLA-34 |
| **타겟 모듈** | Conv, C2f, SPPF, Detect | BasicBlock |
| **Pruning 방식** | Dynamic (매 step) | One-shot |
| **Knowledge Distillation** | Yes (YOLOv8x → v8n) | No |
| **Training 필요** | Yes | No (사전 학습 모델만) |
| **Reducing 시점** | 학습 후 (선택) | Pruning 직후 |
| **복잡도** | High (통합 학습) | Low (후처리) |
| **성능 유지** | 더 좋음 (KD 효과) | Fine-tuning 권장 |

---

## 핵심 차이점 요약

### CenterPose 특징

1. **One-shot Pruning**
   - 학습 없이 가중치 분석만으로 Pruning
   - 빠른 압축 가능

2. **BasicBlock 단위 처리**
   - DLA 구조의 특성 반영
   - Conv1-BN1-Conv2-BN2 세트 처리

3. **No Knowledge Distillation**
   - Teacher 모델 불필요
   - 메모리 효율적

### YOLO 특징

1. **Training-time Pruning**
   - 학습 과정에서 동적으로 Pruning
   - Distillation과 동시 진행

2. **다양한 모듈 처리**
   - C2f, SPPF, Detect 각각 전용 로직
   - Concatenation 추적 필요

3. **Knowledge Distillation**
   - Teacher(YOLOv8x) → Student(YOLOv8n)
   - 더 나은 성능 유지

---

## 실제 실행 가이드 (Step-by-Step)

이 섹션에서는 **처음부터 끝까지** 압축을 진행하고 실제로 사용하는 전체 과정을 설명합니다.

### 사전 요구사항

```bash
# Python 3.8+ 및 PyTorch 1.10+ 필요
pip install torch torchvision
pip install opencv-python numpy
```

### Step 1: 원본 CenterPose 레포지토리 클론

```bash
# CenterPose 원본 레포지토리 클론
git clone https://github.com/NVlabs/CenterPose.git
cd CenterPose

# 의존성 설치
pip install -r requirements.txt

# DCNv2 빌드 (필요한 경우)
cd src/lib/models/networks/DCNv2
python setup.py build develop
cd ../../../../..
```

### Step 2: Custom 압축 파일 적용

```bash
# Custom 파일들을 CenterPose에 복사
# (Yolo_Custom 프로젝트 루트에서 실행)

# 핵심 압축 모듈 복사
cp "CenterPose src custom/lib/pruning/dlasg_pruning.py" CenterPose/src/lib/pruning/
cp "CenterPose src custom/lib/pruning/memory_usage.py" CenterPose/src/lib/pruning/

# 실행 스크립트 복사
cp "CenterPose src custom/pruning.py" CenterPose/src/
cp "CenterPose src custom/reducing.py" CenterPose/src/
cp "CenterPose src custom/reduced_demo.py" CenterPose/src/

# (선택) Training-time 압축을 사용할 경우
cp "CenterPose src custom/lib/trains/base_trainer.py" CenterPose/src/lib/trains/
```

### Step 3: 사전 학습된 모델 준비

```bash
# CenterPose 사전 학습 모델 다운로드
# https://github.com/NVlabs/CenterPose#pre-trained-models 참조

# 예: shoe 카테고리 모델
mkdir -p CenterPose/models/CenterPoseTrack
# 다운로드한 .pth 파일을 models/ 폴더에 저장
```

### Step 4: Pruning 실행 (가중치 마스킹)

```bash
cd CenterPose/src

# pruning.py 내 모델 경로 수정
# model_path = "경로/to/your/model.pth"
# sparsity = 0.50  # 50% 필터 제거

python pruning.py
```

**pruning.py 수정 예시:**
```python
if __name__ == "__main__":
    # 본인의 모델 경로로 수정
    model_path = "../models/CenterPoseTrack/shoe_15.pth"
    sparsity = 0.50  # 50% pruning
    model = load_and_prune_model(model_path, sparsity)
```

**출력:**
```
Creating model...
Loading model weights from ../models/CenterPoseTrack/shoe_15.pth...
Applying pruning with sparsity 0.5...
Pruned model weights saved as my_pruned_model_50.pth (epoch 15)
```

### Step 5: Reducing 실행 (물리적 채널 제거)

```bash
# reducing.py 내 모델 경로 수정 후 실행
python reducing.py
```

**reducing.py 수정 예시:**
```python
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model_path = "../models/CenterPoseTrack/shoe_15.pth"  # 원본 모델
    sparsity = 0.50
    reduced_model = load_and_reduce_model(model_path, sparsity)
```

**출력:**
```
Total parameters: 20000000
zero parameters: 10000000
Reducing model with pruning threshold...
Reduced model weights saved as reduced_model_50_shoe.pth (epoch 15)
After reduce Total parameters: 12000000
After reduce zero parameters: 0
```

### Step 6: 압축된 모델로 추론 실행

```bash
# 이미지/비디오에 대해 추론 실행
python reduced_demo.py \
    --load_model ../models/reduced_model_50_shoe.pth \
    --demo ../images/test_shoe.jpg \
    --arch dla_34 \
    --c shoe \
    --debug 2
```

**reduced_demo.py 사용법:**
```bash
# 단일 이미지
python reduced_demo.py --load_model reduced_model_50.pth --demo image.jpg

# 이미지 폴더
python reduced_demo.py --load_model reduced_model_50.pth --demo ./images/

# 비디오 파일
python reduced_demo.py --load_model reduced_model_50.pth --demo video.mp4

# 웹캠
python reduced_demo.py --load_model reduced_model_50.pth --demo webcam
```

### Step 7: 압축 결과 확인

```python
import torch

# 원본 모델 파라미터 수
original = torch.load("original_model.pth")
original_params = sum(p.numel() for p in original['state_dict'].values())

# 압축 모델 파라미터 수
reduced = torch.load("reduced_model_50.pth")
reduced_params = sum(p.numel() for p in reduced.parameters())

print(f"Original: {original_params:,} params")
print(f"Reduced:  {reduced_params:,} params")
print(f"Reduction: {(1 - reduced_params/original_params)*100:.1f}%")
```

---

## 전체 실행 흐름 요약

```
[1] 환경 준비
    └── CenterPose 원본 클론 + 의존성 설치
            ↓
[2] Custom 파일 적용
    └── lib/pruning/, pruning.py, reducing.py 복사
            ↓
[3] 모델 준비
    └── 사전 학습된 .pth 파일 다운로드
            ↓
[4] Pruning 실행
    └── python pruning.py
    └── 출력: my_pruned_model_50.pth (마스킹된 상태)
            ↓
[5] Reducing 실행
    └── python reducing.py
    └── 출력: reduced_model_50.pth (물리적 축소)
            ↓
[6] 추론 실행
    └── python reduced_demo.py --load_model reduced_model_50.pth
            ↓
[7] 결과 확인
    └── 파라미터 수, 메모리 사용량, 추론 속도 비교
```

---

## 주의사항

### 1. 모델 경로 수정 필수
- `pruning.py`, `reducing.py` 내의 `model_path` 변수를 본인 환경에 맞게 수정

### 2. GPU 메모리
- Reducing 과정에서 GPU 메모리 필요 (최소 4GB 권장)
- `os.environ["CUDA_VISIBLE_DEVICES"] = '0'`로 GPU 지정

### 3. 카테고리 설정
- `opt.c` 변수를 압축하려는 객체 카테고리로 설정
- 지원: `shoe`, `chair`, `cup`, `camera`, `bike`, `book`, `bottle`, `cereal_box`, `laptop`, `mug`

### 4. Tracking Task
- CenterPoseTrack 모델 사용 시 `opt.tracking_task = True` 설정 필요

### 5. 성능 저하 대응
- 압축 후 성능 저하가 크면 Fine-tuning 권장
- sparsity 값을 낮춰서 (0.3~0.4) 재시도
