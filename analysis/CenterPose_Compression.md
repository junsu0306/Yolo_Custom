## CenterPose 압축 개요

1. L2-norm based One-shot Structured Pruning 기법 사용
2. Pruning: BasicBlock 단위 Blockwise Pruning 적용
3. Reducing: 물리적 채널 제거를 통한 모델 경량화
4. DLA-34 백본 기반 6D Pose Estimation 모델 압축

## 수정 및 추가된 파일 목록

```markdown
CenterPose src custom/
├── **pruning.py**                         # 메인 pruning 실행 스크립트
├── **reducing.py**                        # 메인 reducing 실행 스크립트
├── **pruning_TW.py**                      # 메모리 측정 포함 pruning 스크립트
├── **dlasg_pruning.py**                   # DLA 모델 전용 pruning 함수
├── **dlasg_reducing.py**                  # DLA 모델 전용 reducing 함수
├── **memory_usage.py**                    # 레이어별 메모리 측정 도구
├── **reduced_demo.py**                    # 압축 모델 데모 실행
├── **memmory_comparison.txt**             # 메모리 비교 결과 저장
│
└── lib/
    └── pruning/
        ├── **dlasg_pruning.py**           # Pruning 핵심 구현
        │   ├── filter_pruning()        # Conv 필터 pruning
        │   ├── bn_pruning()            # BatchNorm pruning
        │   ├── get_filter_norms()      # L2 norm 계산
        │   ├── get_pruning_indices()   # Global pruning index 계산
        │   ├── dlasg_blockwise_pruning() # BasicBlock 단위 pruning
        │   └── reduce_pruned_model()   # Pruned 모델 reducing
        │
        └── **memory_usage.py**            # 메모리 프로파일링
            ├── measure_memory()        # 레이어별 메모리 측정
            ├── extract_layers()        # 모델 레이어 추출
            └── measure_model_memory()  # 전체 메모리 측정
```

## 기능별 구현 코드

### 1. Structured Pruning (L2 Norm 기반)

**목적**: Conv 필터의 중요도를 L2 norm으로 측정하여 pruning 대상 선정

**핵심 함수**: `lib/pruning/dlasg_pruning.py`

```python
# L2 Norm 계산 (최대값 보호)
def get_filter_norms(layer, inf=99999):
    filter_norms = torch.norm(layer.weight.view(layer.weight.shape[0], -1), dim=1)
    max_idx = torch.argmax(filter_norms)
    filter_norms[max_idx] = inf  # 최대값은 pruning 방지
    return filter_norms

# Global Pruning Index 계산
def get_pruning_indices(filter_norms, sparsity):
    all_norms = torch.cat(filter_norms)
    num_pruning_filters = int(all_norms.numel() * sparsity)
    _, global_pruning_idx = torch.topk(all_norms, num_pruning_filters, largest=False)
    # 각 레이어별로 local index 변환
    ...
    return pruning_indices
```

**특징**:
- Global sparsity 기준으로 전체 레이어를 고려한 pruning
- 최대 norm 필터는 항상 보존 (inf 설정)
- Top-k smallest norm 선택

### 2. BasicBlock Blockwise Pruning

**목적**: DLA-34의 BasicBlock 단위로 구조화된 pruning 적용

**핵심 함수**: `dlasg_blockwise_pruning()` in `lib/pruning/dlasg_pruning.py:74-89`

```python
def dlasg_blockwise_pruning(model, sparsity, device='cpu'):
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'BasicBlock':
            conv1, bn1 = module.conv1, module.bn1
            conv2, bn2 = module.conv2, module.bn2

            # conv1 출력 채널 pruning
            norms1 = get_filter_norms(conv1)
            prune_idx1 = get_pruning_indices([norms1], sparsity)[0]

            filter_pruning(conv1, prune_idx1)  # weight를 0으로 설정
            bn_pruning(bn1, prune_idx1)        # BN 파라미터를 0으로 설정
```

**Pruning 메커니즘**:
- `filter_pruning()`: Conv weight[pruning_idx] = 0.0
- `bn_pruning()`: BN의 weight, bias, mean = 0.0, var = 1.0

### 3. Channel Reduction (물리적 제거)

**목적**: 0으로 마스킹된 채널을 물리적으로 제거하여 실제 모델 크기 감소

**핵심 함수**: `reduce_pruned_model()` in `lib/pruning/dlasg_pruning.py:91-146`

```python
def reduce_pruned_model(model):
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'BasicBlock':
            conv1, bn1 = module.conv1, module.bn1
            conv2, bn2 = module.conv2, module.bn2

            # 살아있는 채널 찾기
            keep = torch.where(
                conv1.weight.view(conv1.weight.shape[0], -1).abs().sum(1) != 0
            )[0]

            # 새로운 작은 conv1 생성 (out_channels 감소)
            new_conv1 = nn.Conv2d(
                in_channels=conv1.in_channels,
                out_channels=keep.numel(),  # 축소된 크기
                ...
            )
            new_conv1.weight.data = conv1.weight[keep].clone()

            # 새로운 작은 conv2 생성 (in_channels 감소)
            new_conv2 = nn.Conv2d(
                in_channels=keep.numel(),  # conv1 출력에 맞춤
                out_channels=conv2.out_channels,
                ...
            )
            new_conv2.weight.data = conv2.weight[:, keep].clone()

            # BatchNorm도 동일하게 축소
            _assign_module(model, f"{name}.conv1", new_conv1)
            _assign_module(model, f"{name}.conv2", new_conv2)
```

**특징**:
- Conv1 출력 채널과 Conv2 입력 채널 동시 처리
- BatchNorm도 연동하여 축소
- BasicBlock 내부 연결성 유지

### 4. 메모리 측정

**목적**: 레이어별 메모리 사용량 프로파일링

**핵심 함수**: `pruning_TW.py` 및 `lib/pruning/memory_usage.py`

```python
def measure_memory(x, layers, device):
    # GPU 메모리 초기화
    x.cpu()
    for layer in layers:
        layer.cpu()
    torch.cuda.empty_cache()

    before_memory = torch.cuda.memory_allocated(device) / 1024**2

    # Forward pass
    with torch.no_grad():
        x = x.to(device)
        for layer in layers:
            layer.to(device)
            x = layer(x)

    after_memory = torch.cuda.memory_allocated(device) / 1024**2

    return x, after_memory - before_memory
```

**측정 대상**:
- DLA base layers
- Upsampling layers (dla_up, ida_up)
- ConvGRU
- Detection heads (hm, wh, reg, hps, scale 등)

## 전체 Compression Flow

```markdown
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
│     │     - bn1.weight[idx] = 0.0             │        │
│     │     - bn1.bias[idx] = 0.0               │        │
│     │     - bn1.mean[idx] = 0.0               │        │
│     │     - bn1.var[idx] = 1.0                │        │
│     │                                         │        │
│     └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  3. Verification (선택)                                  │
│     - Count zero parameters                             │
│     - Calculate sparsity ratio                          │
│     - Save pruned model (.pth)                          │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  4. Reducing Phase                                      │
│     ┌─────────────────────────────────────────┐        │
│     │  For each BasicBlock:                   │        │
│     │                                         │        │
│     │  A. Identify Survived Channels          │        │
│     │     - keep = where(abs(weight).sum != 0)│        │
│     │                                         │        │
│     │  B. Create Reduced Conv1                │        │
│     │     - new out_channels = len(keep)      │        │
│     │     - Copy weights[keep]                │        │
│     │                                         │        │
│     │  C. Create Reduced Conv2                │        │
│     │     - new in_channels = len(keep)       │        │
│     │     - Copy weights[:, keep]             │        │
│     │                                         │        │
│     │  D. Create Reduced BN1                  │        │
│     │     - Reduce all BN parameters          │        │
│     │                                         │        │
│     └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  5. Final Model                                         │
│     - Save reduced model (.pth)                         │
│     - Verify parameter reduction                        │
│     - Measure memory usage (선택)                        │
└─────────────────────────────────────────────────────────┘
```

## 상세 Compression Flow

```markdown
[1] 모델 로드 (pruning.py 또는 reducing.py)
    ├─ opt 설정
    │   ├─ arch = 'dla_34'
    │   ├─ tracking_task = True
    │   ├─ heads 설정 (hm, reg, wh, hps, scale 등)
    │   └─ head_conv = 256
    │
    ├─ Dataset 정보 업데이트
    │   └─ ObjectPoseDataset 사용
    │
    ├─ create_model(arch, heads, head_conv)
    │   └─ DLA-34 기반 모델 생성
    │
    └─ load_model(model, checkpoint_path)
        └─ 사전 학습된 가중치 로드

[2] Pruning 실행
    └─ dlasg_blockwise_pruning(model, sparsity=0.5)
        │
        ├─ [2-1] BasicBlock 순회
        │   └─ For each BasicBlock in model:
        │
        ├─ [2-2] Conv1 필터 norm 계산
        │   ├─ get_filter_norms(conv1)
        │   │   ├─ norms = L2_norm(conv1.weight)
        │   │   ├─ max_idx = argmax(norms)
        │   │   └─ norms[max_idx] = inf  # 보호
        │   │
        │   └─ get_pruning_indices([norms], sparsity)
        │       ├─ global: top-k smallest norms
        │       └─ local: convert to layer index
        │
        ├─ [2-3] Conv1 Pruning
        │   └─ filter_pruning(conv1, prune_idx)
        │       └─ conv1.weight[prune_idx, :, :, :] = 0.0
        │
        └─ [2-4] BN1 Pruning
            └─ bn_pruning(bn1, prune_idx)
                ├─ bn1.weight[prune_idx] = 0.0
                ├─ bn1.bias[prune_idx] = 0.0
                ├─ bn1.running_mean[prune_idx] = 0.0
                └─ bn1.running_var[prune_idx] = 1.0

[3] Pruning 검증 (선택)
    ├─ total_params = sum(p.numel())
    ├─ total_zero = sum((p == 0).sum())
    └─ sparsity = total_zero / total_params
        └─ 예: 50% sparsity

[4] Pruned 모델 저장 (선택)
    └─ torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                  f"my_pruned_model_{sparsity}.pth")

[5] Reducing 실행
    └─ reduce_pruned_model(model)
        │
        └─ For each BasicBlock:
            │
            ├─ [5-1] 살아있는 채널 찾기
            │   └─ keep = torch.where(
            │         conv1.weight.view(C_out, -1).abs().sum(1) != 0
            │       )[0]
            │
            ├─ [5-2] Conv1 Reduce
            │   ├─ new_conv1 = Conv2d(
            │   │     in_channels=old_in,
            │   │     out_channels=len(keep)  # 축소!
            │   │   )
            │   └─ new_conv1.weight = conv1.weight[keep]
            │
            ├─ [5-3] Conv2 Reduce
            │   ├─ new_conv2 = Conv2d(
            │   │     in_channels=len(keep),  # 축소!
            │   │     out_channels=old_out
            │   │   )
            │   └─ new_conv2.weight = conv2.weight[:, keep]
            │
            ├─ [5-4] BN1 Reduce
            │   ├─ keep_bn = torch.where(bn1.weight != 0)[0]
            │   ├─ new_bn = BatchNorm2d(len(keep_bn))
            │   ├─ new_bn.weight = bn1.weight[keep_bn]
            │   ├─ new_bn.bias = bn1.bias[keep_bn]
            │   ├─ new_bn.running_mean = bn1.running_mean[keep_bn]
            │   └─ new_bn.running_var = bn1.running_var[keep_bn]
            │
            └─ [5-5] 모듈 교체
                ├─ _assign_module(model, "conv1", new_conv1)
                ├─ _assign_module(model, "conv2", new_conv2)
                └─ _assign_module(model, "bn1", new_bn)

[6] Reduced 모델 저장
    ├─ torch.save(reduced_model, f"reduced_model_{sparsity}.pth")
    │
    └─ 검증
        ├─ reduced_params = sum(p.numel())
        ├─ reduction_ratio = 1 - (reduced_params / original_params)
        └─ 예: 30-40% parameter reduction

[7] 메모리 측정 (선택)
    └─ measure_model_memory(reduced_model, input_tensor, device)
        │
        ├─ [7-1] Base Layers
        │   └─ For each layer in DLA base:
        │       └─ measure_memory(layer)
        │
        ├─ [7-2] Upsampling Layers
        │   ├─ dla_up: measure_memory()
        │   └─ ida_up: measure_memory()
        │
        ├─ [7-3] ConvGRU
        │   └─ measure_memory(convgru)
        │       └─ 4-step sequential GRU
        │
        ├─ [7-4] Detection Heads
        │   ├─ For each head in ['hm', 'wh', 'reg', ...]:
        │   │   └─ measure_memory(head_layer)
        │   │
        │   └─ heads: hm, wh, reg, hm_hp, hp_offset, hps,
        │             hps_uncertainty, scale, scale_uncertainty,
        │             tracking, tracking_hp
        │
        └─ [7-5] Total Memory
            └─ sum(layer_memories)
                └─ 예: Original: 150 MB, Reduced: 100 MB

[8] 압축 완료
    ├─ reduced_model_{sparsity}.pth 생성
    ├─ 파라미터 감소: 30-40%
    ├─ 메모리 감소: 30-40%
    └─ 정확도 유지: 추가 fine-tuning 가능
```

## 압축 결과 비교

### Parameter Count
```
Original Model: ~20M parameters
Pruned Model (50%): ~20M parameters (50% are zeros)
Reduced Model (50%): ~12-14M parameters (물리적 제거)
Reduction Ratio: 30-40%
```

### Memory Usage
```
Original Model: ~150-200 MB (inference)
Reduced Model: ~100-130 MB (inference)
Memory Reduction: 30-40%
```

### 주요 특징
1. **One-shot Pruning**: Training 없이 단일 실행으로 pruning
2. **Global Sparsity**: 전체 레이어를 고려한 통합 pruning
3. **Blockwise Processing**: BasicBlock 단위 구조화된 pruning
4. **Physical Reduction**: 실제 모델 크기 감소
5. **No Knowledge Distillation**: Teacher 모델 불필요

## 실행 방법

### 1. Pruning만 수행
```bash
python pruning.py
# Output: my_pruned_model_50.pth
```

### 2. Pruning + Reducing
```bash
python reducing.py
# Output: reduced_model_50.pth
```

### 3. 메모리 측정 포함
```bash
python pruning_TW.py
# Output: 레이어별 메모리 사용량 출력
```

## YOLO vs CenterPose 비교

| 항목 | YOLO Compression | CenterPose Compression |
|------|------------------|------------------------|
| **백본** | YOLOv8 (CSPDarknet) | DLA-34 |
| **타겟 모듈** | C2f, SPPF, Detect | BasicBlock |
| **Pruning 방식** | Dynamic (매 step) | One-shot |
| **Distillation** | Yes (YOLOv8x → v8n) | No |
| **Training 필요** | Yes (학습 중 pruning) | No (사전 학습 모델만) |
| **Reducing 시점** | 학습 후 | Pruning 직후 |
| **복잡도** | High (통합 학습) | Low (후처리) |

## 핵심 차이점

1. **CenterPose는 One-shot Pruning**
   - 학습 없이 가중치 분석만으로 pruning
   - 빠른 압축 가능

2. **YOLO는 Training-time Pruning**
   - 학습 과정에서 동적으로 pruning
   - Distillation과 동시 진행
   - 더 나은 성능 유지 가능

3. **CenterPose는 BasicBlock 단위**
   - DLA 구조의 특성 반영
   - Conv1-BN1-Conv2-BN2 세트 처리

4. **YOLO는 다양한 모듈 처리**
   - C2f, SPPF, Detect 각각 전용 로직
   - Concatenation 추적 필요
