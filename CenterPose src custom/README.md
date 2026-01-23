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

