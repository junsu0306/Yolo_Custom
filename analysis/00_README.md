# YOLOv8 AI 모델 압축 분석 문서

## 개요

이 폴더는 `ultralytics_custom`에 구현된 YOLOv8 신경망 모델 압축 기법에 대한 상세 분석을 담고 있습니다.

## 압축 기법 요약

### 방법론: Structured Pruning + Channel Reduction + Knowledge Distillation

1. **Phase 1: Pruning (가지치기)**
   - L2 Norm 기반 Filter Pruning
   - 중요하지 않은 필터의 가중치를 0으로 설정
   - 학습 중 동적으로 pruning (매 optimizer step 후)

2. **Phase 2: Reducing (축소)**
   - 0이 된 필터를 물리적으로 제거
   - 실제 모델 크기 감소
   - 메모리 절약 및 속도 향상

3. **Knowledge Distillation (지식 증류)**
   - Teacher 모델(큰 모델)의 지식을 Student(압축 모델)에게 전달
   - Classification: KL Divergence
   - Bbox Regression: MSE Loss
   - 압축으로 인한 정확도 손실 최소화

4. **Memory-aware Training**
   - 목표 메모리 사용량 설정
   - 초과 시 loss에 패널티 추가
   - 자동으로 메모리 제약 충족

### 결과 (예상)
- **모델 크기**: ~25-30% 감소 (sparsity=0.3 기준)
- **추론 속도**: ~20-30% 향상
- **정확도**: ~2-3% 손실 → Distillation으로 회복

---

## 문서 구조

### 📘 [01_model_compression_overview.md](./01_model_compression_overview.md)
**Phase 1: Pruning 상세 분석**

- Magnitude-based Filter Pruning 원리
- L2 Norm 계산 방법
- Batch Normalization 프루닝
- YOLOv8 Backbone/Head 프루닝 구현
- Structured vs Unstructured Pruning 비교

**핵심 내용**:
```python
# 가장 중요하지 않은 필터 찾기
filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)
_, pruning_idx = torch.topk(filter_norms, num_pruning_filters, largest=False)

# 필터를 0으로 만들기
weight[pruning_idx, :, :, :] = 0.0
```

---

### 📗 [02_model_reducing_analysis.md](./02_model_reducing_analysis.md)
**Phase 2: Reducing 상세 분석**

- 살아남은 필터 식별
- Conv/BN Layer 축소
- YOLOv8 Architecture-aware 압축
- Skip Connection (Concatenation) 처리
- Detect Head 특별 처리

**핵심 내용**:
```python
# 0이 아닌 필터만 선택
survived_idx = torch.where(filter_norms != 0)[0]

# 작은 모델에 복사
reduced_weight.copy_(
    weight[survived_out_channels_idx, :, :, :][
        :, survived_in_channels_idx, :, :
    ]
)
```

**특별 처리**:
- Concatenation 레이어 (11, 14, 17, 20번)
- Multi-scale Detect head (P3, P4, P5)

---

### 📙 [03_code_integration_guide.md](./03_code_integration_guide.md)
**기존 코드 통합 가이드 - NEW!**

원본 `ultralytics_github` 코드에 압축 기능을 **어떻게 통합했는지** 상세히 설명:

#### 주요 내용

1. **설정 파일 수정**
   - `cfg/default.yaml`: `pruning_ratio`, `mem_usg` 파라미터 추가
   - `cfg/__init__.py`: 새 파라미터 등록

2. **Trainer 수정 - Import**
   - `model_compression.funcs4`: 외부 압축 라이브러리
   - `torch.nn.functional as F`: Distillation loss
   - `memory_usage_MH`: 메모리 측정

3. **Trainer 수정 - Loss 함수**
   ```python
   def distillation_loss(student_logits, teacher_logits, T=2.0):
       s = student_logits / T
       t = teacher_logits / T
       return F.kl_div(F.log_softmax(s, dim=-1),
                       F.softmax(t, dim=-1),
                       reduction='batchmean') * T * T
   ```

4. **Trainer 수정 - Teacher 모델 초기화**
   ```python
   # _setup_train() 메서드에 추가
   self.teacher_model = YOLO('yolov8x.pt').to(self.device)
   for p in self.teacher_model.parameters():
       p.requires_grad = False
   self.teacher_model.half()
   ```

5. **Trainer 수정 - Training Loop**
   ```python
   # Forward: Student + Teacher
   self.loss, self.loss_items, feats = self.model(batch, distillation=True)
   teacher_out = self.teacher_model.model(batch['img'])

   # KD Loss
   loss_cls = F.kl_div(student_cls, teacher_cls)
   loss_bbox = F.mse_loss(student_bbox, teacher_bbox)
   total_kd_loss = (loss_cls + loss_bbox) * self.args.distill_ratio

   # Memory Loss
   self.loss += max(0, target_mem - current_mem) * hyperparam

   # Optimize + Pruning
   self.optimizer_step()
   yolov8_pruning(self.model.model, self.args.pruning_ratio)
   ```

#### 전체 Loss 구성
```python
total_loss = task_loss + kd_loss + memory_loss

여기서:
- task_loss: YOLO detection loss
- kd_loss = (cls_loss + bbox_loss) * distill_ratio
- memory_loss = max(0, target_mem - current_mem) * hyperparam
```

#### 코드 통합 체크리스트
- ✅ 설정 파일에 파라미터 추가
- ✅ Import 추가
- ✅ Teacher 모델 초기화
- ✅ Training loop에 distillation 추가
- ✅ Optimizer step 후 pruning 호출
- ✅ 압축 모듈 (`compression.py`) 생성

---

## 주요 개념

### 1. Structured Pruning

**장점**:
- ✅ 하드웨어 친화적 (일반 GPU/CPU에서 효율적)
- ✅ 실제 속도 향상
- ✅ Dense tensor 연산 유지

**단점**:
- ❌ Unstructured보다 압축률 낮음
- ❌ 단위가 크기 때문에 정교한 조절 어려움

### 2. Magnitude-based Pruning

**원리**: L2 Norm이 작은 필터 = 덜 중요

```
||W_i||_2 = sqrt(Σ w_ij^2)
```

**장점**:
- ✅ 간단한 구현
- ✅ 계산 비용 낮음
- ✅ 추가 학습 불필요 (importance 계산)

**단점**:
- ❌ 최적이 아닐 수 있음
- ❌ Layer-wise 방식 (Global optimal 아님)

### 3. Knowledge Distillation

**원리**: Teacher의 soft label을 Student에게 전달

```
Distillation Loss = KL_Div(Student || Teacher)

Temperature Scaling:
- T = 1: Hard label (원래 확률)
- T > 1: Soft label (부드러운 확률)
```

**장점**:
- ✅ 압축 모델의 정확도 향상
- ✅ Teacher의 암묵적 지식 학습
- ✅ Generalization 개선

**YOLO에서의 적용**:
```python
# Classification distillation (KL Divergence)
loss_cls = KL_Div(student_cls[:, 4:], teacher_cls[:, 4:])

# Bbox distillation (MSE)
loss_bbox = MSE(student_bbox[:, 0:4], teacher_bbox[:, 0:4])
```

### 4. Memory-aware Training

**원리**: 메모리 사용량을 loss에 반영

```python
memory_loss = max(0, target_memory - current_memory) * weight
```

**효과**:
- 자동으로 메모리 제약 충족
- Edge device 배포 시 유용

### 5. Architecture-Aware Compression

**YOLOv8의 복잡성**:
```
Skip Connections:
  Layer 8 ─┐
           ├─ concat ─→ Layer 11
  Layer 5 ─┘

Multi-scale Detection:
  P3 (small)  ─→ Detect[0]
  P4 (medium) ─→ Detect[1]
  P5 (large)  ─→ Detect[2]
```

**해결책**:
- Concatenation 지점 명시적 처리
- 인덱스 offset 계산
- Detect head 특별 처리

### 6. Dynamic Pruning

**일반 Pruning**: 학습 → Pruning → Fine-tuning (3단계)

**Dynamic Pruning**: 학습 중 지속적으로 pruning
```python
for epoch in epochs:
    for batch in dataloader:
        forward()
        backward()
        optimizer_step()
        pruning()  # ← 매번 실행!
```

**장점**:
- Pruning과 fine-tuning 동시 진행
- 더 빠른 수렴
- 최종 성능 향상

---

## 코드 구조

```
ultralytics_custom/
├── cfg/
│   ├── default.yaml                        # 설정 (pruning_ratio, mem_usg 추가)
│   └── __init__.py                         # 파라미터 등록
│
├── engine/
│   ├── trainer.py                          # 핵심 수정 파일
│   │   ├── distillation_loss()            # KD loss 함수
│   │   ├── MSE_loss()                     # Feature matching
│   │   ├── _setup_train()                 # Teacher 초기화
│   │   └── _do_train()                    # Training loop (KD + Pruning)
│   │
│   ├── compression.py                      # YOLOv8 전용 압축
│   │   ├── yolov8_pruning()               # Phase 1
│   │   └── yolov8_reducing()              # Phase 2
│   │
│   └── compression_src/
│       ├── pruning/
│       │   └── common.py                   # 범용 프루닝 함수
│       │       ├── get_filter_pruning_idx()
│       │       ├── filter_pruning()
│       │       └── bn_pruning()
│       │
│       └── reducing/
│           └── common.py                   # 범용 축소 함수
│               ├── get_survived_filter_idx()
│               ├── conv_reduce()
│               ├── bn_reduce()
│               └── fc_reduce()
│
└── utils/
    └── memory_usage_MH.py                  # 메모리 측정
```

---

## 사용 예제

### 기본 사용법

```python
from ultralytics_custom import YOLO

# 1. 모델 로드
model = YOLO('yolov8n.pt')

# 2. 학습 (자동으로 pruning + distillation)
model.train(
    data='coco.yaml',
    epochs=100,
    pruning_ratio=0.3,      # 30% pruning
    distill_ratio=0.5,      # 50% KD weight
    mem_usg=100.0           # Target 100MB
)

# 3. 압축된 모델 저장
model.save('yolov8n_compressed.pt')
```

### YAML 설정

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
distill_ratio: 0.5      # 50% KD weight
mem_usg: 100.0          # Target 100MB
```

```bash
yolo train model=yolov8n.pt data=coco8.yaml \
  pruning_ratio=0.3 distill_ratio=0.5 mem_usg=100
```

### Sparsity 선택

```python
# 약한 압축 (안전)
model.train(data='coco.yaml', pruning_ratio=0.1)  # 10% 제거

# 중간 압축 (균형)
model.train(data='coco.yaml', pruning_ratio=0.3)  # 30% 제거

# 강한 압축 (공격적)
model.train(data='coco.yaml', pruning_ratio=0.5)  # 50% 제거
```

---

## 주요 파일 위치

### 압축 코드
```
/Users/junsu/Projects/Yolo_Custom/ultralytics_custom/engine/compression.py
/Users/junsu/Projects/Yolo_Custom/ultralytics_custom/engine/compression_src/pruning/common.py
/Users/junsu/Projects/Yolo_Custom/ultralytics_custom/engine/compression_src/reducing/common.py
/Users/junsu/Projects/Yolo_Custom/ultralytics_custom/engine/trainer.py (Lines 11-18, 67-75, 264-272, 429-492)
```

### 설정 파일
```
/Users/junsu/Projects/Yolo_Custom/ultralytics_custom/cfg/default.yaml (Lines 16-19)
/Users/junsu/Projects/Yolo_Custom/ultralytics_custom/cfg/__init__.py
```

### 원본 비교용
```
/Users/junsu/Projects/Yolo_Custom/ultralytics_github/
```

---

## 추가 분석 가능 주제

### 🔬 고급 주제

1. **Layer-wise Sparsity**
   - 레이어마다 다른 sparsity 적용
   - 중요한 레이어는 덜 프루닝

2. **Sensitivity Analysis**
   - 각 레이어의 프루닝 민감도 분석
   - 어떤 레이어가 정확도에 중요한가?

3. **다른 Pruning 방법 비교**
   - Magnitude-based (현재)
   - Gradient-based
   - Hessian-based
   - Taylor expansion

4. **Quantization 결합**
   - Pruning + Quantization
   - 더 높은 압축률

5. **Progressive Distillation**
   - 여러 단계의 teacher 사용
   - Large → Medium → Small

### 📊 실험 가능 주제

1. **Sparsity vs Accuracy Trade-off**
   - 다양한 sparsity에서 성능 측정
   - 최적 sparsity 찾기

2. **Distillation Ratio 실험**
   - distill_ratio: 0.0 ~ 1.0
   - 최적 가중치 찾기

3. **Teacher 모델 선택**
   - Self-distillation vs Larger teacher
   - 어느 것이 더 효과적?

4. **동적 vs 정적 Pruning**
   - Dynamic: 학습 중 pruning
   - Static: 학습 후 pruning
   - 성능 비교

5. **속도 벤치마크**
   - 다양한 하드웨어에서 측정
   - GPU vs CPU vs Edge device

---

## 참고 자료

### 논문

**Pruning**:
- **Pruning Filters for Efficient ConvNets** (Li et al., 2017)
- **Learning Efficient Convolutional Networks through Network Slimming** (Liu et al., 2017)
- **ThiNet: A Filter Level Pruning Method** (Luo et al., 2017)

**Knowledge Distillation**:
- **Distilling the Knowledge in a Neural Network** (Hinton et al., 2015)
- **FitNets: Hints for Thin Deep Nets** (Romero et al., 2015)

**Joint Compression**:
- **Learning Efficient Object Detection Models with KD** (Chen et al., 2017)

### 관련 기술
- Network Pruning
- Knowledge Distillation
- Quantization
- Neural Architecture Search (NAS)
- Low-rank Factorization

---

## 결론

이 커스텀 버전은 **종합적인 모델 압축 프레임워크**를 제공합니다:

✅ **구현 완성도**
- Pruning, Reducing, Distillation 모두 구현
- YOLOv8 아키텍처 완전 지원
- Skip connection 올바르게 처리
- Memory-aware training

✅ **사용성**
- 간단한 API (YAML 설정)
- 모듈화된 구조
- 다른 모델 확장 가능

✅ **학습 통합**
- 동적 pruning (학습 중)
- Knowledge distillation 자동 적용
- Fine-tuning 동시 진행

⚠️ **주의사항**
- Teacher 모델 경로 하드코딩 (수정 필요)
- 외부 의존성 (`model_compression.funcs4`)
- Distillation mode 지원 필요 (model.py 수정)

💡 **활용 시나리오**
- Edge device 배포
- 실시간 추론 요구사항
- 메모리 제약 환경
- 대량 배치 처리

---

## 질문이 있다면

1. 특정 부분 상세 분석 요청
2. 코드 통합 관련 질문
3. 실험 결과 해석
4. 다른 압축 기법 적용
5. 성능 최적화 방법

언제든지 요청하세요!
