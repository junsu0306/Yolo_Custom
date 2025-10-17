## Yolo 압축 개요

1. Pruning with Distillation 압축 기법 사용
2. Pruning: L2-norm based One-shot Structured Filter Pruning
3. Distillation: YOLOv8x(Teacher Model), Yolov8n(Student  Model)

## 수정 및 추가된 파일 목록

```markdown
ultralytics_custom/
├── cfg/
│   ├── **default.yaml**                    # 압축 관련 파라미터 추가
│   └── __init__.py                     # 파라미터 등록
│
├── engine/
│   ├── **trainer.py**                      
│   │   ├── [Lines 11-18]   Import 추가
│   │   ├── [Lines 67-75]   Loss 함수 추가
│   │   ├── [Lines 264-272] Teacher 모델 로드 및 고정
│   │   └── [Lines 429-492] Training loop 수정
│   │
│   ├── **compression.py**                
│   │   ├── yolov8_pruning()           # L2 norm 기반 필터 선택 및 0으로 설정
│   │   └── yolov8_reducing()          # 0인 필터 물리적 제거
│   │
│   └── compression_src/                # 범용 압축 모듈 추가
│       ├── **pruning/common.py**
│       │   ├── get_filter_pruning_idx()
│       │   ├── filter_pruning()
│       │   └── bn_pruning()
│       └── **reducing/common.py**
│           ├── get_survived_filter_idx()
│           ├── conv_reduce()
│           └── bn_reduce()
│
└── utils/
    └── **memory_usage_MH.py**              # 메모리 측정 추가
```

## 기능별 구현 코드

1. Structured Pruning
   - L2 Norm 기반 필터 선택
   - Dynamic Pruning (매 step 실행)
   - BN 동시 마스킹
   - 코드: `compression.py`, `pruning/common.py`

2. Channel Reduction
   - 마스킹된 필터 제거
   - Skip Connection 추적
   - 코드: `compression.py`, `reducing/common.py`

3. Knowledge Distillation
   - KL Divergence (Classification)
   - MSE Loss (Bbox)
   - 코드: `trainer.py`

4. Training Loop 통합
   - Simultaneous 실행
   - Loss 및 Backward 통합
   - 코드: `trainer.py`

## 전체 Training Flow

```markdown
┌─────────────────────────────────────────────────────────┐
│                    Training Start                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  1. Setup Phase (_setup_train)                          │
│     - Load teacher model                                │
│     - Freeze teacher parameters                         │
│     - Convert to FP16                                   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────┐
│  2. Training Loop (_do_train)                          │
│     ┌───────────────────────────────────────────┐      │
│     │  For each batch:                          │      │
│     │                                           │      │
│     │  A. Forward Pass                          │      │
│     │     - Student: model(batch, distil=True)  │      │
│     │     - Teacher: teacher_model(batch)       │      │
│     │                                           │      │
│     │  B. Loss Calculation                      │      │
│     │     - Task loss (detection)               │      │
│     │     - KD loss (cls + bbox)                │      │
│     │     - Memory loss                         │      │
│     │                                           │      │
│     │  C. Backward Pass                         │      │
│     │     - Compute gradients                   │      │
│     │                                           │      │
│     │  D. Optimization                          │      │
│     │     - Update weights                      │      │
│     │     - Pruning                             │      │
│     │                                           │      │
│     └───────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  3. Validation & Checkpointing                          │
│     - Evaluate compressed model                         │
│     - Save best checkpoint                              │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  4. Final Reducing (after training)                     │
│     - yolov8_reducing()                                 │
│     - Create smaller model                              │
│     - Save compressed .pt file                          │
└─────────────────────────────────────────────────────────┘
```

## 상세 Training Flow

```markdown
[1] 설정 파일 로드
    ├─ cfg/default.yaml 읽기
    ├─ pruning_ratio: 0.3
    ├─ distill_ratio: 0.5 (가정)
    └─ mem_usg: 100.0 (가정)

[2] Trainer 생성 및 초기화
    ├─ BaseTrainer.__init__()
    └─ _setup_train() 호출
        ├─ [2-1] Teacher 모델 로드
        │   ├─ YOLO('...yolov8x.../best.pt')
        │   ├─ teacher_model.eval()
        │   ├─ requires_grad = False
        │   └─ teacher_model.half()
        │
        ├─ [2-2] Student 모델 로드
        │   ├─ setup_model()
        │   └─ model.to(device)
        │
        ├─ [2-3] Dataloader 생성
        └─ [2-4] Optimizer 생성

[3] Training Loop 시작 (_do_train)
    └─ For epoch in range(epochs):
        └─ For batch_idx, batch in enumerate(train_loader):

            [3-1] Forward Pass
                ├─ autocast 활성화
                ├─ 메모리 측정
                │   └─ device_condition_mem = torch.cuda.max_memory_allocated()
                │
                ├─ Student Forward
                │   └─ loss, loss_items, feats = model(batch, distillation=True)
                │
                ├─ Teacher Forward (if epoch > 0)
                │   ├─ with torch.no_grad():
                │   │   └─ teacher_out = teacher_model.model(batch['img'])
                │   │
                │   ├─ Feature 추출
                │   │   ├─ stu_result = feats[0]
                │   │   └─ teacher_result = teacher_out[0]
                │   │
                │   ├─ Classification KD Loss
                │   │   ├─ s_cls = F.log_softmax(stu_result[:, 4:, :])
                │   │   ├─ t_cls = F.softmax(teacher_result[:, 4:, :])
                │   │   └─ loss_cls = F.kl_div(s_cls, t_cls)
                │   │
                │   ├─ Bbox Regression Loss
                │   │   └─ loss_bbox = F.mse_loss(stu[:, 0:4], tea[:, 0:4])
                │   │
                │   ├─ Total KD Loss
                │   │   └─ kd_loss = (loss_cls + loss_bbox) * distill_ratio
                │   │
                │   └─ Memory Loss
                │       └─ mem_loss = max(0, target - current) * 0.5
                │
                └─ Loss 통합
                    └─ total_loss = task_loss + kd_loss + mem_loss

            [3-2] Backward Pass
                └─ scaler.scale(loss).backward()

            [3-3] Optimizer Step (if accumulation done)
                ├─ optimizer_step()
                │
                └─ [3-4]  Pruning  (if pruning_ratio > 0)
                    └─ yolov8_pruning(model, sparsity=0.3)
                        ├─ For each block (0-22):
                        │   ├─ Conv:
                        │   │   ├─ pruning_idx = get_filter_pruning_idx(block.conv, 0.3)
                        │   │   │   ├─ filter_norms = torch.norm(weight)
                        │   │   │   └─ topk(norm, largest=False)
                        │   │   ├─ filter_pruning(block.conv, pruning_idx)
                        │   │   │   └─ weight[pruning_idx] = 0.0
                        │   │   └─ bn_pruning(block.bn, pruning_idx)
                        │   │       ├─ weight[pruning_idx] = 0.0
                        │   │       ├─ bias[pruning_idx] = 0.0
                        │   │       ├─ mean[pruning_idx] = 0.0
                        │   │       └─ var[pruning_idx] = 1.0
                        │   │
                        │   ├─ C2f/SPPF:
                        │   │   └─ (cv2에 대해 동일 작업)
                        │   │
                        │   └─ Detect:
                        │       └─ For 3 scales × 2 Conv:
                        │           └─ (cv2, cv3에 대해 동일 작업)

            다음 batch

        [4] Epoch 종료
            └─ Validation
                └─ metrics = model.val()

[5] Training 종료

[6] (선택) Reducing 실행
    └─ yolov8_reducing(model, reduced_model)
        ├─ For each layer:
        │   ├─ survived_idx = get_survived_filter_idx(layer)
        │   │   └─ torch.where(filter_norms != 0)
        │   │
        │   ├─ conv_reduce(layer, reduced_layer, survived_idx)
        │   │   ├─ 새 레이어 생성 (작은 크기)
        │   │   └─ 살아남은 가중치만 복사
        │   │
        │   └─ bn_reduce(bn, reduced_bn, survived_idx)
        │
        ├─ Concatenation 레이어 (11, 14, 17, 20) 특별 처리
        │   ├─ 두 입력의 survived_idx 가져오기
        │   ├─ offset 추가 (두 번째 입력)
        │   └─ concat
        │
        └─ 축소된 모델 반환

[7] 모델 저장
    └─ reduced_model.save('compressed.pt')
```