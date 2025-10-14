1. 시작: 사용자가 학습 명령 실행

  yolo train model=yolov8n.pt data=coco.yaml pruning_ratio=0.3
  distill_ratio=0.5

  이 명령이 들어오면:

  ---
  2. 설정 로드 (cfg/default.yaml)

⏺ Read(ultralytics_custom/cfg/default.yaml)
  ⎿  Read 25 lines

⏺ Lines 17-18이 핵심입니다:
  pruning_ratio: 0.0  # 기본값 0 = 압축 안 함
  mem_usg: 0.0        # 목표 메모리

  명령어에서 pruning_ratio=0.3을 주면 이 값이 0.3으로 오버라이드됩니다.

  ---
  3. Trainer 초기화 - Teacher 모델 로드

⏺ Read(ultralytics_custom/engine/trainer.py)
  ⎿  Read 20 lines

⏺ 이 부분이 초기화 시점입니다:

  # Line 266: Teacher 모델 로드 (사전 학습된 큰 모델)
  self.teacher_model =
  YOLO('/workspace/TW/YOLO/.../best.pt').to(self.device)

  # Line 268-269: Teacher는 학습 안 함 (파라미터 고정)
  for p in self.teacher_model.parameters():
      p.requires_grad = False

  # Line 271: 메모리 절약을 위해 FP16으로 변환
  self.teacher_model.half()

  그 다음 (Line 277-278):
  # Student 모델 로드 (압축할 모델)
  self.model = self.model.to(self.device)

  결과:
  - self.teacher_model: 큰 모델 (YOLOv8x), 고정됨, 추론만
  - self.model: 작은 모델 (YOLOv8n), 학습됨, 압축 대상

  ---
  4. Training Loop 시작

  매 배치마다 이런 일이 일어납니다:

⏺ Read(ultralytics_custom/engine/trainer.py)
  ⎿  Read 70 lines

⏺ 좋습니다! 이제 핵심 부분을 순서대로 설명하겠습니다.

  Step 1: Student Forward (Line 436)

  self.loss, self.loss_items, feats = self.model(batch, distillation=True)

  무슨 일이?
  - Student 모델(압축할 모델)이 이미지를 처리
  - distillation=True: 중간 feature도 반환하라는 의미
  - 반환값:
    - self.loss: 원래 YOLO detection loss
    - self.loss_items: loss 항목들
    - feats: 중간 feature (KD에 사용)

  Step 2: Teacher Forward (Lines 439-449)

  with torch.no_grad():  # Gradient 계산 안 함
      # Teacher가 같은 이미지 처리
      teacher_out = self.teacher_model.model(batch['img'])

      # Student도 한번 더 (eval mode로)
      self.model.eval()
      stu_out = self.model(batch['img'])
      self.model.train()

      # 출력 추출
      teacher_result = teacher_out[0]  # [batch, 84, 8400]
      stu_result = stu_out[0]

  왜 Student를 두 번 실행?
  - Line 436: train mode로 실행 → loss 계산용
  - Line 445: eval mode로 실행 → Teacher와 비교용 (BN, Dropout 동작 다름)

  Step 3: Knowledge Distillation Loss (Lines 450-458)

  # Classification KD
  s_cls = F.log_softmax(stu_result[:, 4:, :], dim=1)  # Student class 확률
  t_cls = F.softmax(teacher_result[:, 4:, :], dim=1)  # Teacher class 확률
  loss_cls = F.kl_div(s_cls, t_cls, reduction='batchmean')

  # Bbox KD
  loss_bbox = F.mse_loss(stu_result[:, 0:4, :], teacher_result[:, 0:4, :])

  # Total KD loss
  total_kd_loss = (loss_cls + loss_bbox) * self.args.distill_ratio

  YOLO 출력 형식:
  stu_result.shape = [batch, 84, 8400]
                            ↑
                      84 channels:
                      [0:4]  → bbox (x, y, w, h)
                      [4:84] → class 확률 (80 classes)

  Loss 계산:
  - loss_cls: Teacher와 Student의 class 확률 분포 차이 (KL Divergence)
  - loss_bbox: Teacher와 Student의 bbox 좌표 차이 (MSE)
  - 둘을 합쳐서 distill_ratio 만큼 가중치 부여

  Step 4: Total Loss (Line 466)

  self.loss += total_kd_loss

  최종 Loss:
  total_loss = detection_loss + kd_loss
             = (원래 YOLO loss) + (Teacher와의 차이)

  Step 5: Backward (Line 477)

  self.scaler.scale(self.loss).backward()

  일반적인 backpropagation

  Step 6: Optimizer Step (Line 481)

  self.optimizer_step()  # Weight 업데이트

  Step 7: ★ Pruning ★ (Lines 484-487)

  if self.args.pruning_ratio == 0:
      continue  # 압축 안 함
  else:
      yolov8_pruning(self.model.model, self.args.pruning_ratio,
  device=self.device)

  여기가 핵심입니다! 매 optimizer step 후에 pruning 실행됩니다.

  ---
  5. Pruning 내부 동작

⏺ Read(ultralytics_custom/engine/compression.py)
  ⎿  Read 50 lines

⏺ yolov8_pruning() 동작:

  # YOLOv8 구조에서 레이어 0-22를 가져옴
  block_list = [model[i] for i in range(23)]

  for i, block in enumerate(block_list):
      if type(block).__name__ == 'Conv':
          # Conv 블록 프루닝
          pruning_idx = get_filter_pruning_idx(block.conv, sparsity=0.3)
          filter_pruning(block.conv, pruning_idx)
          bn_pruning(block.bn, pruning_idx)

  어떤 레이어를 프루닝?
  - Conv: block.conv, block.bn
  - C2f, SPPF: block.cv2.conv, block.cv2.bn
  - Detect: cv2[i][j], cv3[i][j] (3 scales × 2 layers)

  이제 핵심 함수들을 보겠습니다:

⏺ Read(ultralytics_custom/engine/compression_src/pruning/common.py)
  ⎿  Read 41 lines

⏺ 완벽합니다! 이제 핵심 함수들을 설명하겠습니다.

  get_filter_pruning_idx() - 어떤 필터를 제거할지 선택

  def get_filter_pruning_idx(layer, sparsity):
      weight = layer.weight  # 예: [256, 128, 3, 3]
      num_filters = weight.shape[0]  # 256개 필터
      num_pruning_filters = int(num_filters * sparsity)  # 0.3 → 76개 제거

      # 각 필터의 L2 norm 계산
      filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)
      # [256] - 각 필터마다 하나의 norm 값

      # L2 norm이 가장 작은 76개 선택
      _, pruning_idx = torch.topk(filter_norms, num_pruning_filters,
  largest=False)

      return pruning_idx  # 예: [1, 17, 89, ...] (76개 인덱스)

  구체적 예시:
  Conv layer weight: [256, 128, 3, 3]
  sparsity = 0.3

  Step 1: 각 필터의 norm 계산
  filter_norms = [0.8, 0.3, 1.2, 0.5, 0.9, 0.1, ...]
                 ↓     ↓               ↓     ↓
              살림  제거             살림  제거

  Step 2: 가장 작은 76개 선택
  pruning_idx = [1, 5, 17, 89, ...] (norm이 작은 76개)

  filter_pruning() - 선택된 필터를 0으로

  def filter_pruning(layer, pruning_idx):
      weight = layer.weight
      weight[pruning_idx, :, :, :] = 0.0

  무슨 일이?
  # Before
  weight[1] = [[0.5, -0.3], [0.8, 0.2]]  # 필터 1
  weight[5] = [[0.1, 0.05], [0.02, 0.0]]  # 필터 5

  # After
  weight[1] = [[0.0, 0.0], [0.0, 0.0]]  # 모두 0!
  weight[5] = [[0.0, 0.0], [0.0, 0.0]]  # 모두 0!

  결과:
  - 모델 크기는 그대로 [256, 128, 3, 3]
  - 하지만 76개 필터는 모두 0
  - 출력: 0이 된 필터는 아무 기여도 안 함

  bn_pruning() - Batch Normalization도 같이

  def bn_pruning(layer, pruning_idx):
      weight[pruning_idx] = 0.0   # γ = 0
      bias[pruning_idx] = 0.0     # β = 0
      mean[pruning_idx] = 0.0     # μ = 0
      var[pruning_idx] = 1.0      # σ² = 1

  왜?
  Batch Norm 수식:
  BN(x) = γ * ((x - μ) / sqrt(σ²)) + β

  파라미터를 위처럼 설정하면:
  BN(x) = 0 * ((x - 0) / sqrt(1)) + 0 = 0

  Conv 출력이 0이면 BN 출력도 0이 되도록!

  ---
  6. 매 Iteration 요약

  1. Student Forward
     - 이미지 처리
     - loss + feature 반환

  2. Teacher Forward
     - 같은 이미지 처리
     - soft label 생성

  3. Loss 계산
     - detection_loss (원래 YOLO loss)
     - kd_loss = KL_div(student, teacher) + MSE(bbox)
     - total_loss = detection_loss + kd_loss

  4. Backward
     - Gradient 계산

  5. Optimizer Step
     - Weight 업데이트

  6. ★ Pruning ★
     For each layer:
       - L2 norm 계산
       - 작은 norm의 필터 76개 선택
       - weight[선택된 필터] = 0
       - bn 파라미터도 조정

  결과:
  - 매 iteration마다 필터가 0이 됨
  - 중요한 필터는 살아남고
  - 중요하지 않은 필터는 0이 됨
  - 동시에 Teacher로부터 지식을 배움

  ---
  7. 학습 완료 후 (선택적)

  학습이 끝나면 모델에 0인 필터가 많이 있습니다. 이걸 물리적으로 제거하려면:

  from ultralytics_custom.engine.compression import yolov8_reducing

  # 프루닝된 모델
  model = YOLO('best.pt')

  # 새 모델 (같은 구조)
  reduced_model = YOLO('yolov8n.yaml')

  # 0인 필터 제거
  yolov8_reducing(model.model, reduced_model.model)

  # 저장
  reduced_model.save('compressed_final.pt')

  이 과정에서 [256, 128, 3, 3] → [180, 115, 3, 3] 처럼 실제 크기가
  줄어듭니다.

  ---
  전체 플로우 다이어그램

  사용자 명령
    ↓
  설정 로드 (pruning_ratio=0.3)
    ↓
  Teacher 모델 로드 (YOLOv8x)
    ↓
  Student 모델 로드 (YOLOv8n)
    ↓
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  For each batch:
    ↓
    Student Forward → loss + feature
    ↓
    Teacher Forward → soft label
    ↓
    Loss = detection_loss + KD_loss
    ↓
    Backward → Gradient
    ↓
    Optimizer → Weight update
    ↓
    ★ Pruning ★
      - 각 레이어에서
      - L2 norm 작은 필터 선택
      - weight = 0
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
  학습 완료
    ↓
  (선택) Reducing
    - 0인 필터 물리적 제거
    - 작은 모델 생성

  이게 전체 커스텀 YOLO의 작동 방식입니다!

  핵심은:
  1. Teacher-Student: 큰 모델이 작은 모델을 가르침
  2. Dynamic Pruning: 매 iteration마다 중요하지 않은 필터를 0으로
  3. 두 과정이 동시에: 학습하면서 압축됨

  궁금한 부분이 있으면 물어보세요!



  8.pruning
   코드에서 적용한 pruning 방식을 분석하면:

  Structured Filter Pruning (구조적 필터 프루닝)

  이 프로젝트는 L2-norm 기반 구조적 필터 프루닝을 사용합니다.

  핵심 특징:

  1. Filter-level Pruning (필터 단위 프루닝)

  - common.py:15-18: 개별 weight가 아닌 필터 전체를 0으로 만듦
  - Unstructured pruning(개별 weight 제거)이 아닌 Structured pruning

  2. L2 Norm 기반 중요도 측정

  common.py:33-40에서 핵심 로직:
  def get_filter_pruning_idx(layer, sparsity):
      weight = layer.weight
      num_filters = weight.shape[0]
      num_pruning_filters = int(num_filters * sparsity)

      # L2 norm으로 각 필터의 중요도 계산
      filter_norms = torch.norm(weight.view(num_filters, -1), dim=1)

      # 가장 작은 norm을 가진 필터들을 제거 대상으로 선택
      _, pruning_idx = torch.topk(filter_norms, num_pruning_filters,
  largest=False)
      return pruning_idx

  작동 원리:
  - 각 필터의 weight를 1D로 펼침
  - L2 norm (유클리드 거리) 계산: √(w₁² + w₂² + ... + wₙ²)
  - norm이 작을수록 = 덜 중요한 필터로 판단
  - largest=False로 가장 작은 norm 필터들을 선택

  3. Magnitude-based Pruning

  - 가중치의 크기(magnitude)를 기준으로 중요도 판단
  - 다른 기준들:
    - ❌ Gradient-based (학습 중 gradient 크기)
    - ❌ Activation-based (출력 activation 크기)
    - ❌ Hessian-based (2차 미분 정보)
    - ✅ Magnitude-based (가중치 L2 norm)

  4. One-shot Pruning

  - compression.py:7-49: 전체 모델을 한 번에 pruning
  - Iterative pruning(점진적 제거)이 아닌 일괄 pruning
  - Fine-tuning 없이 바로 적용

  정리:

  이 방식은 **"L2-norm based One-shot Structured Filter Pruning"**입니다:

  | 분류 기준             | 적용된 방식                    | 설명
             |
  |-------------------|---------------------------|-------------------------
  |
  | Granularity       | Structured (Filter-level) | 필터 전체 제거
        |
  | Importance Metric | Magnitude (L2 norm)       | 가중치 크기 기반
         |
  | Pruning Schedule  | One-shot                  | 한 번에 일괄 적용
         |
  | Layer Selection   | Global Sparsity           | 모든 레이어에 동일한
  sparsity 적용 |

  장점:
  - 구현 간단
  - HW 가속 용이 (실제 메모리/연산량 감소)
  - 추가 학습 없이 빠른 압축

  단점:
  - Layer별 중요도 차이 무시
  - Fine-tuning 없어 정확도 하락 가능