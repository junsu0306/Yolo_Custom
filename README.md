# Model Compression Project

YOLO와 CenterPose 모델에 대한 **Structured Pruning** 및 **Knowledge Distillation** 기반 압축 프로젝트입니다.

## 프로젝트 구조

```
Yolo_Custom/
├── ultralytics_custom/      # YOLO 압축 구현 (Training-time)
├── ultralytics_github/      # YOLO 원본 코드
├── CenterPose src custom/   # CenterPose 압축 구현 (One-shot)
├── Centerpose src github/   # CenterPose 원본 코드
└── analysis/                # 상세 기술 문서
    ├── Yolo_Ultranics_Compression.md
    └── CenterPose_Compression.md
```

## 압축 기법 비교

| 항목 | YOLO | CenterPose |
|------|------|------------|
| **백본** | YOLOv8 (CSPDarknet) | DLA-34 |
| **Pruning** | Dynamic (매 step) | One-shot |
| **Distillation** | YOLOv8x → YOLOv8n | 없음 |
| **압축률** | 30-35% | 30-40% |

## 핵심 기술

### 1. L2 Norm 기반 Structured Pruning
- 필터 중요도를 L2 norm으로 측정
- Global sparsity 기준 통합 pruning
- BN 레이어 동시 처리

### 2. Knowledge Distillation (YOLO)
- Teacher: YOLOv8x (68.2M params)
- Student: YOLOv8n (3.2M params)
- KL Divergence + MSE Loss

### 3. Channel Reduction
- 0으로 마스킹된 필터 물리적 제거
- 실제 모델 크기 및 메모리 감소

## 빠른 시작

### YOLO 압축 학습

```bash
cd ultralytics_custom
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='coco8.yaml', epochs=100, pruning_ratio=0.3)
"
```

### CenterPose 압축

```bash
cd "CenterPose src custom"
python pruning.py      # Pruning
python reducing.py     # Channel Reduction
```

## 상세 문서

- [YOLO Compression 상세](ultralytics_custom/README.md)
- [CenterPose Compression 상세](CenterPose%20src%20custom/README.md)
- [YOLO 기술 분석](analysis/Yolo_Ultranics_Compression.md)
- [CenterPose 기술 분석](analysis/CenterPose_Compression.md)

---

## 실제 실행 가이드 (Quick Start)

### 사전 요구사항

```bash
pip install torch torchvision opencv-python numpy
```

---

### YOLO 압축 실행

#### 1. 원본 레포지토리 준비
```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics && pip install -e .
```

#### 2. Custom 파일 적용
```bash
# 핵심 파일 복사
cp ultralytics_custom/engine/trainer.py ultralytics/ultralytics/engine/
cp ultralytics_custom/engine/compression.py ultralytics/ultralytics/engine/
cp ultralytics_custom/cfg/default.yaml ultralytics/ultralytics/cfg/
```

#### 3. 압축 학습
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='coco8.yaml',
    epochs=100,
    pruning_ratio=0.3,  # 30% 필터 제거
    mem_usg=100.0
)
```

#### 4. 추론
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=image.jpg
```

---

### CenterPose 압축 실행

#### 1. 원본 레포지토리 준비
```bash
git clone https://github.com/NVlabs/CenterPose.git
cd CenterPose && pip install -r requirements.txt
```

#### 2. Custom 파일 적용
```bash
# 핵심 파일 복사
cp "CenterPose src custom/lib/pruning/dlasg_pruning.py" CenterPose/src/lib/pruning/
cp "CenterPose src custom/pruning.py" CenterPose/src/
cp "CenterPose src custom/reducing.py" CenterPose/src/
cp "CenterPose src custom/reduced_demo.py" CenterPose/src/
```

#### 3. Pruning + Reducing
```bash
cd CenterPose/src

# pruning.py 내 model_path 수정 후 실행
python pruning.py      # 출력: my_pruned_model_50.pth

# reducing.py 내 model_path 수정 후 실행
python reducing.py     # 출력: reduced_model_50.pth
```

#### 4. 추론
```bash
python reduced_demo.py --load_model reduced_model_50.pth --demo image.jpg
```

---

## 실행 흐름 비교

| 단계 | YOLO | CenterPose |
|------|------|------------|
| **1. 환경** | `ultralytics` 클론 | `CenterPose` 클론 |
| **2. 파일 적용** | trainer.py, compression.py 복사 | dlasg_pruning.py, pruning.py 복사 |
| **3. 압축** | `model.train(pruning_ratio=0.3)` | `python pruning.py` + `python reducing.py` |
| **4. 추론** | `model.predict(source=...)` | `python reduced_demo.py` |
| **학습 필요** | Yes (Training-time) | No (One-shot) |
| **KD 적용** | Yes (Teacher→Student) | No |

---

## 압축 결과 예상

| 모델 | 원본 파라미터 | 압축 후 | 감소율 | mAP 변화 |
|------|-------------|---------|--------|----------|
| YOLOv8n | 3.2M | 2.0-2.2M | 30-35% | -3~5% |
| CenterPose (DLA-34) | ~20M | ~12-14M | 30-40% | Fine-tuning 권장 |

---

## 문제 해결

### GPU 메모리 부족
```bash
# YOLO: 배치 크기 줄이기
model.train(batch=8)

# CenterPose: GPU 지정
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
```

### 성능 저하가 큰 경우
- `pruning_ratio`를 0.2~0.3으로 낮추기
- 학습 에폭 증가
- CenterPose는 압축 후 Fine-tuning 권장
