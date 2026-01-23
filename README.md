# On-Device AI Model Compression

**IITP (ETRI Project) - On_Device_AI**

---

## 레포지토리 구조

```
On_Device_AI/
├── ultralytics_custom/          # YOLOv8 압축 커스텀 코드 (원본 중 src 폴더 해당)
│   ├── engine/compression.py    # Pruning/Reducing 함수
│   ├── engine/trainer.py        # KD 통합 학습 로직
│   └── README.md                # 상세 사용법
│
├── CenterPose src custom/       # CenterPose 압축 커스텀 코드 (원본 중 src 폴더 해당)
│   ├── lib/pruning/             # Pruning 핵심 모듈
│   ├── pruning.py               # Pruning 실행 스크립트
│   ├── reducing.py              # Reducing 실행 스크립트
│   └── README.md                # 상세 사용법
│
└── analysis/                    # 기술 분석 문서
```

> **Note**: 이 레포지토리에는 **커스텀 코드만** 포함되어 있습니다.
> 원본 YOLO/CenterPose 레포지토리는 별도로 클론하여 사용해야 합니다.

---

## 지원 모델

### YOLOv8 (Object Detection)

- **압축 방식**: Training-time Dynamic Pruning + Knowledge Distillation
- **Teacher → Student**: YOLOv8x (68.2M) → YOLOv8n (3.2M)
- **압축률**: 30-35%
- **상세 문서**: [ultralytics_custom/README.md](ultralytics_custom/README.md)

### CenterPose (6D Pose Estimation)

- **압축 방식**: One-shot Structured Pruning + Channel Reduction
- **백본**: DLA-34 (~20M params)
- **압축률**: 30-40%
- **상세 문서**: [CenterPose src custom/README.md](CenterPose%20src%20custom/README.md)

---

## 압축 기법 비교

| 항목 | YOLOv8 | CenterPose |
|------|--------|------------|
| **Pruning 방식** | Dynamic (매 step) | One-shot |
| **Knowledge Distillation** | Yes | No |
| **학습 필요** | Yes | No |
| **압축률** | 30-35% | 30-40% |
| **성능 유지** | 우수 (KD 효과) | Fine-tuning 권장 |

---

## 빠른 시작

### 사전 요구사항

```bash
pip install torch torchvision opencv-python numpy
```

### YOLOv8 압축

```bash
# 1. 원본 레포 클론
git clone https://github.com/ultralytics/ultralytics.git

# 2. 커스텀 파일 적용 후 학습
# 상세: ultralytics_custom/README.md 참조
```

### CenterPose 압축

```bash
# 1. 원본 레포 클론
git clone https://github.com/NVlabs/CenterPose.git

# 2. 커스텀 파일 적용 후 실행
# 상세: "CenterPose src custom/README.md" 참조
```

---

## 상세 문서

각 모델의 **설치 방법, 실행 가이드, 파라미터 설명** 등 상세 내용은 아래 문서를 참조하세요.

| 모델 | 문서 |
|------|------|
| YOLOv8 | [ultralytics_custom/README.md](ultralytics_custom/README.md) |
| CenterPose | [CenterPose src custom/README.md](CenterPose%20src%20custom/README.md) |
| YOLO 기술 분석 | [analysis/Yolo_Ultranics_Compression.md](analysis/Yolo_Ultranics_Compression.md) |
| CenterPose 기술 분석 | [analysis/CenterPose_Compression.md](analysis/CenterPose_Compression.md) |
