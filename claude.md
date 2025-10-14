# Claude AI Assistant - 작업 지침서

## 프로젝트 목적

**이 프로젝트의 목표는 AI 모델(신경망) 압축 기법을 탐색하는 것입니다.**

- ❌ 코드 파일 압축이 아님
- ❌ 코드베이스 리팩토링 분석이 아님
- ✅ **신경망 모델의 크기/연산량을 줄이는 AI 모델 압축 기법 탐색**

## 프로젝트 구조

```
/Users/junsu/Projects/Yolo_Custom/
├── ultralytics_github/      # 공개 GitHub 원본 버전
├── ultralytics_custom/       # AI 모델 압축 기능이 추가된 커스텀 버전
├── claude.md                 # 이 파일 - Claude를 위한 지침서
└── analysis/                 # 분석 결과 저장 폴더
```

## 주요 작업

### 1. AI 모델 압축 코드 위치
- `ultralytics_custom/engine/compression.py` - 메인 압축 로직
- `ultralytics_custom/engine/compression_src/pruning/` - 프루닝 구현
- `ultralytics_custom/engine/compression_src/reducing/` - 채널 축소 구현

### 2. 분석해야 할 내용
- YOLOv8 모델 압축 알고리즘 이해
- Pruning (가지치기) 기법: 어떤 필터/채널을 제거하는가?
- Reducing (축소) 기법: 프루닝 후 실제 모델 크기를 어떻게 줄이는가?
- Architecture-aware 압축: YOLO 구조 특성을 어떻게 고려하는가?

### 3. 비교 포인트
- GitHub 원본에는 없는 압축 기능이 Custom에 추가됨
- 두 버전을 비교하여 추가된 모델 압축 로직을 파악

## 중요 파일 경로

### 압축 관련 핵심 파일
```
ultralytics_custom/engine/compression.py
ultralytics_custom/engine/compression_src/pruning/common.py
ultralytics_custom/engine/compression_src/pruning/resnet.py
ultralytics_custom/engine/compression_src/reducing/common.py
ultralytics_custom/engine/compression_src/reducing/resnet.py
ultralytics_custom/engine/compression_src/models/resnet.py
```

### 분석 결과 저장
- 모든 분석 결과는 `analysis/` 폴더에 저장
- 마크다운 형식으로 정리
- 코드 예제와 함께 설명

## 작업 시 주의사항

1. **모델 압축에 집중**: 코드 리팩토링이나 파일 구조 변경은 부차적
2. **기술적 세부사항**: 알고리즘의 수학적/기술적 세부사항 파악
3. **실제 구현**: 코드 레벨에서 어떻게 구현되었는지 분석
4. **성능 영향**: 압축이 모델 성능에 미치는 영향 (정확도 vs 속도)

## 다음 단계

사용자가 요청하면:
- 압축 알고리즘 상세 분석
- 코드 실행 흐름 추적
- 특정 기법 (pruning, quantization 등) 심층 분석
- 다른 압축 기법과의 비교

## 참고사항

- ultralytics_custom은 v8.2.64 기반
- ultralytics_github는 v8.3.213
- Custom 버전에 AI 모델 압축 프레임워크가 추가됨
