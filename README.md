# Toss 광고 클릭 예측 모델

> Toss 사용자의 광고 클릭 여부를 예측하는 재현 가능한 파이프라인 구축: 탐색적 데이터 분석부터 프로덕션 배포까지

## 목차
- [프로젝트 개요](#프로젝트-개요)
- [저장소 구조](#저장소-구조)
- [시작하기](#시작하기)
- [데이터 준비](#데이터-준비)
- [실험 워크플로우](#실험-워크플로우)
- [모델링 로드맵](#모델링-로드맵)
- [평가 전략](#평가-전략)
- [프로젝트 현황 및 향후 계획](#프로젝트-현황-및-향후-계획)
- [기여 방법](#기여-방법)
- [라이선스](#라이선스)

## 프로젝트 개요

본 저장소는 Toss 광고 캠페인을 위한 클릭률(CTR, Click-Through Rate) 예측 시스템의 기반 구조입니다. 최종 목표는 노출 로그 데이터를 수집하고, 예측 특성을 엔지니어링하며, 랭킹 모델을 학습하여 다운스트림 입찰 또는 추천 서비스에서 사용할 수 있는 신뢰할 수 있는 클릭 확률 추정치를 제공하는 것입니다.

현재 버전은 깔끔한 프로젝트 레이아웃, 템플릿 노트북, 파이프라인 확장 가이드를 제공합니다. 의도적으로 가볍게 설계하여 자체 로그 데이터를 연결하고 빠르게 반복 실험할 수 있습니다.

## 저장소 구조

```
.
├── data/                  # 외부 데이터셋 저장 (버전 관리 제외)
├── notebooks/             # 탐색적 분석 및 기준 모델 실험
├── src/                   # 핵심 워크플로우를 위한 프로덕션급 노트북/스크립트
├── requirements.txt       # Python 의존성 (프로젝트 진행에 따라 업데이트)
├── LICENSE                # MIT 라이선스
└── README.md              # 프로젝트 문서
```

**주요 파일**
- `notebooks/EDA.ipynb`: 원시 노출/클릭 로그의 탐색적 데이터 분석 템플릿
- `notebooks/baseline.ipynb`: 로지스틱 회귀 또는 트리 기반 모델로 빠른 베이스라인 구축
- `src/train.ipynb`, `src/model.ipynb`, `src/inference.ipynb`, `src/evaluate.ipynb`: 워크플로우 안정화 시 스크립트로 내보낼 구조화된 노트북

## 시작하기

**1. 저장소 클론**
```bash
git clone https://github.com/<your-username>/toss-ad-click-prediction.git
cd toss-ad-click-prediction
```

**2. Python 환경 생성** (Conda 예시)
```bash
conda create -n toss-ctr python=3.10
conda activate toss-ctr
```

**3. 의존성 설치** (`requirements.txt` 작성 후)
```bash
pip install -r requirements.txt
```

**4. Jupyter Lab 실행**
```bash
jupyter lab
```

## 데이터 준비

### 데이터 소스
- **출처**: 내부 Toss 광고 로그 또는 공식 공개 데이터셋 사용
- 라이선스 문제로 데이터셋은 저장소에 커밋되지 않음

### 디렉토리 구조
- 원시 파일: `data/raw/` 저장
- 가공 데이터: `data/processed/` 저장
- `.gitignore`로 민감한 파일 버전 관리 제외

### 스키마
표준 CTR 태스크에는 다음 항목 포함:
- 노출 식별자
- 광고 메타데이터
- 사용자 특성
- 타임스탬프
- 클릭 레이블 (`clicked` ∈ {0,1})

**권장 전처리 단계**

1. **스키마 검증 및 결측치 처리**
   - 데이터 타입 확인 및 보정
   - 결측값 대체 또는 제거 전략 수립

2. **범주형 변수 인코딩**
   - 타겟 인코딩, 임베딩, 원-핫 인코딩 (카디널리티에 따라 선택)

3. **시간 특성 엔지니어링**
   - 최신성(recency): 마지막 클릭/노출 이후 경과 시간
   - 빈도(frequency): 사용자 활동 빈도
   - 체류 시간(dwell time): 광고 노출 시간

4. **상호작용 특성 생성**
   - 사용자 × 광고 히스토리
   - 교차 특성 조합

## 실험 워크플로우

### 1단계: 탐색적 분석 (`notebooks/EDA.ipynb`)
- 분포 이해
- 시간 기반 드리프트 분석
- 클릭의 주요 동인 파악

### 2단계: 베이스라인 모델링 (`notebooks/baseline.ipynb`)
- 빠른 벤치마크 수립 (로지스틱 회귀, LightGBM 등)
- 향후 개선 목표 설정

### 3단계: 특성 엔지니어링 및 학습 (`src/train.ipynb`, `src/model.ipynb`)
- 특성 파이프라인 반복 실험
- 학습/검증 분할 관리
- 교차 검증 수행

### 4단계: 평가 (`src/evaluate.ipynb`)
- 리더보드 점수 계산: AP 및 WLL
- 보조 진단: 보정 플롯, 혼동 행렬 요약
- 실험 메타데이터 저장 (MLflow 활용 권장)

### 5단계: 추론 (`src/inference.ipynb`)
- 배치/온라인 추론 로직 준비
- 후처리: 점수 클리핑, 보정

**자동화**: 노트북 안정화 후 `make` 또는 경량 스크립트로 자동화

## 모델링 로드맵

### 후보 모델
- **Gradient Boosted Trees**: LightGBM, XGBoost
- **Factorization Machines**: FM, FFM
- **딥러닝 CTR 모델**: Wide & Deep, DeepFM
- **시퀀스 모델**: 세션 데이터용 Transformer 기반 인코더

### 정규화 및 일반화
- 시간 기반 층화 K-Fold
- 특성 중요도 기반 가지치기
- Adversarial Validation으로 분포 변화 방지

### Feature Store 고려사항
- 학습과 서빙 간 재사용을 위한 특성 정의 표준화

### 온라인 서빙
- ONNX 또는 TorchScript로 모델 내보내기
- FastAPI 마이크로서비스로 실시간 스코어링

## 평가 전략

### 리더보드 점수 산식
```
score = 0.5 × AP + 0.5 × (1 / (1 + WLL))
```
- 높은 AP(Average Precision)와 낮은 WLL(Weighted LogLoss)이 더 나은 점수 생성

### Average Precision (AP)
- 예측 확률값 기반 계산
- 모든 임계값에서 정밀도 측정 후 전체 재현율 범위에서 평균

### Weighted LogLoss (WLL)
- 클래스 가중치 조정 로그 손실
- `clicked=0`과 `clicked=1`이 동등하게 기여 (50:50)

### 비즈니스 KPI
- 예상 수익 증가 시뮬레이션 (CPC × 예측 CTR)
- False Positive 비용 모니터링

### 실험 추적
- MLflow/Weights & Biases 통합 권장
- 데이터셋 버전, 특성 세트, 하이퍼파라미터, 시드 기록

## 프로젝트 현황 및 향후 계획

<img width="607" height="169" alt="Screenshot 2025-10-14 at 14 42 34" src="https://github.com/user-attachments/assets/e902bd46-9a95-46c3-95f5-c204ad24b9ed" />

### 현재 상태
- 저장소 기본 구조 완성
- 데이터셋 통합 및 베이스라인 구현 대기 중

### 즉시 실행 항목
1. 실험 중 선택한 라이브러리로 `requirements.txt` 작성
2. EDA 및 베이스라인 노트북에 실제 분석 내용 작성
3. 학습 스크립트 구현 (`src/train.ipynb` → `.py`)
4. `artifacts/` 폴더 생성 후 학습된 모델 저장

### 확장 목표
- **CI 루틴 구축**: nbQA, black으로 노트북 린트, 내보낸 코드 유닛 테스트
- **Docker 컨테이너화**: 재현 가능한 배포를 위한 파이프라인 컨테이너화
- **모니터링 대시보드**: 프로덕션 추론 드리프트 감지 대시보드 추가

## 기여 방법

Pull Request 및 Issue를 환영합니다.

**기여 가이드라인**
1. 계획한 변경 사항을 설명하는 Issue 생성
2. Feature-branch 워크플로우 사용, 집중된 커밋 유지
3. PR 설명에 재현성 정보 포함 (데이터셋 버전, 랜덤 시드)

## 라이선스

본 프로젝트는 [MIT License](LICENSE) 하에 배포됩니다.

---

## 기술 스택 요약

**언어**: Python 3.10+  
**주요 라이브러리** (예상):
- 데이터 처리: Pandas, NumPy
- 모델링: Scikit-learn, LightGBM, XGBoost, PyTorch
- 실험 추적: MLflow, Weights & Biases
- 추론: ONNX Runtime, FastAPI

**인프라** (확장 시):
- 컨테이너: Docker
- 오케스트레이션: Kubernetes
- 모니터링: Prometheus, Grafana

## 참고 자료

CTR 예측 모델링에 대한 추가 학습 자료:
- [Wide & Deep Learning](https://arxiv.org/abs/1606.07792)
- [DeepFM](https://arxiv.org/abs/1703.04247)
- [LightGBM 공식 문서](https://lightgbm.readthedocs.io/)
