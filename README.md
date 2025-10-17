# Toss 광고 클릭 예측 모델

> Toss 사용자의 광고 클릭 여부를 예측하는 재현 가능한 파이프라인 구축: 탐색적 데이터 분석부터 프로덕션 배포까지

## 프로젝트 개요

본 저장소는 Toss 광고 캠페인을 위한 클릭률(CTR, Click-Through Rate) 예측 시스템의 기반 구조입니다. 최종 목표는 노출 로그 데이터를 수집하고, 예측 특성을 엔지니어링하며, 랭킹 모델을 학습하여 다운스트림 입찰 또는 추천 서비스에서 사용할 수 있는 신뢰할 수 있는 클릭 확률 추정치를 제공하는 것입니다.

### 주요 도전 과제

본 프로젝트는 다음과 같은 기술적 난제들을 포함하고 있었습니다:

**데이터 규모 및 복잡성**
- 1천만 건 이상의 대규모 데이터셋 처리
- 1:52의 극심한 클래스 불균형 문제
- 익명화된 피처로 인한 도메인 지식 적용의 한계
- 메모리 및 연산 자원 제약

**모델링 복잡도**
- 고차원 특성 공간에서의 의미 있는 패턴 추출
- 불균형 데이터에 대한 적절한 샘플링 전략 수립
- 평가 지표(AP + WLL)에 최적화된 손실 함수 설계

### 주요 성과

**정량적 성과**
- 약 700개 참가팀 중 상위 10% 달성 (리더보드 점수: 0.34814)
- 체계적인 특성 엔지니어링을 통한 성능 개선
- DCN-v3 모델 적용으로 전통적 ML 대비 우수한 성능 달성

**기술적 기여**
- 대규모 데이터 처리를 위한 청크 기반 전처리 파이프라인 구축
- Cyclic Encoding 및 고차 상호작용 특성을 통한 시간 패턴 모델링
- Cross Network 안정화를 위한 정규화 전략 수립


---

## 프로젝트 현황

### 리더보드 순위
약 700개 팀 중 상위 10% 달성

<img width="607" height="169" alt="Screenshot 2025-10-14 at 14 42 34" src="https://github.com/user-attachments/assets/e902bd46-9a95-46c3-95f5-c204ad24b9ed" />

- **현재 점수**: 0.34814
- **사용 모델**: DCN-v3

---
# Toss 광고 클릭 예측 파이프라인

Toss 디지털 광고 캠페인의 클릭 여부를 예측하기 위한 엔드투엔드 파이프라인입니다. `src/main.py`가 전처리 → 학습 → 추론을 순차 실행하며, 실험 반복과 배포까지 재현 가능한 형태로 구성되어 있습니다. 대규모 익명 로그를 안정적으로 다루기 위해 데이터 핸들링, 모델 학습, 모니터링 모듈을 분리했습니다.

## 주요 특징
- **DCN-v3 기반 CTR 모델**: Deep & Cross Network v3로 고차 상호작용을 학습하고, 필요 시 전통 ML 베이스라인과 비교 실험을 수행합니다.
- **Polars 전처리 파이프라인**: `src/preprocess/` 모듈이 청크 단위로 데이터를 읽어 파생 특성을 생성하고 `data/processed/`에 저장합니다.
- **실험 추적과 온도 보정**: Weights & Biases로 학습 로그를 기록하고, 추론 단계에서 온도 보정(Temperature Scaling)으로 예측 확률을 교정합니다.
- **평가 지표 대응**: 대회 점수 산식인 `0.5 × AP + 0.5 × (1 / (1 + WLL))`를 기준으로 모델을 조율합니다.

## 시작하기
### 1. 개발 환경 준비
```bash
python3 -m venv toss-ml-venv
source toss-ml-venv/bin/activate
pip install -r requirements.txt
```
CUDA 11.8 이상의 GPU 환경을 권장하지만, GPU가 없다면 자동으로 CPU로 폴백합니다.

### 2. 데이터 준비
- 원본 파케이 데이터는 `data/` 아래에 두고, 가공된 피처는 `data/processed/`에 저장합니다.
- 학습 전 `data/processed/train_processed_2.parquet`, `data/processed/test_processed_2.parquet` 존재 여부를 확인하세요. 없다면 아래 전처리 스크립트를 실행합니다.

```bash
python src/preprocess/preprocess.py
```

### 3. 핵심 명령어
- 학습: `python src/train.py` (스모크 테스트 시 `config.CFG["EPOCHS"] = 1` 설정)
- 평가: `python src/evaluate.py` (모델 출력과 지표 계산)
- 추론 및 제출 생성: `python src/inference.py`
- 통합 파이프라인: `python src/main.py` → `data/output/`에 제출 CSV 작성

## 저장소 구조
```
.
├── data/              # 원본 데이터 (버전 관리 제외)
├── data/processed/    # 전처리된 특성 저장 위치
├── data/output/       # 모델 아티팩트와 제출 파일
├── notebooks/         # 탐색적 분석 및 실험용 노트북
├── src/               # 프로덕션 파이프라인 코드
│   ├── preprocess/    # 전처리 및 피처 엔지니어링 모듈
│   ├── train.py       # K-Fold 학습 루프
│   ├── evaluate.py    # AP, WLL 계산 및 시각화
│   ├── inference.py   # 추론 파이프라인 및 보정
│   └── main.py        # 전체 워크플로 orchestration 스크립트
└── tests/             # pytest 케이스 (필요 시 추가)
```

## 파이프라인 개요
1. **전처리 (`src/preprocess/`)**: 범주형 정수 인코딩, 시간 주기성(cyclic) 특징, 시퀀스 통계, 상호작용 피처를 생성합니다.
2. **데이터 로더 (`src/data_loader.py`)**: Polars DataFrame을 PyTorch `Dataset`으로 변환하고 배치 샘플링 전략을 제어합니다.
3. **모델 학습 (`src/train.py`)**: K-폴드 교차 검증을 수행하고 EMA, AMP, 코사인 스케줄러로 안정적 학습을 보장합니다. 학습 기록은 W&B에 남기고 체크포인트는 `models/`에 저장합니다.
4. **평가 (`src/evaluate.py`)**: AP, WLL, Calibration Curve를 계산하며, 최적 임계값 탐색을 지원합니다.
5. **추론 (`src/inference.py`)**: 최적 모델 가중치와 온도 보정 파라미터를 불러와 최종 예측 확률을 계산하여 CSV를 생성합니다.

## 설정 관리 (`src/config.py`)
- 모든 하이퍼파라미터는 `config.CFG` 딕셔너리에서 관리합니다.
- `CFG["DATA_PATH"]`, `CFG["OUTPUT_PATH"]`, `CFG["CHECKPOINT_DIR"]` 등 경로 관련 설정을 환경에 맞게 조정할 수 있습니다.
- `CFG["USE_WANDB"]`를 `False`로 두면 W&B 로깅을 비활성화합니다.

## 모델 및 실험 관리
- **모델 구성**: 임베딩 드롭아웃, 다층 퍼셉트론(Deep Tower), Cross Network, 시퀀스 LSTM을 조합합니다.
- **스케줄러**: 워ーム업 후 코사인 애널링 스케줄을 적용하며, EMA로 가중치를 스무딩합니다.
- **캘리브레이션**: `src/calibrate/`에서 온도 보정 모델을 저장하고, 추론 시 `CFG["CALIBRATION_ENABLED"]` 플래그로 제어합니다.

### 주요 파일
- `notebooks/EDA.ipynb`: 원시 노출/클릭 로그의 탐색적 데이터 분석 템플릿
- `notebooks/baseline.ipynb`: 로지스틱 회귀 또는 트리 기반 모델로 빠른 베이스라인 구축
- `src/train.ipynb`, `src/model.ipynb`, `src/inference.ipynb`, `src/evaluate.ipynb`: 워크플로우 안정화 시 스크립트로 내보낼 구조화된 노트북

---

## 실험 워크플로우

### 0단계: EDA 및 데이터 전처리

#### 탐색적 데이터 분석
- 익명화된 피처의 의미 탐구 및 전략 수립
- 피처간 상관관계 분석을 통한 중복 특성 식별
- 1:52 비율에서 1:10의 다운샘플링 전략 선택

<img width="1490" height="1390" alt="image (4)" src="https://github.com/user-attachments/assets/dcc2360f-6cd1-4f64-ad00-b6cb43878567" />

#### 기본 전처리
**데이터 타입 통일**
- 범주형 변수를 Int32로 변환: gender, age_group, inventory_id, day_of_week, hour

**수치형 안정화**
- inventory_id: 로그 변환 후 정규화 (범위가 큰 ID 값 압축)
- age_group: Min-Max 정규화

#### 시간 특성 엔지니어링
**Cyclic Encoding (주기성 표현)**
- `hour_sin/cos`: 24시간 주기 (0시와 23시 연결)
- `dow_sin/cos`: 7일 주기 (월요일과 일요일 연결)
- `week_hour_sin/cos`: 168시간 주기 (주간 패턴 캡처)

**시간대 범주화**
- 클릭률 패턴 기반 5단계 분류
  - 새벽(0-4시), 오전(5-8시), 업무시간(9-12시), 오후(13-16시), 저녁(17-23시)

**이진 및 정규화 특성**
- 주말 여부, 토요일 여부
- hour, day_of_week를 0-1 범위로 정규화
- 시간대별 CTR 수준 3단계 분류 (고/중/저)

#### 시퀀스 특성 처리
**기본 통계 추출**
- 시퀀스 길이, 첫 번째/마지막 값

**범주화 및 플래그**
- 길이 기반 5단계 분류: 초단/단/중/장/초장 시퀀스
- 고CTR 플래그: 5000+ 길이 시퀀스 식별

**변환 및 최적화**
- 로그 정규화: 넓은 범위(0-15000+)를 로그 변환으로 압축
- 메모리 최적화: 청크 단위 처리로 안정적 대용량 데이터 처리

#### 상호작용 특성 생성
**사용자-광고 상호작용**
- `gender × age_group`: 성별-연령 조합
- `inventory_id × gender`: 광고-성별 조합
- `inventory_id × age_group`: 광고-연령 조합
- `inventory_id × gender × age_group`: 3차 상호작용

**시간-사용자 상호작용**
- `hour × day_of_week`: 시간-요일 조합
- `is_weekend × hour`: 주말-시간 조합
- `hour × age_group`: 시간-연령 조합

#### 특성 선택 및 필터링

**제거 기준 (3단계 필터링)**

1. **도메인 지식 기반**: 낮은 정보량 특성 제거
   - feat_a 계열: 12개
   - feat_b, feat_e 계열: 3개
   - history_a, history_b 계열: 11개
   - l_feat 계열: 9개

2. **완전상관 특성**: 중복 정보 제거
   - `history_b_24` ↔ `history_16`
   - `feat_d_5/6` ↔ `feat_d_1`

3. **Feature Importance 하위 10%**: LightGBM/XGBoost 기반
   - 총 11개 컬럼 (seq_high_ctr_flag, feat_b_1, history_b 계열 등)

#### 정규화 및 안정화

**범주형 재인코딩**
- gender, age_group을 0-based indexing 변환

**수치형 Min-Max 정규화**
- `feat_*` 계열: 0-1 범위 정규화
- `history_*` 계열: 0-1 범위 정규화
- `l_feat_*` 계열: 0-1 범위 정규화

**Cross Network 안정성**
- 정규화를 통한 그래디언트 폭발 방지

---

### 1단계: 베이스라인 모델 구축

#### 전통적 ML 모델 실험
- LightGBM, XGBoost를 활용한 베이스라인 구축
- Feature Importance 분석 및 특성 선택
- 교차 검증을 통한 모델 안정성 확인

#### 성능 개선
- Feature importance 기반 특성 선별 및 추가 엔지니어링
- 시간 기반 특성 강화: 요일, 시간대, 계절성 패턴 반영

#### 모델 최적화
- Wandb Sweep으로 하이퍼파라미터 실험
- 평가지표에 맞는 custom 손실함수 설정

---

### 2단계: 고급 모델 실험

#### 딥러닝 CTR 모델 적용
**모델 선정 과정**
- DeepFM, Wide & Deep 등 딥러닝 CTR 모델 실험
- DCN (Deep & Cross Network) 테스트
- **최종 선택**: DCN-v3 (Transformer 기반 Cross Network)
  - 사용자 행동 패턴의 고차 상호작용 자동 학습
  - Cross Network로 명시적 특성 교차 효과 극대화

#### 모델 해석성 강화
**예측 분석**
- 예측 확률 분포 시각화
- 파라미터 미세조정 및 민감도 분석

**신뢰도 개선**
- Calibration 플롯 분석
- 예측 확률 보정 (Temperature Scaling)

#### 재현성 확보
**실험 관리**
- Weights & Biases를 통한 실험 추적
- 하이퍼파라미터, 메트릭, 모델 버전 기록

**코드 품질**
- 특성 파이프라인 문서화
- 데이터 버전 관리 (DVC 활용 고려)
- 모듈화 및 단위 테스트 작성

---

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

---

## 프로젝트 회고

### 1. 기술적 학습
**추천 시스템 이해 심화**
- DCN(Deep & Cross Network) 아키텍처의 원리 및 구현 경험
- CTR 예측에서 명시적 특성 교차(Explicit Feature Crossing)의 중요성 인식
- Wide & Deep, DeepFM 등 다양한 딥러닝 CTR 모델 비교 분석

**실무 데이터 처리 역량**
- 익명화된 기업 데이터 분석 및 특성 추론 능력 향상
- 대규모 불균형 데이터 처리 경험 축적
- Polars를 활용한 효율적인 데이터 파이프라인 구축

**실험 관리 체계화**
- Weights & Biases를 통한 체계적인 실험 추적
- 재현 가능한 연구를 위한 버전 관리 및 문서화

### 2. 한계점 및 개선 방향

**샘플링 전략의 제한**
- 1:10 다운샘플링 적용으로 데이터 활용도 감소
- 향후 연구: Stratified Sampling, SMOTE, Focal Loss 등 다양한 불균형 처리 기법 비교 실험 필요
- 전체 데이터 활용을 위한 분산 학습 인프라 구축 고려

**모델 앙상블 미적용**
- 단일 모델(DCN-v3)에만 의존
- 향후 개선: LightGBM + DCN-v3 Stacking, Weighted Ensemble 등 앙상블 기법 적용 가능성 탐색

**하이퍼파라미터 탐색 범위**
- Wandb Sweep을 활용했으나 계산 자원 제약으로 탐색 범위 제한
- 향후 연구: Learning Rate Scheduling, Regularization 강도 등 추가 튜닝 여지 존재

**시간적 제약**
- Transformer 기반 시퀀스 모델 등 고급 기법 실험 시간 부족
- 사용자 행동 시퀀스 모델링의 심화 연구 필요

### 3. 결론

본 프로젝트는 실제 기업 데이터를 활용한 대규모 CTR 예측 시스템 구축 경험을 제공했습니다. 특히 DCN 모델의 이론적 이해와 실무 적용을 통해 추천 시스템 분야의 전문성을 크게 향상시킬 수 있었습니다. 

비록 샘플링 전략과 앙상블 기법 등 일부 고급 기술을 충분히 탐색하지 못한 아쉬움이 남지만, 제한된 자원 내에서 체계적인 특성 엔지니어링과 모델 최적화를 통해 의미 있는 성과를 달성했습니다. 

이번 경험은 향후 유사한 불균형 분류 문제나 추천 시스템 프로젝트에서 실질적인 가이드라인으로 활용될 수 있을 것으로 기대합니다.

---

## 기술 스택 요약

**언어**: Python 3.10+  

**주요 라이브러리**:
- 데이터 처리: Pandas, NumPy, polars
- 모델링 및 실험 추적: Scikit-learn, Weights & Biases

---

## 참고 자료

CTR 예측 모델링에 대한 추가 학습 자료:
- [Wide & Deep Learning](https://arxiv.org/abs/1606.07792)
- [DeepFM](https://arxiv.org/abs/1703.04247)
- [LightGBM 공식 문서](https://lightgbm.readthedocs.io/)
