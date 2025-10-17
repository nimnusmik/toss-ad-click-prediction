# Toss 광고 클릭 예측 모델

> Toss 사용자의 광고 클릭 여부를 예측하는 재현 가능한 파이프라인 구축: 탐색적 데이터 분석부터 프로덕션 배포까지

## 프로젝트 개요

본 저장소는 Toss 광고 캠페인을 위한 클릭률(CTR, Click-Through Rate) 예측 시스템의 기반 구조입니다. 최종 목표는 노출 로그 데이터를 수집하고, 예측 특성을 엔지니어링하며, 랭킹 모델을 학습하여 다운스트림 입찰 또는 추천 서비스에서 사용할 수 있는 신뢰할 수 있는 클릭 확률 추정치를 제공하는 것입니다.

현재 버전은 깔끔한 프로젝트 레이아웃, 템플릿 노트북, 파이프라인 확장 가이드를 제공합니다. 의도적으로 가볍게 설계하여 자체 로그 데이터를 연결하고 빠르게 반복 실험할 수 있습니다.

---

## 프로젝트 현황

### 리더보드 순위
약 700개 팀 중 상위 10% 달성

<img width="607" height="169" alt="Screenshot 2025-10-14 at 14 42 34" src="https://github.com/user-attachments/assets/e902bd46-9a95-46c3-95f5-c204ad24b9ed" />

- **현재 점수**: 0.34814
- **사용 모델**: DCN-v3

---

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

## 기술 스택 요약

**언어**: Python 3.10+  

**주요 라이브러리**:
- 데이터 처리: Pandas, NumPy
- 모델링: Scikit-learn
- 실험 추적: Weights & Biases

---

## 참고 자료

CTR 예측 모델링에 대한 추가 학습 자료:
- [Wide & Deep Learning](https://arxiv.org/abs/1606.07792)
- [DeepFM](https://arxiv.org/abs/1703.04247)
- [LightGBM 공식 문서](https://lightgbm.readthedocs.io/)
