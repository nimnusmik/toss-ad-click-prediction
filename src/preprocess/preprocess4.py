#%%
import os
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import polars as pl

# 데이터 불러오기
all_train = pl.read_parquet("../../data/train.parquet")
test = pl.read_parquet("../../data/test.parquet")

print("Train shape:", all_train.shape)
print("Test shape:", test.shape)

test.head(3)

#%%
train_clicked = all_train['clicked']
test_ID = test['ID'] 

#%%
# 1st filtering - drop_nulls and change types 
def simple_preprocess(train_df, test_df):
    """
    간단한 전처리 함수 - 기본적인 정리만 수행
    """
    # 데이터 타입 변환
    int_cols = ['gender','age_group','inventory_id','day_of_week','hour']
    
    train_clean = train_df.with_columns([
        pl.col(col).cast(pl.Float64).cast(pl.Int32) for col in int_cols
    ]).with_columns([
        # DCN 안정성을 위한 정규화 추가
        # inventory_id는 범위가 클 수 있으므로 log 변환 후 정규화
        (pl.col('inventory_id').log() / pl.col('inventory_id').log().max()).alias('inventory_id_normalized'),
        (pl.col('age_group') / pl.col('age_group').max()).alias('age_group_normalized')
    ])
    
    test_clean = test_df.with_columns([
        pl.col(col).cast(pl.Float64).cast(pl.Int32) for col in int_cols
    ]).with_columns([
        # 테스트 데이터도 동일한 정규화 적용 (훈련 데이터의 통계 사용)
        (pl.col('inventory_id').log() / train_clean['inventory_id'].log().max()).alias('inventory_id_normalized'),
        (pl.col('age_group') / train_clean['age_group'].max()).alias('age_group_normalized')
    ])
    
    print(f"전처리 완료 - train: {train_clean.shape}, test: {test_clean.shape}")
    return train_clean, test_clean

train, test = simple_preprocess(all_train, test)
#train.head(3)
#test.head(3)
print(train.shape, test.shape)




#%%
# 특성공학 한번에 처리 하는 함수 만들기
def add_dcn_optimized_features(df):
    """DCN 모델에 최적화된 시간 특성 인코딩"""
    return df.with_columns([
        # 타입 변환
        pl.col('day_of_week').cast(pl.Int32).alias('day_of_week'),
        pl.col('hour').cast(pl.Int32).alias('hour'),
    ]).with_columns([
        # ============ DCN에 최적화된 Cyclic Features ============
        
        # 1. 시간 Cyclic (24시간 주기) - DCN Cross Network가 상호작용 학습
        (2 * np.pi * pl.col('hour') / 24).sin().alias('hour_sin'),
        (2 * np.pi * pl.col('hour') / 24).cos().alias('hour_cos'),
        
        # 2. 요일 Cyclic (7일 주기) - 월요일과 일요일 연결성 표현
        (2 * np.pi * (pl.col('day_of_week') - 1) / 7).sin().alias('dow_sin'),
        (2 * np.pi * (pl.col('day_of_week') - 1) / 7).cos().alias('dow_cos'),
        
        # 3. 주간 시간 Cyclic (168시간 주기) - 장기 패턴 캡처
        # Cross Network가 이 특성들의 고차 상호작용을 자동 학습
        (2 * np.pi * ((pl.col('day_of_week') - 1) * 24 + pl.col('hour')) / 168).sin().alias('week_hour_sin'),
        (2 * np.pi * ((pl.col('day_of_week') - 1) * 24 + pl.col('hour')) / 168).cos().alias('week_hour_cos'),
        
        # ============ DCN Embedding용 범주형 특성 ============
        
        # 4. 클릭률 기반 시간대 (Embedding 처리용)
        # DCN의 임베딩 레이어에서 dense representation 학습
        pl.when(pl.col('hour').is_between(0, 4))
          .then(pl.lit(0))      # night_active
          .when(pl.col('hour').is_between(5, 8))
          .then(pl.lit(1))      # morning_prep  
          .when(pl.col('hour').is_between(9, 12))
          .then(pl.lit(2))      # work_focus
          .when(pl.col('hour').is_between(13, 16))
          .then(pl.lit(3))      # afternoon_active
          .otherwise(pl.lit(4)) # evening_leisure
          .alias("time_period_id"),
        
        # 5. 주말 여부 (Binary feature - DCN이 효율적으로 처리)
        ((pl.col("day_of_week") == 6) | (pl.col("day_of_week") == 7)).cast(pl.Int32).alias("is_weekend"),
        
        # ============ DCN Cross Network 활용을 위한 추가 특성 ============
        
        # 6. 시간 정규화 (0~1 범위) - Cross Network에서 상호작용 계산에 유리
        (pl.col('hour') / 23.0).alias('hour_normalized'),
        
        # 7. 요일 정규화 (0~1 범위)
        ((pl.col('day_of_week') - 1) / 6.0).alias('dow_normalized'),
        
        # 8. 클릭률 수준 (3단계) - 간단한 범주형으로 임베딩
        pl.when(pl.col('hour').is_between(0, 4) | pl.col('hour').is_between(17, 23))
          .then(pl.lit(2))      # high_ctr
          .when(pl.col('hour').is_between(13, 16))
          .then(pl.lit(1))      # medium_ctr  
          .otherwise(pl.lit(0)) # low_ctr
          .alias("ctr_level_id")
    ])#.drop(['day_of_week','hour'])

# cyclic 피처 추가
train_with_cyclic = add_dcn_optimized_features(train)
test_with_cyclic = add_dcn_optimized_features(test)

print("Cyclic 피처 추가 완료!")

#train_with_cyclic.head(3)
print(train_with_cyclic.columns)
print(train_with_cyclic.shape, test_with_cyclic.shape)


# %%
# seq 컬럼 처리 - 시퀀스에서 간단한 특성 추출
def process_seq_column(df):
    """seq 컬럼에서 유용한 특성들 추출"""
    return df.with_columns([
        # 시퀀스 길이
        pl.col('seq').str.count_matches(',').add(1).alias('seq_length'),
        
        (pl.col('seq').str.count_matches(',').add(1) / 100.0).alias('seq_length_norm'),
        
        # 첫 번째 값 (변환 실패하면 null)
        pl.col('seq').str.split(',').list.first().cast(pl.Int32, strict=False).alias('seq_first'),
        
        # 마지막 값
        pl.col('seq').str.split(',').list.last().cast(pl.Int32, strict=False).alias('seq_last'),
        
        # ============ 시퀀스 길이 범주화 (임베딩용) ============
        
        # 시퀀스 길이를 범주로 분류 - DCN 임베딩 레이어에서 처리
        pl.when(pl.col('seq').str.count_matches(',').add(1) <= 100)
          .then(pl.lit(0))      # 초단 시퀀스 (가장 많은 데이터, 낮은 CTR)
          .when(pl.col('seq').str.count_matches(',').add(1) <= 500)
          .then(pl.lit(1))      # 단 시퀀스 (중간 데이터, 낮은 CTR)
          .when(pl.col('seq').str.count_matches(',').add(1) <= 2000)
          .then(pl.lit(2))      # 중 시퀀스 (적은 데이터, 낮은 CTR)
          .when(pl.col('seq').str.count_matches(',').add(1) <= 5000)
          .then(pl.lit(3))      # 장 시퀀스 (매우 적은 데이터, CTR 상승 시작)
          .otherwise(pl.lit(4)) # 초장 시퀀스 (극소 데이터, 높은 CTR)
          .alias('seq_length_category'),

        # ============ 클릭률 수준별 시퀀스 분류 (단순화) ============

        pl.when(pl.col('seq').str.count_matches(',').add(1) >= 5000)
          .then(pl.lit(1))      # 고CTR 구간 (5000+)
          .otherwise(pl.lit(0)) # 저CTR 구간 (5000 미만)
          .alias('seq_high_ctr_flag'),
        
        # ============ 시퀀스 길이 로그 변환 (수치형) ============
        
        # 매우 넓은 범위 (0~15000+)를 로그 변환으로 압축
        # Cross Network에서 안정적인 학습을 위함
        (pl.col('seq').str.count_matches(',').add(1).log() / 
        pl.col('seq').str.count_matches(',').add(1).log().max()).alias('seq_length_log_norm')
    ])

def process_seq_column2(df, chunk_size: int = 200_000):
    """seq 컬럼을 청크 단위로 전처리해 메모리 사용량을 안정화"""
    if 'seq' not in df.columns:
        return df

    # 전체 로그 최댓값 계산 (정규화 스케일)
    seq_log_max = 0.0
    for chunk in df.iter_slices(chunk_size):
        chunk_log_max = (
            chunk.select(
                pl.col('seq')
                .str.count_matches(',')
                .add(1)
                .cast(pl.Float64)
                .log()
                .max()
                .alias('seq_log_max')
            )['seq_log_max'][0]
        )
        if chunk_log_max is not None and np.isfinite(chunk_log_max):
            seq_log_max = max(seq_log_max, chunk_log_max)

    if not np.isfinite(seq_log_max) or seq_log_max <= 0:
        seq_log_max = 1.0

    processed_chunks = []

    # 청크 단위로 특성 생성
    for chunk in df.iter_slices(chunk_size):
        chunk_processed = (
            chunk.with_columns(
                pl.col('seq').str.count_matches(',').add(1).alias('seq_length'),
            )
            .with_columns([
                (pl.col('seq_length') / 100.0).alias('seq_length_norm'),
                pl.col('seq')
                .str.split(',')
                .list.first()
                .cast(pl.Int32, strict=False)
                .alias('seq_first'),
                pl.col('seq')
                .str.split(',')
                .list.last()
                .cast(pl.Int32, strict=False)
                .alias('seq_last'),
                (pl.col('seq_length') >= 5000).cast(pl.Int32).alias('seq_high_ctr_flag'),
                pl.when(pl.col('seq_length') <= 1)
                .then(0.0)
                .otherwise(
                    pl.col('seq_length').cast(pl.Float64).log() / seq_log_max
                )
                .alias('seq_length_log_norm'),
            ])
        )

        processed_chunks.append(chunk_processed)

    return pl.concat(processed_chunks, how='vertical', rechunk=True)

# seq 처리 적용
X_train_processed = process_seq_column2(train_with_cyclic)
X_test_processed = process_seq_column2(test_with_cyclic)

print("seq 컬럼 처리 완료!")
print(f"X_train 최종 크기: {X_train_processed.shape}")
print(f"X_test 최종 크기: {X_test_processed.shape}")

# %%
#X_train_processed.head(3)
X_test_processed.head(3)




# %%
def process_essential_interaction(df):
    """ 특성공학inventory_id * gender , age_group * gender , inventory_id * age_group 
        hour * day_of_week', 'is_weekend * hour',  'age_group * hour'  """
    return df.with_columns([
        (pl.col('gender') * pl.col('age_group')).alias('gender_age_group'),
        (pl.col('inventory_id') * pl.col('gender')).alias('inventory_gender'),
        (pl.col('inventory_id') * pl.col('age_group')).alias('inventory_age_group'),
        (pl.col('inventory_id') * pl.col('gender') * pl.col('age_group')).alias('inventory_gender_age'),
        
        (pl.col('hour') * pl.col('day_of_week')).alias('hour_day_of_week'),
        (pl.col('is_weekend') * pl.col('hour')).alias('is_weekend_hour'),
        (pl.col('hour') * pl.col('age_group')).alias('hour_age_group'),
    ])

# 사용
X_train_processed = process_essential_interaction(X_train_processed)
X_test_processed = process_essential_interaction(X_test_processed)

X_test_processed.head(3)




#%%
# drop list 
drop_columns = [
    # feat_a
    'feat_a_1', 'feat_a_2', 'feat_a_3', 'feat_a_7', 'feat_a_8', 'feat_a_9',
    'feat_a_10', 'feat_a_11', 'feat_a_12', 'feat_a_13', 'feat_a_15', 'feat_a_16',

    # feat_b
    'feat_b_2',

    # feat_e
    'feat_e_4', 'feat_e_6',

    # gender
    #'gender',

    # history_a
    'history_a_7',

    # history_b
    'history_b_1', 'history_b_7', 'history_b_8', 'history_b_10',
    'history_b_13', 'history_b_18', 'history_b_20', 'history_b_21', 'history_b_25',

    # l_feat
    'l_feat_7', 'l_feat_17', 'l_feat_19', 'l_feat_20',
    'l_feat_22', 'l_feat_23', 'l_feat_24', 'l_feat_25', 'l_feat_26',
]

additional_drop = [
    #'l_feat_16',       # l_feat_1과 완전상관 
    'history_b_24',    # history 16과 완전 상관
    'feat_d_5',        # feat_d_1과 완전상관
    'feat_d_6',         # feat_d_1과 완전상관


    # feature importance bottop 10%
    'seq_high_ctr_flag', #0.000000
    'feat_b_1', #0.000484
    'history_b_19', #0.000924
    'history_b_14', #0.001050
    'history_b_9', #0.001128
    'feat_a_4', #0.001142
    'history_b_22', #0.001145
    'history_b_6', #0.001224
    'history_b_11', #0.001240
    'history_b_15', #0.001350
    'history_b_23', #0.001390
]


# 위 특성들 제거
X_train_selected = X_train_processed[[col for col in X_train_processed.columns if col not in drop_columns and col not in additional_drop]]
X_test_selected = X_test_processed[[col for col in X_test_processed.columns if col not in drop_columns and col not in additional_drop]]


print("선택된 특성으로 데이터 준비 완료!")
print(f"훈련 데이터 크기: {X_train_selected.shape}")
print(f"테스트 데이터 크기: {X_test_selected.shape}")

X_train_selected.head(3)
#X_test_selected.head(3)





# %%
def add_dcn_feature_engineering(df):

    return df.with_columns([
        # ============ 범주형 특성 개수 최적화 ============
        
        # gender 재인코딩 (0-based indexing)
        # DCN 임베딩에서는 0부터 시작하는 인덱스가 효율적
        (pl.col('gender') - 1).alias('gender_id'),
        
        # age_group 재인코딩 (0-based)
        (pl.col('age_group') - 1).alias('age_group_id'),
        
        # ============ 수치형 특성 안정화 ============
        
        # 모든 feat_ 컬럼들 정규화 (Cross Network 안정성)
        # feat_로 시작하는 컬럼들을 찾아서 min-max 정규화 적용
        *[
            ((pl.col(col) - pl.col(col).min()) / 
             (pl.col(col).max() - pl.col(col).min() + 1e-8)).alias(f"{col}_norm")
            for col in df.columns if col.startswith('feat_')
        ],
        
        # history_ 컬럼들도 정규화
        *[
            ((pl.col(col) - pl.col(col).min()) / 
             (pl.col(col).max() - pl.col(col).min() + 1e-8)).alias(f"{col}_norm")
            for col in df.columns if col.startswith('history_')
        ],
        
        # l_feat_ 컬럼들도 정규화
        *[
            ((pl.col(col) - pl.col(col).min()) / 
             (pl.col(col).max() - pl.col(col).min() + 1e-8)).alias(f"{col}_norm")
            for col in df.columns if col.startswith('l_feat_')
        ]


    ])


X_train_normalized = add_dcn_feature_engineering(X_train_selected)
X_test_normalized = add_dcn_feature_engineering(X_test_selected)



# %%
# 순서 정리 
if "clicked" in X_train_selected.columns:
    X_train_selected = X_train_selected.drop("clicked")

train_final = pl.concat([
    X_train_selected,                   # 선택된 특성들
    train_clicked.to_frame()            # 타겟 변수   
], how="horizontal")

if "ID" in X_test_selected.columns:
    X_test_selected = X_test_selected.drop("ID")

test_final = pl.concat([
    test_ID.to_frame(),               # ID 컬럼 (이미 저장해둠)
    X_test_selected                   # 선택된 특성들
], how="horizontal")


print(train_final.shape, test_final.shape)
# %%
#train_final.head(3)
test_final.head(3)

# %%
print(train_final.columns) 
print(test_final.columns)
# %%
# 모델 저장
train_final.write_parquet("../../data/processed/train_processed_4.parquet")
test_final.write_parquet("../../data/processed/test_processed_4.parquet")
# %%







