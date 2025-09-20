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
    ])
    
    test_clean = test_df.with_columns([
        pl.col(col).cast(pl.Float64).cast(pl.Int32) for col in int_cols
    ])
    
    print(f"전처리 완료 - train: {train_clean.shape}, test: {test_clean.shape}")
    return train_clean, test_clean

train, test = simple_preprocess(all_train, test)
#train.head(3)
#test.head(3)
print(train.shape, test.shape)




#%%
# 특성공학 한번에 처리 하는 함수 만들기
def add_cyclic_features(df):
    """데이터프레임에 cyclic 피처 추가"""
    return df.with_columns([
        # 타입 변환 먼저
        pl.col('day_of_week').cast(pl.Int32).alias('day_of_week'),
        pl.col('hour').cast(pl.Int32).alias('hour'),
    ]).with_columns([
        # 요일 사이클릭 (7일 주기)
        (2 * np.pi * (pl.col('day_of_week') - 1) / 7).sin().alias('dow_sin'),
        (2 * np.pi * (pl.col('day_of_week') - 1) / 7).cos().alias('dow_cos'),
        
        # 시간 사이클릭 (24시간 주기)  
        (2 * np.pi * pl.col('hour') / 24).sin().alias('hour_sin'),
        (2 * np.pi * pl.col('hour') / 24).cos().alias('hour_cos'),
        
        # 복합 사이클릭 (168시간 주기)
        (2 * np.pi * ((pl.col('day_of_week') - 1) * 24 + pl.col('hour')) / 168).sin().alias('week_hour_sin'),
        (2 * np.pi * ((pl.col('day_of_week') - 1) * 24 + pl.col('hour')) / 168).cos().alias('week_hour_cos'),
        
        # is_weekend 변수 추가
        ((pl.col("day_of_week") == 6) | (pl.col("day_of_week") == 7)).cast(pl.Int32).alias("is_weekend")
    ])#.drop(['day_of_week','hour'])

# cyclic 피처 추가
train_with_cyclic = add_cyclic_features(train)
test_with_cyclic = add_cyclic_features(test)

print("Cyclic 피처 추가 완료!")

#train_with_cyclic.head(3)
#print(train_with_cyclic.columns)
print(train.shape, test.shape)


# %%
# seq 컬럼 처리 - 시퀀스에서 간단한 특성 추출
def process_seq_column(df):
    """seq 컬럼에서 유용한 특성들 추출"""
    return df.with_columns([
        # 시퀀스 길이
        pl.col('seq').str.count_matches(',').add(1).alias('seq_length'),
        
        # 첫 번째 값 (변환 실패하면 null)
        pl.col('seq').str.split(',').list.first().cast(pl.Int32, strict=False).alias('seq_first'),
        
        # 마지막 값
        pl.col('seq').str.split(',').list.last().cast(pl.Int32, strict=False).alias('seq_last'),
        
        # 특정 패턴 개수 (예: '101' 패턴)
        #pl.col('seq').str.count_matches('101').alias('seq_101_count'),
        
        # 고유값 대략 개수 (','로 구분된 값들의 수)
        #pl.col('seq').str.count_matches(',').add(1).alias('seq_unique_approx')
    ])

# seq 처리 적용
X_train_processed = process_seq_column(train_with_cyclic)
X_test_processed = process_seq_column(test_with_cyclic)

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
    'gender',

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
    'l_feat_16',       # l_feat_1과 완전상관 
    'history_b_24',    # history 16과 완전 상관
    'feat_d_5',        # feat_d_1과 완전상관
    'feat_d_6'         # feat_d_1과 완전상관
]


# 위 특성들 제거
X_train_selected = X_train_processed[[col for col in X_train_processed.columns if col not in drop_columns and additional_drop]]
X_test_selected  = X_test_processed[[col for col in X_test_processed.columns if col not in drop_columns and additional_drop]]


print("선택된 특성으로 데이터 준비 완료!")
print(f"훈련 데이터 크기: {X_train_selected.shape}")
print(f"테스트 데이터 크기: {X_test_selected.shape}")

X_train_selected.head(3)
#X_test_selected.head(3)


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
# 모델 저장
train_final.write_parquet("../../data/processed/train_processed_2.parquet")
test_final.write_parquet("../../data/processed/test_processed_2.parquet")
# %%







