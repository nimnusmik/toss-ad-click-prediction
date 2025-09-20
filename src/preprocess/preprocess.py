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

#%%
# 1st filtering - drop_nulls and change types 
def simple_preprocess(train_df, test_df):
    """
    간단한 전처리 함수 - 기본적인 정리만 수행
    """
    # feat_e_3 컬럼 제거
    train_clean = train_df.drop('feat_e_3') 
    test_clean = test_df.drop('feat_e_3') 
    
    # 데이터 타입 변환
    int_cols = ['gender','age_group','inventory_id','day_of_week','hour']
    
    train_clean = train_clean.with_columns([
        pl.col(col).cast(pl.Float64).cast(pl.Int32) for col in int_cols
    ])
    
    test_clean = test_clean.with_columns([
        pl.col(col).cast(pl.Float64).cast(pl.Int32) for col in int_cols
    ])
    
    print(f"전처리 완료 - train: {train_clean.shape}, test: {test_clean.shape}")
    return train_clean, test_clean

train, test = simple_preprocess(all_train, test)

train_clicked = train['clicked']
test_ID = test['ID'] 

#%%
test.head(3)
#%%
train.head(3)

#%%
# 특성공학 한번에 처리 하는 함수 만들기
def add_cyclic_features(df):
    """데이터프레임에 cyclic 피처 추가"""
    return df.with_columns([
        # 요일 사이클릭 (7일 주기)
        (2 * np.pi * (pl.col('day_of_week') - 1) / 7).sin().alias('dow_sin'),
        (2 * np.pi * (pl.col('day_of_week') - 1) / 7).cos().alias('dow_cos'),
        
        # 시간 사이클릭 (24시간 주기)  
        (2 * np.pi * pl.col('hour') / 24).sin().alias('hour_sin'),
        (2 * np.pi * pl.col('hour') / 24).cos().alias('hour_cos'),
        
        # 복합 사이클릭 (168시간 주기)
        (2 * np.pi * ((pl.col('day_of_week') - 1) * 24 + pl.col('hour')) / 168).sin().alias('week_hour_sin'),
        (2 * np.pi * ((pl.col('day_of_week') - 1) * 24 + pl.col('hour')) / 168).cos().alias('week_hour_cos')
        
        # is_weekend 변수 추가
        ((pl.col("day_of_week") >= 6).cast(pl.Int32)).alias("is_weekend")
    ]).drop(['day_of_week','hour'])

# cyclic 피처 추가
train_with_cyclic = add_cyclic_features(train)
test_with_cyclic = add_cyclic_features(test)

print("Cyclic 피처 추가 완료!")
print(f"훈련 데이터: {train_with_cyclic.shape}")
print(f"테스트 데이터: {test_with_cyclic.shape}")


# %%
#train_with_cyclic.head(3)
#print(train_with_cyclic.columns)


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
        pl.col('seq').str.count_matches(',').add(1).alias('seq_unique_approx')
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
#%%
# 상관관계 상위 20개 특성
correlation_features = [
    'history_a_1', 'history_a_2', 'history_a_3', 'history_a_5', 'history_b_2',
    'history_b_30', 'history_b_1', 'inventory_id', 'history_b_10', 'feat_b_4',
    'feat_d_3', 'feat_e_1', 'history_b_17', 'history_b_15', 'history_b_9',
    'history_b_3', 'history_b_20', 'history_b_27', 'history_b_28', 'history_b_5'
]

# LightGBM 상위 20개 특성 (차트에서)
lightgbm_features = [
    'history_a_1', 'inventory_id', 'history_b_2', 'history_a_3', 'age_group',
    'feat_e_3', 'feat_d_4', 'l_feat_5', 'l_feat_2', 'l_feat_6', 'history_a_2',
    'l_feat_10', 'feat_c_8', 'l_feat_7', 'l_feat_14', 'l_feat_4', 'l_feat_9',
    'l_feat_15', 'l_feat_1', 'l_feat_12'
]

selected_columns = [
    'age_group', 
    'inventory_id',
    'seq', # 얘는 일단 추가
    'seq_length', 'seq_first', #'seq_last, 'seq_101_count', 'seq_unique_approx'
    #'day_of_week', 'hour',
    'dow_sin', 'dow_cos', 'hour_sin', 'hour_cos', 'week_hour_sin', 'week_hour_cos',

    'feat_b_4', 'feat_c_8', 'feat_d_3', 'feat_d_4', 'feat_e_1', #'feat_e_3', #결측치 많아도 땡겨

    'history_a_1', 'history_a_2', 'history_a_3', 'history_a_5', 

    'history_b_1',  'history_b_2', 'history_b_3', 'history_b_5', 'history_b_9', 'history_b_10', 
    'history_b_15', 'history_b_17','history_b_20', 'history_b_27', 'history_b_28', 'history_b_30', 
   
    'l_feat_1', 'l_feat_2', 'l_feat_4', 'l_feat_5', 'l_feat_6', 'l_feat_7', 'l_feat_9',
    'l_feat_10', 'l_feat_12', 'l_feat_14', 'l_feat_15',

    #'clicked',
]

# 선택된 특성들만 추출
X_train_selected =  X_train_processed[selected_columns]
X_test_selected = X_test_processed[selected_columns]


print("선택된 특성으로 데이터 준비 완료!")
print(f"훈련 데이터 크기: {X_train_selected.shape}")
print(f"테스트 데이터 크기: {X_test_selected.shape}")


# Mutual Information (MI) 기반 상호 작용 변수 추가, 10피처 쌍 선정.
# age_group × gender
# age_group × hour
# gender × hour
 














#%%
X_test_selected.head(3)
# %%
train_final = pl.concat([
    X_train_selected,                   # 선택된 특성들
    train_clicked.to_frame()            # 타겟 변수   
], how="horizontal")

test_final = pl.concat([
    test_ID.to_frame(),               # ID 컬럼 (이미 저장해둠)
    X_test_selected                   # 선택된 특성들
], how="horizontal")


print(train_final.shape, test_final.shape)
# %%
train_final.head(3)
# %%
test_final.head(3)
# %%
# 모델 저장
train_final.write_parquet("../../data/processed/train_processed_2.parquet")
test_final.write_parquet("../../data/processed/test_processed_2.parquet")
# %%







