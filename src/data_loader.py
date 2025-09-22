import pandas as pd
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def load_data(data_path):
    """데이터 로딩 및 전처리"""
    # 데이터 로드
    all_train = pl.read_parquet(f"{data_path}train_processed_3.parquet")
    test = pl.read_parquet(f"{data_path}test_processed_3.parquet")
    
    print("Train shape:", all_train.shape)
    print("Test shape:", test.shape)
    
    # 클래스 불균형 해결을 위한 샘플링
    clicked_1 = all_train.filter(pl.col('clicked') == 1)
    
    # clicked == 0 데이터에서 샘플링
    clicked_0 = all_train.filter(pl.col('clicked') == 0).sample(
        n=len(clicked_1) * 10,   # 1:10 비율 해보기
        seed=42  # random_state 대신 seed 사용
    )
    
    # 두 데이터프레임 합치기
    train = pl.concat([clicked_1, clicked_0]).sample(
        fraction=1.0,  # frac 대신 fraction 사용
        seed=42
    )
    
    # Polars를 Pandas로 변환
    train_pd = train.to_pandas()
    train = train_pd.copy()
    
    print("Train shape:", train.shape)
    print("Train clicked:0:", train[train['clicked']==0].shape)
    print("Train clicked:1:", train[train['clicked']==1].shape)
    
    return train, test

def get_feature_columns(train_df):
    """피처 컬럼 추출"""
    target_col = "clicked"
    seq_col = "seq"
    
    # 학습에 사용할 피처: ID/seq/target 제외, 나머지 전부
    FEATURE_EXCLUDE = {target_col, seq_col, "ID"}
    feature_cols = [c for c in train_df.columns if c not in FEATURE_EXCLUDE]
    
    print("Num features:", len(feature_cols))
    print("Sequence:", seq_col)
    print("Target:", target_col)
    
    return feature_cols, seq_col, target_col

def collate_fn_train(batch):
    """훈련용 collate 함수"""
    xs, seqs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)  # 빈 시퀀스 방지
    return xs, seqs_padded, seq_lengths, ys

def collate_fn_infer(batch):
    """추론용 collate 함수"""
    xs, seqs = zip(*batch)
    xs = torch.stack(xs)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return xs, seqs_padded, seq_lengths

class ClickDataset(Dataset):
    """클릭 예측을 위한 데이터셋 클래스"""
    def __init__(self, df, feature_cols, seq_col, target_col=None, has_target=True):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.has_target = has_target

        # 비-시퀀스 피처: 전부 연속값으로
        self.X = self.df[self.feature_cols].astype(float).fillna(0).values

        # 시퀀스: 문자열 그대로 보관 (lazy 파싱)
        self.seq_strings = self.df[self.seq_col].astype(str).values

        if self.has_target:
            self.y = self.df[self.target_col].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float)
        # 전체 시퀀스 사용 (빈 시퀀스만 방어)
        s = self.seq_strings[idx]
        if s:
            arr = np.fromstring(s, sep=",", dtype=np.float32)
        else:
            arr = np.array([], dtype=np.float32)

        if arr.size == 0:
            arr = np.array([0.0], dtype=np.float32)  # 빈 시퀀스 방어

        seq = torch.from_numpy(arr)  # shape (seq_len,)

        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float)
            return x, seq, y
        else:
            return x, seq
