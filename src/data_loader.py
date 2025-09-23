import pandas as pd
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
import torch.nn as nn

DEFAULT_CATEGORICAL_COLS = [
    'gender',
    'age_group',
    'inventory_id',
    'day_of_week',
    'hour',
]


def load_data(data_path):
    """데이터 로딩 및 전처리"""
    all_train = pl.read_parquet(f"{data_path}train_processed_3.parquet")
    test = pl.read_parquet(f"{data_path}test_processed_3.parquet")

    print("Train shape:", all_train.shape)
    print("Test shape:", test.shape)

    clicked_1 = all_train.filter(pl.col('clicked') == 1)
    clicked_0 = all_train.filter(pl.col('clicked') == 0).sample(
        n=len(clicked_1) * 10,
        seed=42,
    )

    train = pl.concat([clicked_1, clicked_0]).sample(
        fraction=1.0,
        seed=42,
    )

    train_pd = train.to_pandas()
    train = train_pd.copy()

    print("Train shape:", train.shape)
    print("Train clicked:0:", train[train['clicked'] == 0].shape)
    print("Train clicked:1:", train[train['clicked'] == 1].shape)

    return train, test


def _calc_embedding_dim(cardinality: int) -> int:
    """Heuristic로 임베딩 차원을 결정."""
    if cardinality == 18:  # inventory_id
        return 12
    if cardinality == 24:  # hour
        return 12
    return min(50, max(1, round(1.6 * (max(cardinality, 1) ** 0.56))))


def get_feature_columns(train_df, categorical_cols=None):
    """피처 컬럼 추출 및 카테고리 메타데이터 구성"""
    if categorical_cols is None:
        categorical_cols = DEFAULT_CATEGORICAL_COLS.copy()

    target_col = "clicked"
    seq_col = "seq"

    feature_exclude = {target_col, seq_col, "ID"}
    feature_exclude.update(categorical_cols)

    numeric_cols = [c for c in train_df.columns if c not in feature_exclude]

    categorical_info = {
        'columns': [],
        'maps': {},
        'cardinalities': [],
        'embedding_dims': [],
        'unique_counts': [],
    }

    for col in categorical_cols:
        if col not in train_df.columns:
            print(f"[WARN] Categorical column '{col}' missing in training data; skipping embedding.")
            continue

        uniques = pd.Series(train_df[col].dropna().unique())
        mapping = {val: idx for idx, val in enumerate(sorted(uniques))}
        cardinality = len(mapping)
        embedding_dim = _calc_embedding_dim(cardinality)

        categorical_info['columns'].append(col)
        categorical_info['maps'][col] = mapping
        categorical_info['cardinalities'].append(cardinality + 1)  # +1 for unknown token
        categorical_info['embedding_dims'].append(embedding_dim)
        categorical_info['unique_counts'].append(cardinality)

        print(
            f"Embedding setup - {col}: unique={cardinality}, dim={embedding_dim}, vocab={cardinality + 1}"
        )

    print("Num numeric features:", len(numeric_cols))
    print("Categorical features:", categorical_info['columns'])
    print("Sequence:", seq_col)
    print("Target:", target_col)

    return numeric_cols, categorical_info, seq_col, target_col


def collate_fn_train(batch):
    """훈련용 collate 함수"""
    nums, cats, seqs, ys = zip(*batch)

    nums = torch.stack(nums)
    ys = torch.stack(ys)

    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)

    if cats[0] is None:
        cats_tensor = None
    else:
        cats_tensor = torch.stack(cats)

    return nums, cats_tensor, seqs_padded, seq_lengths, ys


def collate_fn_infer(batch):
    """추론용 collate 함수"""
    nums, cats, seqs = zip(*batch)

    nums = torch.stack(nums)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)

    if cats[0] is None:
        cats_tensor = None
    else:
        cats_tensor = torch.stack(cats)

    return nums, cats_tensor, seqs_padded, seq_lengths


class ClickDataset(Dataset):
    """클릭 예측을 위한 데이터셋 클래스"""

    def __init__(
        self,
        df,
        numeric_cols,
        seq_col,
        target_col=None,
        categorical_info=None,
        has_target=True,
    ):
        self.df = df.reset_index(drop=True)
        self.numeric_cols = numeric_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.has_target = has_target

        self.categorical_info = categorical_info or {
            'columns': [],
            'maps': {},
            'cardinalities': [],
            'embedding_dims': [],
            'unique_counts': [],
        }
        self.categorical_cols = self.categorical_info.get('columns', [])
        self.category_maps = self.categorical_info.get('maps', {})

        self.numeric_data = (
            self.df[self.numeric_cols].fillna(0.0).to_numpy(dtype=np.float32)
            if self.numeric_cols
            else np.zeros((len(self.df), 0), dtype=np.float32)
        )

        if self.categorical_cols:
            cat_arrays = []
            for col in self.categorical_cols:
                mapping = self.category_maps.get(col, {})
                default_idx = len(mapping)
                mapped = (
                    self.df[col]
                    .map(mapping)
                    .fillna(default_idx)
                    .astype(np.int64)
                    .to_numpy()
                )
                cat_arrays.append(mapped)
            self.cat_data = np.stack(cat_arrays, axis=1)
        else:
            self.cat_data = None

        self.seq_strings = self.df[self.seq_col].astype(str).values

        if self.has_target and self.target_col is not None:
            self.y = self.df[self.target_col].astype(np.float32).values
        else:
            self.y = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_num = torch.from_numpy(self.numeric_data[idx])

        if self.cat_data is not None:
            x_cat = torch.from_numpy(self.cat_data[idx])
        else:
            x_cat = None

        s = self.seq_strings[idx]
        if s:
            arr = np.fromstring(s, sep=",", dtype=np.float32)
        else:
            arr = np.array([], dtype=np.float32)

        if arr.size == 0:
            arr = np.array([0.0], dtype=np.float32)

        seq = torch.from_numpy(arr)

        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_num, x_cat, seq, y
        else:
            return x_num, x_cat, seq
