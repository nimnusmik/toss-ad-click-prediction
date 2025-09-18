#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[15]:


#get_ipython().system('pip install -r "../requirements.txt"')


# In[16]:


import pandas as pd
import numpy as np
import os
import random

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# ## Setting

# 

# In[17]:


CFG = {
    'BATCH_SIZE': 4096,
    'EPOCHS': 20,
    'LEARNING_RATE': 1e-3,
    'SEED' : 42
}
device = "cuda" if torch.cuda.is_available() else "cpu"


# In[18]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED']) # Seed 고정


# ## Data Load

# In[19]:


import polars as pl

all_train = pl.read_parquet("../data/processed/train_processed.parquet")
test = pl.read_parquet("../data/processed/test_processed.parquet")

print("Train shape:", all_train.shape)
print("Test shape:", test.shape)


# 

# In[20]:


# 데이터 로드
#all_train = pd.read_parquet("../data/raw/train.parquet", engine="pyarrow")
#test = pd.read_parquet("../data/raw/test.parquet", engine="pyarrow").drop(columns=['ID'])

#print("Train shape:", all_train.shape)
#print("Test shape:", test.shape)


# ## Data Down-Sampling

# In[21]:


# Polars 방식으로 데이터 필터링 및 샘플링
# filter() 메서드를 사용해 조건에 맞는 데이터 추출
clicked_1 = all_train.filter(pl.col('clicked') == 1)

# clicked == 0 데이터에서 샘플링
# Polars의 sample() 메서드 사용 (pandas와 매개변수명이 다름)
clicked_0 = all_train.filter(pl.col('clicked') == 0).sample(
    n=len(clicked_1) * 2,  # 샘플 개수 지정
    seed=42  # random_state 대신 seed 사용
)

# 두 데이터프레임 합치기
# pl.concat()으로 데이터프레임 연결 후 샘플링으로 셔플
train = pl.concat([clicked_1, clicked_0]).sample(
    fraction=1.0,  # frac 대신 fraction 사용
    seed=42
)


# In[22]:


train_pd = train.to_pandas()

train = train_pd.copy()
print("Train shape:", train.shape)
print("Train clicked:0:", train[train['clicked']==0].shape)
print("Train clicked:1:", train[train['clicked']==1].shape)


# ## Data Column Setting

# In[23]:


# Target / Sequence
target_col = "clicked"
seq_col = "seq"

# 학습에 사용할 피처: ID/seq/target 제외, 나머지 전부
FEATURE_EXCLUDE = {target_col, seq_col, "ID"}
feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]

print("Num features:", len(feature_cols))
print("Sequence:", seq_col)
print("Target:", target_col)


# ## Define Custom Dataset

# In[24]:


class ClickDataset(Dataset):
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


# In[25]:


def collate_fn_train(batch):
    xs, seqs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)  # 빈 시퀀스 방지
    return xs, seqs_padded, seq_lengths, ys

def collate_fn_infer(batch):
    xs, seqs = zip(*batch)
    xs = torch.stack(xs)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return xs, seqs_padded, seq_lengths


# ## Define Model Architecture

# In[26]:


class TabularSeqModel(nn.Module):
    def __init__(self, d_features, lstm_hidden=32, hidden_units=[1024, 512, 256, 128], dropout=0.2):
        super().__init__()
        # 모든 비-시퀀스 피처에 BN
        self.bn_x = nn.BatchNorm1d(d_features)
        # seq: 숫자 시퀀스 → LSTM
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)

        # 최종 MLP
        input_dim = d_features + lstm_hidden
        layers = []
        for h in hidden_units:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_feats, x_seq, seq_lengths):
        # 비-시퀀스 피처
        x = self.bn_x(x_feats)

        # 시퀀스 → LSTM (pack)
        x_seq = x_seq.unsqueeze(-1)  # (B, L, 1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = h_n[-1]                  # (B, lstm_hidden)

        z = torch.cat([x, h], dim=1)
        return self.mlp(z).squeeze(1)  # logits


# ## Train / Validation

# In[27]:


def train_model(train_df, feature_cols, seq_col, target_col,
                batch_size=512, epochs=3, lr=1e-3, device="cuda"):

    # 1) split
    tr_df, va_df = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=True)

    # 2) Dataset / Loader (l_max 인자 제거)
    train_dataset = ClickDataset(tr_df, feature_cols, seq_col, target_col, has_target=True)
    val_dataset   = ClickDataset(va_df, feature_cols, seq_col, target_col, has_target=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn_train)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn_train)

    # 3) 모델
    d_features = len(feature_cols)
    model = TabularSeqModel(d_features=d_features, lstm_hidden=64, hidden_units=[256,128], dropout=0.2).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4) Loop
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for xs, seqs, seq_lens, ys in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
            optimizer.zero_grad()
            logits = model(xs, seqs, seq_lens)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * ys.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xs, seqs, seq_lens, ys in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
                logits = model(xs, seqs, seq_lens)
                loss = criterion(logits, ys)
                val_loss += loss.item() * len(ys)
        val_loss /= len(val_dataset)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model


# ## Run!!

# In[ ]:


model = train_model(
    train_df=train,
    feature_cols=feature_cols,
    seq_col=seq_col,
    target_col=target_col,
    batch_size=CFG['BATCH_SIZE'],
    epochs=CFG['EPOCHS'],
    lr=CFG['LEARNING_RATE'],
    device=device
)


# In[ ]:


torch.cuda.empty_cache()  # 사용하지 않는 GPU 메모리 해제


# ## Inference

# In[ ]:


test_pd = test.to_pandas()

# 1) Dataset/Loader
test_ds = ClickDataset(test_pd, feature_cols, seq_col, has_target=False)
test_ld = DataLoader(test_ds, batch_size=CFG['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn_infer)

# 2) Predict
model.eval()
outs = []
with torch.no_grad():
    for xs, seqs, lens in tqdm(test_ld, desc="Inference"):
        xs, seqs, lens = xs.to(device), seqs.to(device), lens.to(device)
        outs.append(torch.sigmoid(model(xs, seqs, lens)).cpu())

test_preds = torch.cat(outs).numpy()


# ## Submission

# In[ ]:


submit = pd.read_csv('../data/sample_submission.csv')
submit['clicked'] = test_preds


# In[ ]:


submit.to_csv('../data/output/baseline_submit2.csv', index=False)


# In[ ]:




