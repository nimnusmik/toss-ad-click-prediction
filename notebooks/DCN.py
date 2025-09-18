
# In[1]:
import os
import random
import pandas as pd
import numpy as np
import polars as pl

from tqdm import tqdm
from sklearn.metrics import average_precision_score, log_loss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# In[2]:

CFG = {
    'BATCH_SIZE': 4096,
    'EPOCHS': 20,
    'LEARNING_RATE': 1e-3,
    'SEED' : 42
}
device = "cuda" if torch.cuda.is_available() else "cpu"


# In[3]:
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED']) # Seed 고정

# In[4]:

all_train = pl.read_parquet("../data/processed/train_processed.parquet")
test = pl.read_parquet("../data/processed/test_processed.parquet")

print("Train shape:", all_train.shape)
print("Test shape:", test.shape)


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


# In[5]:

train_pd = train.to_pandas()
train = train_pd.copy()

print("Train shape:", train.shape)
print("Train clicked:0:", train[train['clicked']==0].shape)
print("Train clicked:1:", train[train['clicked']==1].shape)

# In[6]:
# Target / Sequence
target_col = "clicked"
seq_col = "seq"

# 학습에 사용할 피처: ID/seq/target 제외, 나머지 전부
FEATURE_EXCLUDE = {target_col, seq_col, "ID"}
feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]

print("Num features:", len(feature_cols))
print("Sequence:", seq_col)
print("Target:", target_col)

# In[7]:

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

# In[8]:
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

# In[9]:
class CrossNetwork(nn.Module):
    """
    DCN의 Cross Network 부분
    피처 간의 상호작용을 명시적으로 학습하는 네트워크
    """
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # 각 레이어마다 weight와 bias 파라미터 생성
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)
        ])
        
    def forward(self, x0):
        """
        x0: 초기 입력 (batch_size, input_dim)
        각 레이어에서 x_l+1 = x_0 * (W_l * x_l + b_l) + x_l 계산
        """
        x_l = x0  # 현재 레이어의 출력
        
        for layer in self.cross_layers:
            # Cross operation: x_l+1 = x_0 * (W_l * x_l + b_l) + x_l
            xl_w = layer(x_l)  # (batch_size, 1)
            x_l = x0 * xl_w + x_l  # element-wise multiplication + residual connection
            
        return x_l


# In[10]:
class DCNModel(nn.Module):
    """
    Deep & Cross Network 모델
    Cross Network와 Deep Network를 병렬로 연결한 후 최종 출력층에서 합침
    """
    def __init__(self, d_features, lstm_hidden=64, cross_layers=3, 
                 deep_hidden=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        # 입력 피처 정규화
        self.bn_input = nn.BatchNorm1d(d_features)
        
        # LSTM for sequence features (기존과 동일)
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)
        
        # 전체 입력 차원 (tabular features + lstm output)
        total_input_dim = d_features + lstm_hidden
        
        # Cross Network: 피처 간 상호작용 학습
        self.cross_net = CrossNetwork(total_input_dim, cross_layers)
        
        # Deep Network: 비선형 변환 학습
        deep_layers = []
        input_dim = total_input_dim
        
        for hidden_dim in deep_hidden:
            deep_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # 배치 정규화 추가
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
            
        self.deep_net = nn.Sequential(*deep_layers)
        
        # 최종 출력층: Cross Network + Deep Network 결합
        final_input_dim = total_input_dim + input_dim  # cross output + deep output
        self.final_layer = nn.Linear(final_input_dim, 1)
        
    def forward(self, x_feats, x_seq, seq_lengths):
        # 1. Tabular features 정규화
        x_tab = self.bn_input(x_feats)
        
        # 2. Sequence features 처리 (LSTM)
        x_seq = x_seq.unsqueeze(-1)  # (batch_size, seq_len, 1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        x_lstm = h_n[-1]  # 마지막 hidden state (batch_size, lstm_hidden)
        
        # 3. 모든 피처 결합
        x_combined = torch.cat([x_tab, x_lstm], dim=1)
        
        # 4. Cross Network와 Deep Network 병렬 처리
        cross_output = self.cross_net(x_combined)  # 피처 상호작용 학습
        deep_output = self.deep_net(x_combined)    # 비선형 변환 학습
        
        # 5. 두 네트워크 출력 결합
        final_input = torch.cat([cross_output, deep_output], dim=1)
        logits = self.final_layer(final_input)
        
        return logits.squeeze(1)

def calculate_metrics(y_true, y_pred_proba):
    """
    대회 평가지표 계산
    AP (Average Precision): 예측 확률에 대한 평균 정밀도
    WLL (Weighted LogLoss): 클래스 불균형을 고려한 가중 로그 손실
    """
    
    # 1. Average Precision 계산
    ap_score = average_precision_score(y_true, y_pred_proba)
    
    # 2. Weighted LogLoss 계산 (50:50 비율로 가중치 조정)
    # 클래스별 샘플 수 계산
    unique_classes = np.unique(y_true)
    class_counts = np.bincount(y_true.astype(int))
    
    # 50:50 비율로 가중치 계산
    total_samples = len(y_true)
    weight_0 = 0.5 / (class_counts[0] / total_samples)  # class 0에 대한 가중치
    weight_1 = 0.5 / (class_counts[1] / total_samples)  # class 1에 대한 가중치
    
    # 각 샘플에 대한 가중치 배열 생성
    sample_weights = np.where(y_true == 0, weight_0, weight_1)
    
    # 가중 로그 손실 계산
    wll_score = log_loss(y_true, y_pred_proba, sample_weight=sample_weights)
    
    # 최종 점수: AP와 WLL의 평균 (대회 규칙에 따라)
    final_score = (ap_score + (1 - wll_score)) / 2
    
    return {
        'AP': ap_score,
        'WLL': wll_score, 
        'Final_Score': final_score
    }

def train_dcn_model(train_df, feature_cols, seq_col, target_col,
                    batch_size=512, epochs=10, lr=1e-3, device="cuda"):
    """
    DCN 모델 훈련 함수
    """
    from sklearn.model_selection import train_test_split
    
    # 1. 데이터 분할
    tr_df, va_df = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=True)
    print(f"Train: {len(tr_df)}, Validation: {len(va_df)}")
    
    # 2. Dataset과 DataLoader 생성 (기존 코드 재사용)
    train_dataset = ClickDataset(tr_df, feature_cols, seq_col, target_col, has_target=True)
    val_dataset = ClickDataset(va_df, feature_cols, seq_col, target_col, has_target=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_train)
    
    # 3. DCN 모델 초기화
    d_features = len(feature_cols)
    model = DCNModel(
        d_features=d_features,
        lstm_hidden=64,
        cross_layers=3,          # Cross Network 레이어 수
        deep_hidden=[512, 256, 128],  # Deep Network 구조
        dropout=0.3
    ).to(device)
    
    print(f"Model initialized with {d_features} features")
    
    # 4. 손실함수와 옵티마이저
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 학습률 스케줄러 추가
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5, 
    )
    
    best_val_score = 0
    
    # 5. 훈련 루프
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for xs, seqs, seq_lens, ys in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
            
            optimizer.zero_grad()
            logits = model(xs, seqs, seq_lens)
            loss = criterion(logits, ys)
            loss.backward()
            
            # Gradient clipping (gradient exploding 방지)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * len(ys)
            
            # 평가지표 계산을 위한 예측값 저장
            with torch.no_grad():
                train_preds.extend(torch.sigmoid(logits).cpu().numpy())
                train_targets.extend(ys.cpu().numpy())
        
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for xs, seqs, seq_lens, ys in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
                
                logits = model(xs, seqs, seq_lens)
                loss = criterion(logits, ys)
                val_loss += loss.item() * len(ys)
                
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(ys.cpu().numpy())
        
        val_loss /= len(val_dataset)
        
        # 평가지표 계산
        train_metrics = calculate_metrics(np.array(train_targets), np.array(train_preds))
        val_metrics = calculate_metrics(np.array(val_targets), np.array(val_preds))
        
        # 학습률 스케줄러 업데이트
        scheduler.step(val_loss)
        
        print(f"[Epoch {epoch}]")
        print(f"  Train - Loss: {train_loss:.4f}, AP: {train_metrics['AP']:.4f}, WLL: {train_metrics['WLL']:.4f}, Final: {train_metrics['Final_Score']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, AP: {val_metrics['AP']:.4f}, WLL: {val_metrics['WLL']:.4f}, Final: {val_metrics['Final_Score']:.4f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 베스트 모델 저장
        if val_metrics['Final_Score'] > best_val_score:
            best_val_score = val_metrics['Final_Score']
            torch.save(model.state_dict(), 'best_dcn_model.pth')
            print(f"  ★ New best validation score: {best_val_score:.4f}")
        
        print("-" * 80)
    
    return model


# In[11]:
model = train_dcn_model(
    train_df=train,
    feature_cols=feature_cols,
    seq_col=seq_col,
    target_col=target_col,
    batch_size=CFG['BATCH_SIZE'],
    epochs=CFG['EPOCHS'],
    lr=CFG['LEARNING_RATE'],
    device=device
)


# In[12]:
torch.cuda.empty_cache()  # 사용하지 않는 GPU 메모리 해제



# In[13]:
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


# In[14]:

submit = pd.read_csv('../data/sample_submission.csv')
submit['clicked'] = test_preds


# In[15]:
submit.to_csv('../data/output/baseline_submit3.csv', index=False)
