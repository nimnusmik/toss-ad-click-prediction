import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

from model import DCNModel
from data_loader import ClickDataset, collate_fn_train
from evaluate import calculate_metrics

def train_dcn_model(train_df, feature_cols, seq_col, target_col,
                    batch_size=512, epochs=10, lr=1e-3, device="cuda"):
    """
    DCN 모델 훈련 함수
    """
    
    # 1. 데이터 분할
    tr_df, va_df = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=True)
    print(f"Train: {len(tr_df)}, Validation: {len(va_df)}")
    
    # 2. Dataset과 DataLoader 생성
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
