import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

from model import DCNModel
from data_loader import ClickDataset, collate_fn_train
from evaluate import calculate_metrics

def weighted_binary_crossentropy(y_true, y_pred, weights):
    """
    가중치가 적용된 binary crossentropy 손실
    """
    # Sigmoid를 적용하지 않은 logits을 입력받음
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    weighted_bce = (bce * weights).mean()
    return weighted_bce


def pairwise_ranking_loss(y_true, y_pred, margin=1.0):
    """
    AP 최적화를 위한 pairwise ranking loss
    positive 샘플이 negative 샘플보다 높은 점수를 받도록 학습
    """
    # positive와 negative 샘플 분리
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    
    pos_scores = y_pred[pos_mask]
    neg_scores = y_pred[neg_mask]
    
    # 모든 positive-negative 쌍에 대해 ranking loss 계산
    pos_expanded = pos_scores.unsqueeze(1)  # (num_pos, 1)
    neg_expanded = neg_scores.unsqueeze(0)  # (1, num_neg)
    
    # Hinge loss: max(0, margin - (pos_score - neg_score))
    diff = pos_expanded - neg_expanded  # (num_pos, num_neg)
    hinge_loss = F.relu(margin - diff).mean()
    
    return hinge_loss


def combined_loss(y_true, y_pred, alpha=0.7, margin=1.0):
    """
    WLL + Ranking Loss 복합 손실함수
    
    Args:
        y_true: 실제 라벨 (0 or 1)
        y_pred: 모델 logits (sigmoid 적용 전)
        alpha: WLL과 ranking loss 간 균형 (0.5-0.8 권장)
        margin: ranking loss의 margin
    """
    # 안전장치: 분모가 0이 되지 않도록
    y_mean = y_true.mean().clamp(min=1e-7, max=1-1e-7)
    
    # WLL with class balancing
    pos_weight = 0.5 / y_mean
    neg_weight = 0.5 / (1 - y_mean)
    weights = y_true * pos_weight + (1 - y_true) * neg_weight
    
    wll = weighted_binary_crossentropy(y_true, y_pred, weights)
    ranking = pairwise_ranking_loss(y_true, y_pred, margin=margin)
    
    total_loss = alpha * wll + (1 - alpha) * ranking
    return total_loss

def train_dcn_model(train_df, feature_cols, seq_col, target_col,
                    batch_size=512, epochs=10, lr=1e-3, device="cuda", 
                    alpha=0.7, margin=1.0):
    """
    DCN 모델 훈련 함수 (combined_loss 적용)
    
    Args:
        alpha: WLL과 ranking loss 간 균형 (기본값: 0.7)
        margin: ranking loss의 margin (기본값: 1.0)
    """
    
    # 1. 데이터 분할
    tr_df, va_df = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=True)
    print(f"Train: {len(tr_df)}, Validation: {len(va_df)}")
    
    # 클릭률 확인
    train_click_rate = tr_df[target_col].mean()
    val_click_rate = va_df[target_col].mean()
    print(f"Train click rate: {train_click_rate:.4f}")
    print(f"Val click rate: {val_click_rate:.4f}")
    print(f"Combined Loss - Alpha: {alpha}, Margin: {margin}")
    
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
        cross_layers=3,
        deep_hidden=[512, 256, 128],
        dropout=0.3
    ).to(device)
    
    print(f"Model initialized with {d_features} features")
    
    # 4. 옵티마이저와 스케줄러
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5  # Final Score 최대화
    )
    
    best_val_score = 0
    patience_counter = 0
    early_stopping_patience = 5
    
    # 5. 훈련 루프
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for xs, seqs, seq_lens, ys in train_pbar:
            xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
            
            optimizer.zero_grad()
            logits = model(xs, seqs, seq_lens)
            
            # combined_loss 적용
            try:
                loss = combined_loss(ys.float(), logits.float(), alpha=alpha, margin=margin)
            except Exception as e:
                # Fallback to BCE if combined loss fails
                print(f"Combined loss failed: {e}, using BCE")
                loss = F.binary_cross_entropy_with_logits(logits, ys.float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(ys)
            
            with torch.no_grad():
                train_preds.extend(torch.sigmoid(logits).cpu().numpy())
                train_targets.extend(ys.cpu().numpy())
            
            # Progress bar update
            train_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for xs, seqs, seq_lens, ys in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
                
                logits = model(xs, seqs, seq_lens)
                
                try:
                    loss = combined_loss(ys.float(), logits.float(), alpha=alpha, margin=margin)
                except:
                    loss = F.binary_cross_entropy_with_logits(logits, ys.float())
                
                val_loss += loss.item() * len(ys)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(ys.cpu().numpy())
        
        val_loss /= len(val_dataset)
        
        # 평가지표 계산
        train_metrics = calculate_metrics(np.array(train_targets), np.array(train_preds))
        val_metrics = calculate_metrics(np.array(val_targets), np.array(val_preds))
        
        scheduler.step(val_metrics['Final_Score'])
        
        print(f"[Epoch {epoch}]")
        print(f"  Train - Loss: {train_loss:.4f}, AP: {train_metrics['AP']:.4f}, WLL: {train_metrics['WLL']:.4f}, Final: {train_metrics['Final_Score']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, AP: {val_metrics['AP']:.4f}, WLL: {val_metrics['WLL']:.4f}, Final: {val_metrics['Final_Score']:.4f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 베스트 모델 저장
        if val_metrics['Final_Score'] > best_val_score:
            best_val_score = val_metrics['Final_Score']
            torch.save(model.state_dict(), 'best_dcn_model.pth')
            print(f"  ★ New best validation score: {best_val_score:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 조기 종료
        if patience_counter >= early_stopping_patience:
            print(f"  Early stopping at epoch {epoch}")
            break
        
        print("-" * 80)
    
    print(f"Training completed! Best validation score: {best_val_score:.4f}")
    return model

