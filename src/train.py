import copy
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold  
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from model import DCNModel
from data_loader import ClickDataset, collate_fn_train
from evaluate import calculate_metrics

def weighted_binary_crossentropy(y_true, y_pred, weights):
    """가중치가 적용된 binary crossentropy 손실"""
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    weighted_bce = (bce * weights).mean()
    return weighted_bce

def pairwise_ranking_loss(y_true, y_pred, margin=1.0):
    """AP 최적화를 위한 pairwise ranking loss"""
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    
    pos_scores = y_pred[pos_mask]
    neg_scores = y_pred[neg_mask]
    
    pos_expanded = pos_scores.unsqueeze(1)
    neg_expanded = neg_scores.unsqueeze(0)
    
    diff = pos_expanded - neg_expanded
    hinge_loss = F.relu(margin - diff).mean()
    
    return hinge_loss

def combined_loss(y_true, y_pred, alpha=0.7, margin=1.0):
    """WLL + Ranking Loss 복합 손실함수"""
    y_mean = y_true.mean().clamp(min=1e-7, max=1-1e-7)
    
    pos_weight = 0.5 / y_mean
    neg_weight = 0.5 / (1 - y_mean)
    weights = y_true * pos_weight + (1 - y_true) * neg_weight
    
    wll = weighted_binary_crossentropy(y_true, y_pred, weights)
    ranking = pairwise_ranking_loss(y_true, y_pred, margin=margin)
    
    total_loss = alpha * wll + (1 - alpha) * ranking
    return total_loss


def train_single_fold(fold_num, train_idx, val_idx, train_df, feature_cols, seq_col, target_col,
                     batch_size=512, epochs=10, lr=1e-3, device="cuda", alpha=0.7, margin=1.0,
                     checkpoint_dir=None):
    """단일 fold 훈련 함수"""
    
    print(f"\n{'='*20} FOLD {fold_num} {'='*20}")
    
    # 1. fold별 데이터 분할
    tr_df = train_df.iloc[train_idx].reset_index(drop=True)
    va_df = train_df.iloc[val_idx].reset_index(drop=True)
    
    # 클릭률 확인 (불균형 데이터 검증)
    train_click_rate = tr_df[target_col].mean()
    val_click_rate = va_df[target_col].mean()
    print(f"Train: {len(tr_df)} samples, Click rate: {train_click_rate:.4f}")
    print(f"Val: {len(va_df)} samples, Click rate: {val_click_rate:.4f}")
    
    # 클릭률 차이가 크면 경고
    if abs(train_click_rate - val_click_rate) > 0.005:
        print(f"⚠️  Warning: Click rate difference: {abs(train_click_rate - val_click_rate):.4f}")
    
    # 2. Dataset과 DataLoader 생성
    train_dataset = ClickDataset(tr_df, feature_cols, seq_col, target_col, has_target=True)
    val_dataset = ClickDataset(va_df, feature_cols, seq_col, target_col, has_target=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_train)
    
    # 3. 모델 초기화
    d_features = len(feature_cols)
    model = DCNModel(
        d_features=d_features,
        lstm_hidden=64,
        cross_layers=3,
        deep_hidden=[512, 256, 128],
        dropout=0.3
    ).to(device)
    
    # 4. 옵티마이저와 스케줄러
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )
    
    best_val_score = float("-inf")
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 5
    
    fold_results = []
    
    # 5. 훈련 루프
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        train_pbar = tqdm(train_loader, desc=f"Fold {fold_num} Train Epoch {epoch}")
        for xs, seqs, seq_lens, ys in train_pbar:
            xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
            
            optimizer.zero_grad()
            logits = model(xs, seqs, seq_lens)
            
            try:
                loss = combined_loss(ys.float(), logits.float(), alpha=alpha, margin=margin)
            except Exception as e:
                print(f"Combined loss failed: {e}, using BCE")
                loss = F.binary_cross_entropy_with_logits(logits, ys.float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(ys)
            
            with torch.no_grad():
                train_preds.extend(torch.sigmoid(logits).cpu().numpy())
                train_targets.extend(ys.cpu().numpy())
            
            train_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for xs, seqs, seq_lens, ys in tqdm(val_loader, desc=f"Fold {fold_num} Val Epoch {epoch}"):
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
        
        print(f"[Fold {fold_num} Epoch {epoch}]")
        print(f"  Train - Loss: {train_loss:.4f}, AP: {train_metrics['AP']:.4f}, WLL: {train_metrics['WLL']:.4f}, Final: {train_metrics['Final_Score']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, AP: {val_metrics['AP']:.4f}, WLL: {val_metrics['WLL']:.4f}, Final: {val_metrics['Final_Score']:.4f}")
        
        # 베스트 모델 저장
        if val_metrics['Final_Score'] > best_val_score:
            best_val_score = val_metrics['Final_Score']
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ★ New best score for fold {fold_num}: {best_val_score:.4f}")
        else:
            patience_counter += 1
        
        # epoch별 결과 저장
        fold_results.append({
            'fold': fold_num,
            'epoch': epoch,
            'train_final': train_metrics['Final_Score'],
            'val_final': val_metrics['Final_Score'],
            'val_ap': val_metrics['AP'],
            'val_wll': val_metrics['WLL']
        })
        
        # 조기 종료
        if patience_counter >= early_stopping_patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # 베스트 모델 저장
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be provided when saving model checkpoints")

    fold_model_path = os.path.join(checkpoint_dir, f'best_dcn_model_fold_{fold_num}.pth')
    if best_model_state is None:
        raise RuntimeError("best_model_state is None; ensure at least one validation score was recorded")
    torch.save(best_model_state, fold_model_path)
    
    return {
        'fold': fold_num,
        'best_score': best_val_score,
        'model_path': fold_model_path,
        'fold_results': fold_results
    }

def train_dcn_kfold(train_df, feature_cols, seq_col, target_col, 
                   n_folds=5, batch_size=512, epochs=10, lr=1e-3, device="cuda", 
                   alpha=0.7, margin=1.0, random_state=42, checkpoint_dir=None, log_dir=None):
    """
    K-Fold Cross Validation으로 DCN 모델 훈련
    
    Args:
        n_folds: fold 개수 (불균형 데이터에서는 3-5 추천)
        기타 파라미터는 기존과 동일
    """
    
    print(f"🚀 Starting {n_folds}-Fold Cross Validation Training")
    print(f"📊 Dataset: {len(train_df)} samples")
    print(f"📈 Overall click rate: {train_df[target_col].mean():.4f}")
    print(f"⚖️  Combined Loss - Alpha: {alpha}, Margin: {margin}")
    print("-" * 80)
    
    if checkpoint_dir is None:
        checkpoint_dir = './'
    os.makedirs(checkpoint_dir, exist_ok=True)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

    # StratifiedKFold로 불균형 데이터 처리
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_results = []
    all_fold_metrics = []
    
    # K-Fold 훈련
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[target_col]), 1):
        
        fold_result = train_single_fold(
            fold_num=fold_num,
            train_idx=train_idx,
            val_idx=val_idx,
            train_df=train_df,
            feature_cols=feature_cols,
            seq_col=seq_col,
            target_col=target_col,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            device=device,
            alpha=alpha,
            margin=margin,
            checkpoint_dir=checkpoint_dir
        )
        
        fold_results.append(fold_result)
        all_fold_metrics.extend(fold_result['fold_results'])
    
    # 결과 요약
    print(f"\n{'='*60}")
    print(f"🎯 K-FOLD CROSS VALIDATION RESULTS")
    print(f"{'='*60}")
    
    fold_scores = [result['best_score'] for result in fold_results]
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f"📊 Fold-wise Best Scores:")
    for i, result in enumerate(fold_results):
        print(f"   Fold {result['fold']}: {result['best_score']:.4f}")
    
    print(f"\n📈 Overall Performance:")
    print(f"   Mean CV Score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"   Min Score: {min(fold_scores):.4f}")
    print(f"   Max Score: {max(fold_scores):.4f}")
    
    # 베스트 모델 선택
    best_fold_idx = np.argmax(fold_scores)
    best_fold_result = fold_results[best_fold_idx]
    
    print(f"\n🏆 Best Model: Fold {best_fold_result['fold']} (Score: {best_fold_result['best_score']:.4f})")
    print(f"📁 Best Model Path: {best_fold_result['model_path']}")
    
    # 결과를 DataFrame으로 저장
    results_df = pd.DataFrame(all_fold_metrics)
    results_csv_path = os.path.join(log_dir, 'kfold_training_results.csv') if log_dir else 'kfold_training_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"📄 Detailed results saved to: {results_csv_path}")

    consolidated_model_path = None
    if checkpoint_dir is not None:
        consolidated_model_path = os.path.join(checkpoint_dir, 'best_dcn_model.pth')
        shutil.copy2(best_fold_result['model_path'], consolidated_model_path)

    return {
        'fold_results': fold_results,
        'mean_score': mean_score,
        'std_score': std_score,
        'best_fold': best_fold_result,
        'all_metrics': results_df,
        'best_model_path': consolidated_model_path or best_fold_result['model_path']
    }

def load_best_kfold_model(feature_cols, best_fold_path, device="cuda"):
    """베스트 K-Fold 모델 로드"""
    d_features = len(feature_cols)
    model = DCNModel(
        d_features=d_features,
        lstm_hidden=64,
        cross_layers=3,
        deep_hidden=[512, 256, 128],
        dropout=0.3
    ).to(device)
    
    model.load_state_dict(torch.load(best_fold_path, map_location=device))
    return model
