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

from model import DCNModel, DCNModelEnhanced, DCNModelV3
from data_loader import ClickDataset, collate_fn_train
from evaluate import calculate_metrics


try:
    import wandb
except ImportError:
    wandb = None


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

def train_single_fold(fold_num, train_idx, val_idx, train_df, numeric_cols, seq_col, target_col,
                     batch_size=512, epochs=10, lr=1e-3, device="cuda", alpha=0.7, margin=1.0,
                     checkpoint_dir=None, categorical_info=None, wandb_run=None, wandb_log_every=1,
                     wandb_viz_every=5, confusion_threshold=0.5, global_epoch_offset=0):
    """단일 fold 훈련 함수"""

    categorical_info = categorical_info or {
        'columns': [],
        'maps': {},
        'cardinalities': [],
        'embedding_dims': [],
        'unique_counts': [],
    }

    use_wandb = wandb_run is not None and wandb is not None
    class_names = ['no_click', 'click']

    print(f"\n{'='*20} FOLD {fold_num} {'='*20}")

    tr_df = train_df.iloc[train_idx].reset_index(drop=True)
    va_df = train_df.iloc[val_idx].reset_index(drop=True)

    train_click_rate = tr_df[target_col].mean()
    val_click_rate = va_df[target_col].mean()
    print(f"Train: {len(tr_df)} samples, Click rate: {train_click_rate:.4f}")
    print(f"Val: {len(va_df)} samples, Click rate: {val_click_rate:.4f}")

    if abs(train_click_rate - val_click_rate) > 0.005:
        print(f"⚠️  Warning: Click rate difference: {abs(train_click_rate - val_click_rate):.4f}")

    train_dataset = ClickDataset(
        tr_df,
        numeric_cols,
        seq_col,
        target_col,
        categorical_info=categorical_info,
        has_target=True,
    )
    val_dataset = ClickDataset(
        va_df,
        numeric_cols,
        seq_col,
        target_col,
        categorical_info=categorical_info,
        has_target=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_train,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_train,
    )

    num_numeric_features = len(numeric_cols)
    cat_cardinalities = categorical_info.get('cardinalities', [])
    embedding_dims = categorical_info.get('embedding_dims', [])

    # 1. 기본 DCN
    #model = DCNModel(
    #    num_numeric_features=num_numeric_features,
    #    categorical_cardinalities=cat_cardinalities,
    #    embedding_dims=embedding_dims,
    #    lstm_hidden=64,
    #    cross_layers=4, #6 #3
    #    deep_hidden=[512, 256, 128], # [512, 256, 128], [512, 256, 128, 64]
    #    dropout=0.3,
    #    embedding_dropout=0.05,
    #).to(device)
    
    # 2. 향상된 Cross Network 사용
    # model = DCNModelEnhanced(
    #     num_numeric_features=num_numeric_features,
    #     categorical_cardinalities=cat_cardinalities,
    #     embedding_dims=embedding_dims,
    #     cross_layers=3,
    #     deep_hidden=[512, 256, 128, 64],
    #     use_enhanced_cross=True  # 향상된 Cross Network
    # ).to(device)
    
    # 3. Multi-head Cross Network 사용
    # model = DCNModelEnhanced(
    #     num_numeric_features=num_numeric_features,
    #     categorical_cardinalities=cat_cardinalities,
    #     embedding_dims=embedding_dims,
    #     cross_layers=4,
    #     deep_hidden=[512, 256, 128, 64],
    #     use_multi_head=True,     # Multi-head 사용
    #     num_cross_heads=3
    # )
    
    # 4. DCN v3 (default)
    model = DCNModelV3(
        num_numeric_features=num_numeric_features,
        categorical_cardinalities=cat_cardinalities,
        embedding_dims=embedding_dims,
        lstm_hidden=64,
        cross_layers=4,
        deep_hidden=[768, 512, 256, 128],
        dropout=0.25,
        embedding_dropout=0.05,
        cross_num_experts=4,
        cross_low_rank=64,#16,#32 
        cross_gating_hidden=128,
        cross_dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    best_val_score = float("-inf")
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 5

    fold_results = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        train_pbar = tqdm(train_loader, desc=f"Fold {fold_num} Train Epoch {epoch}")

        for x_num, x_cat, seqs, seq_lens, ys in train_pbar:
            x_num = x_num.to(device)
            seqs = seqs.to(device)
            seq_lens = seq_lens.to(device)
            ys = ys.to(device)
            x_cat = x_cat.to(device) if x_cat is not None else None

            optimizer.zero_grad()
            logits = model(x_num, x_cat, seqs, seq_lens)

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

        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x_num, x_cat, seqs, seq_lens, ys in tqdm(
                val_loader, desc=f"Fold {fold_num} Val Epoch {epoch}"
            ):
                x_num = x_num.to(device)
                seqs = seqs.to(device)
                seq_lens = seq_lens.to(device)
                ys = ys.to(device)
                x_cat = x_cat.to(device) if x_cat is not None else None

                logits = model(x_num, x_cat, seqs, seq_lens)

                try:
                    loss = combined_loss(ys.float(), logits.float(), alpha=alpha, margin=margin)
                except Exception:
                    loss = F.binary_cross_entropy_with_logits(logits, ys.float())

                val_loss += loss.item() * len(ys)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(ys.cpu().numpy())

        val_loss /= len(val_dataset)

        train_targets_arr = np.array(train_targets)
        train_preds_arr = np.array(train_preds)
        val_targets_arr = np.array(val_targets)
        val_preds_arr = np.array(val_preds)

        train_metrics = calculate_metrics(train_targets_arr, train_preds_arr)
        val_metrics = calculate_metrics(val_targets_arr, val_preds_arr)

        scheduler.step(val_metrics['Final_Score'])

        global_epoch_step = global_epoch_offset + epoch

        if use_wandb and (epoch % wandb_log_every == 0):
            wandb_run.log({
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'val_ap': float(val_metrics['AP']),
                'val_wll': float(val_metrics['WLL']),
                'val_final': float(val_metrics['Final_Score']),
                'val_auc': float(val_metrics['AUC']) if not np.isnan(val_metrics['AUC']) else float('nan'),
            }, step=global_epoch_step)

        if use_wandb and wandb_viz_every and (epoch % wandb_viz_every == 0):
            try:
                threshold = confusion_threshold
                preds_binary = (val_preds_arr >= threshold).astype(int)
                cm_plot = wandb.plot.confusion_matrix(
                    y_true=val_targets_arr.astype(int).tolist(),
                    preds=preds_binary.tolist(),
                    class_names=class_names,
                )

                # wandb.plot.pr_curve expects class-probability columns per label
                probas_2d = np.stack(
                    [1.0 - val_preds_arr, val_preds_arr],
                    axis=1,
                ).astype(float)
                pr_curve = wandb.plot.pr_curve(
                    y_true=val_targets_arr.astype(int).tolist(),
                    y_probas=probas_2d.tolist(),
                    labels=class_names,
                )
                wandb_run.log({
                    'val_confusion_matrix': cm_plot,
                    'val_pr_curve': pr_curve,
                }, step=global_epoch_step)
            except Exception as exc:
                print(f"  ⚠️  W&B visualization logging failed at epoch {epoch}: {exc}")

        print(f"[Fold {fold_num} Epoch {epoch}]")
        print(
            f"  Train - Loss: {train_loss:.4f}, AP: {train_metrics['AP']:.4f}, "
            f"WLL: {train_metrics['WLL']:.4f}, Final: {train_metrics['Final_Score']:.4f}"
        )
        print(
            f"  Val   - Loss: {val_loss:.4f}, AP: {val_metrics['AP']:.4f}, "
            f"WLL: {val_metrics['WLL']:.4f}, Final: {val_metrics['Final_Score']:.4f}"
        )

        if val_metrics['Final_Score'] > best_val_score:
            best_val_score = val_metrics['Final_Score']
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ★ New best score for fold {fold_num}: {best_val_score:.4f}")
        else:
            patience_counter += 1

        fold_results.append({
            'fold': fold_num,
            'epoch': epoch,
            'train_final': train_metrics['Final_Score'],
            'val_final': val_metrics['Final_Score'],
            'val_ap': val_metrics['AP'],
            'val_wll': val_metrics['WLL'],
            'val_auc': val_metrics['AUC'],
            'train_loss': train_loss,
            'val_loss': val_loss,
        })

        if patience_counter >= early_stopping_patience:
            print(f"  Early stopping at epoch {epoch}")
            break

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

def train_dcn_kfold(train_df, numeric_cols, categorical_info, seq_col, target_col, 
                   n_folds=5, batch_size=512, epochs=10, lr=1e-3, device="cuda", 
                   alpha=0.7, margin=1.0, random_state=42, checkpoint_dir=None, log_dir=None,
                   wandb_run=None, use_wandb=False, wandb_project=None, wandb_run_name=None,
                   wandb_config=None, wandb_log_every=1, wandb_viz_every=5,
                   confusion_threshold=0.5):
    """
    K-Fold Cross Validation으로 DCN 모델 훈련
    
    Args:
        n_folds: fold 개수 (불균형 데이터에서는 3-5 추천)
        기타 파라미터는 기존과 동일
    """
    
    if (use_wandb or wandb_run is not None) and wandb is None:
        raise ImportError('wandb is required for logging but is not installed.')

    run_config = {
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': lr,
        'alpha': alpha,
        'margin': margin,
        'n_folds': n_folds,
    }
    if wandb_config:
        run_config.update(wandb_config)

    managed_wandb_run = False
    if wandb_run is None and use_wandb:
        wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, config=run_config)
        managed_wandb_run = True
    elif wandb_run is not None and wandb is not None:
        wandb_run.config.update(run_config, allow_val_change=True)

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
            numeric_cols=numeric_cols,
            seq_col=seq_col,
            target_col=target_col,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            device=device,
            alpha=alpha,
            margin=margin,
            checkpoint_dir=checkpoint_dir,
            categorical_info=categorical_info,
            wandb_run=wandb_run,
            wandb_log_every=wandb_log_every,
            wandb_viz_every=wandb_viz_every,
            confusion_threshold=confusion_threshold,
            global_epoch_offset=(fold_num - 1) * epochs,
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

    if wandb_run is not None:
        wandb_run.summary['cv_mean_final'] = float(mean_score)
        wandb_run.summary['cv_std_final'] = float(std_score)
        wandb_run.summary['best_fold'] = int(best_fold_result['fold'])
        wandb_run.summary['best_fold_score'] = float(best_fold_result['best_score'])

    # 결과를 DataFrame으로 저장
    results_df = pd.DataFrame(all_fold_metrics)
    results_csv_path = os.path.join(log_dir, 'kfold_training_results.csv') if log_dir else 'kfold_training_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"📄 Detailed results saved to: {results_csv_path}")

    consolidated_model_path = None
    if checkpoint_dir is not None:
        consolidated_model_path = os.path.join(checkpoint_dir, 'best_dcn_model.pth')
        shutil.copy2(best_fold_result['model_path'], consolidated_model_path)

    output = {
        'fold_results': fold_results,
        'mean_score': mean_score,
        'std_score': std_score,
        'best_fold': best_fold_result,
        'all_metrics': results_df,
        'best_model_path': consolidated_model_path or best_fold_result['model_path']
    }

    if wandb_run is not None:
        wandb_run.summary['best_model_path'] = output['best_model_path']

    if managed_wandb_run and wandb_run is not None:
        wandb_run.finish()

    return output


def load_best_kfold_model(numeric_cols, categorical_info, best_fold_path, device="cuda"):
    """베스트 K-Fold 모델 로드"""
    categorical_info = categorical_info or {
        'columns': [],
        'maps': {},
        'cardinalities': [],
        'embedding_dims': [],
        'unique_counts': [],
    }
    state = torch.load(best_fold_path, map_location=device)
    state_dict = state.get('state_dict') if isinstance(state, dict) and 'state_dict' in state else state

    cross_low_rank = 32
    cross_num_experts = 4
    cross_layers = 4

    sample_key = 'cross_net.layers.0.U'
    if sample_key in state_dict:
        tensor = state_dict[sample_key]
        cross_num_experts = tensor.shape[0]
        cross_low_rank = tensor.shape[-1]

        prefix = 'cross_net.layers.'
        cross_layers = len(
            {
                key.split('.')[2]
                for key in state_dict
                if key.startswith(prefix) and key.endswith('.U')
            }
        )

    model = DCNModelV3(
        num_numeric_features=len(numeric_cols),
        categorical_cardinalities=categorical_info.get('cardinalities', []),
        embedding_dims=categorical_info.get('embedding_dims', []),
        lstm_hidden=64,
        cross_layers=cross_layers,
        deep_hidden=[768, 512, 256, 128],
        dropout=0.25,
        embedding_dropout=0.05,
        cross_num_experts=cross_num_experts,
        cross_low_rank=cross_low_rank,
        cross_gating_hidden=128,
        cross_dropout=0.1,
    ).to(device)

    model.load_state_dict(state_dict)
    return model
