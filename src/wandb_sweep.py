#!/usr/bin/env python3
"""Weights & Biases sweep utilities for DCN CTR training."""
from __future__ import annotations

import argparse
from typing import Optional

import wandb
import os
import sys

from config import CFG, device
from data_loader import get_feature_columns, load_data
from train import train_dcn_kfold

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

SWEEP_CONFIG = {
    'method': 'grid',
    'metric': {'name': 'val_final', 'goal': 'maximize'},
    'parameters': {
        # 현재 0.7이 잘 작동하므로 주변 값들로 범위 설정
        'alpha': {'values': [0.6, 0.8]},
        
        # margin 1.0이 baseline이므로 확장 탐색
        'margin': {'values': [0.8, 1.2]},
        
        # 현재 0.008 기준으로 위아래 범위 설정 (너무 낮으면 학습 속도 저하)
        'learning_rate': {'values': [0.005, 0.01]},
        
        # 현재 1024가 잘 작동하므로 유지 
        'batch_size': {'values': [1024]},
    },
    'program': 'src/wandb_sweep.py' 
}

CORRECTED_SWEEP_CONFIG = {
    'method': 'grid',
    'metric': {'name': 'val_final', 'goal': 'maximize'},
    'parameters': {
        # 원래 좋았던 0.008을 중심으로 범위 확장
        'learning_rate': {'values': [0.005, 0.008, 0.01, 0.012]},
        
        # batch_size와 learning_rate의 조합 고려
        'batch_size': {'values': [512, 1024, 2048]},
        
        # margin은 중요하다고 확인되었으므로 유지
        'margin': {'values': [1.0, 1.5, 2.0]},
        
        'alpha': {'values': [0.7, 0.8]}
    },
    'program': 'src/wandb_sweep.py' 
}

def sweep_train(config: Optional[dict] = None) -> None:
    """Entry point for wandb agent runs."""
    with wandb.init(config=config) as run:
        cfg = wandb.config

        train_df, _ = load_data(CFG['DATA_PATH'])
        numeric_cols, categorical_info, seq_col, target_col = get_feature_columns(train_df)

        train_dcn_kfold(
            train_df=train_df,
            numeric_cols=numeric_cols,
            categorical_info=categorical_info,
            seq_col=seq_col,
            target_col=target_col,
            n_folds=2,
            batch_size=cfg.batch_size,
            epochs=2,
            lr=cfg.learning_rate,
            device=device,
            alpha=cfg.alpha,
            margin=cfg.margin,
            random_state=CFG['SEED'],
            checkpoint_dir=CFG['CHECKPOINT_DIR'],
            log_dir=CFG['LOG_DIR'],
            wandb_run=run,
            wandb_log_every=CFG.get('WANDB_LOG_EVERY', 1),
            wandb_viz_every=CFG.get('WANDB_VIZ_EVERY', 5),
            confusion_threshold=CFG.get('WANDB_THRESHOLD', 0.5),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or register a W&B sweep for the DCN trainer.")
    parser.add_argument(
        '--create',
        action='store_true',
        help='Create the sweep on the W&B backend and print its ID',
    )
    args, _ = parser.parse_known_args()

    if args.create:
        #sweep_id = wandb.sweep(SWEEP_CONFIG, project=CFG.get('WANDB_PROJECT'))
        sweep_id = wandb.sweep(CORRECTED_SWEEP_CONFIG, project=CFG.get('WANDB_PROJECT'))
        print(f"Created sweep: {sweep_id}")
        return

    sweep_train()


if __name__ == '__main__':
    main()
