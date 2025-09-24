#!/usr/bin/env python3
"""
DCN 모델을 사용한 클릭 예측 메인 스크립트
데이터 로딩, 훈련, 추론, 제출 파일 생성까지 전체 파이프라인을 실행합니다.
"""
import os
import sys
import torch

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CFG, device
from data_loader import load_data, get_feature_columns
from train import train_dcn_kfold, load_best_kfold_model
from inference import predict, create_submission


# 1. 데이터 로딩 및 전처리
print("\n1. Loading and preprocessing data...")
train_df, test_df = load_data(CFG['DATA_PATH'])
numeric_cols, categorical_info, seq_col, target_col = get_feature_columns(train_df)

# 2. 모델 훈련
print(f"\n2. Training DCN model...")
print(f"   - Batch size: {CFG['BATCH_SIZE']}")
print(f"   - Epochs: {CFG['EPOCHS']}")
print(f"   - Learning rate: {CFG['LEARNING_RATE']}")
print(f"   - Device: {device}")

wandb_kwargs = {}
if CFG.get('USE_WANDB', False):
    wandb_kwargs = {
        'use_wandb': True,
        'wandb_project': CFG.get('WANDB_PROJECT'),
        'wandb_run_name': CFG.get('WANDB_RUN_NAME'),
        'wandb_log_every': CFG.get('WANDB_LOG_EVERY', 1),
        'wandb_viz_every': CFG.get('WANDB_VIZ_EVERY', 5),
        'confusion_threshold': CFG.get('WANDB_THRESHOLD', 0.5),
        'wandb_config': {
            'model': 'dcn',
            'device': device,
            'dataset_rows': len(train_df),
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_info.get('columns', [])),
        },
    }

kfold_results = train_dcn_kfold(
    train_df=train_df,
    numeric_cols=numeric_cols,
    categorical_info=categorical_info,
    seq_col=seq_col,
    target_col='clicked',

    n_folds=CFG['FOLDS'], 
    batch_size=CFG['BATCH_SIZE'],
    epochs=CFG['EPOCHS'],
    lr= CFG['LEARNING_RATE'],

    device=device,
    alpha=0.7, #0.7->0.6->0.7
    margin=1.0, #1.0 -> 1.5->1.0
    random_state=42,

    checkpoint_dir=CFG['CHECKPOINT_DIR'],
    log_dir=CFG['LOG_DIR'],
    **wandb_kwargs,
)

best_model_path = kfold_results.get('best_model_path') or kfold_results['best_fold']['model_path']
model = load_best_kfold_model(
    numeric_cols=numeric_cols,
    categorical_info=categorical_info,
    best_fold_path=best_model_path,
    device=device
)
# GPU 메모리 정리
torch.cuda.empty_cache()

# 3. 추론 및 제출 파일 생성
print(f"\n3. Making predictions and creating submission...")

# 테스트 데이터를 pandas로 변환
test_pd = test_df.to_pandas()

# 예측 수행
test_preds = predict(
    model=model,
    test_df=test_pd,
    numeric_cols=numeric_cols,
    categorical_info=categorical_info,
    seq_col=seq_col,
    batch_size=CFG['BATCH_SIZE'],
    device=device
)

# 제출 파일 생성
os.makedirs(CFG['OUTPUT_PATH'], exist_ok=True)
submission_filename = "dcn_submission0923.csv"
submission_path = os.path.join(CFG['OUTPUT_PATH'], submission_filename)
submission = create_submission(
    test_preds=test_preds,
    sample_submission_path='../data/sample_submission.csv',
    output_path=submission_path
)

print(f"\n4. Pipeline completed successfully!")
print(f"   - Submission file: {submission_path}")
print(f"   - Predictions shape: {test_preds.shape}")
print(f"   - Prediction range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")

#return model, test_preds, submission
