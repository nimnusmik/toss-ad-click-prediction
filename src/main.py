#!/usr/bin/env python3
"""
DCN 모델을 사용한 클릭 예측 메인 스크립트
데이터 로딩, 훈련, 추론, 제출 파일 생성까지 전체 파이프라인을 실행합니다.
"""
import json
import os
import sys
from pathlib import Path

import torch

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = Path(__file__).resolve().parent

from config import CFG, device
from data_loader import load_data, get_feature_columns, ClickDataset
from train import train_dcn_kfold
from inference import load_models, predict, create_submission
from visualization import EnsembleVisualizer


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
    alpha=0.5,  #0.7->0.6->0.7->0.5
    margin=1.5, #1.0->1.5->1.0->1.5
    random_state=42,

    checkpoint_dir=CFG['CHECKPOINT_DIR'],
    log_dir=CFG['LOG_DIR'],
    **wandb_kwargs,
)

best_model_path = kfold_results.get('best_model_path') or kfold_results['best_fold']['model_path']
fold_model_paths = [fold_result['model_path'] for fold_result in kfold_results['fold_results']]
ensemble_models = load_models(
    model_paths=fold_model_paths,
    numeric_cols=numeric_cols,
    categorical_info=categorical_info,
    device=device,
)

# GPU 메모리 정리
torch.cuda.empty_cache()

calibration_temperature = None
calibration_path_str = CFG.get('CALIBRATION_PATH')
if calibration_path_str:
    calibration_path = (BASE_DIR / calibration_path_str).resolve()
    if calibration_path.exists():
        try:
            with calibration_path.open('r', encoding='utf-8') as calibration_file:
                calibration_payload = json.load(calibration_file)
            calibration_temperature = float(calibration_payload.get('temperature'))
            print(
                f"\n   › Applying temperature calibration: "
                f"T={calibration_temperature:.4f} (source: {calibration_path})"
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            print(
                "\n   ⚠️  Calibration file is malformed "
                f"({exc}); proceeding without temperature scaling."
            )
            calibration_temperature = None
    else:
        print(
            f"\n   ℹ️  Calibration file not found at {calibration_path}; "
            "skipping temperature scaling."
        )

# 3. Running inference
print(f"\n3. Running ensemble inference...")

# 테스트 데이터를 pandas로 변환
test_pd = test_df.to_pandas()

# 예측 수행
ensemble_preds, fold_preds = predict(
    model=ensemble_models,
    test_df=test_pd,
    numeric_cols=numeric_cols,
    categorical_info=categorical_info,
    seq_col=seq_col,
    batch_size=CFG['BATCH_SIZE'],
    device=device,
    temperature=calibration_temperature,
)

# 4. Diagnostics & visualization
diagnostics_dir = Path(CFG['OUTPUT_PATH']) / "diagnostics"
visualizer = EnsembleVisualizer(diagnostics_dir)

probability_plot_path = visualizer.plot_probability_distribution(ensemble_preds)
fold_labels = [f"Fold {fold_info['fold']}" for fold_info in kfold_results['fold_results']]
fold_comparison_path, inspected_sample_index = visualizer.plot_fold_comparison(
    fold_predictions=fold_preds,
    ensemble_predictions=ensemble_preds,
    fold_labels=fold_labels,
)

train_dataset = ClickDataset(
    train_df,
    numeric_cols,
    seq_col,
    target_col=target_col,
    categorical_info=categorical_info,
    has_target=True,
)

fold_order = [fold_info['fold'] for fold_info in kfold_results['fold_results']]
best_fold_id = kfold_results['best_fold']['fold']
best_model_index = fold_order.index(best_fold_id) if best_fold_id in fold_order else 0
explanation_model = ensemble_models[best_model_index]

positive_indices = (
    train_dataset.df.index[train_dataset.df[target_col] == 1].tolist()
    if target_col in train_dataset.df.columns
    else []
)
local_explain_index = int(positive_indices[0]) if positive_indices else 0

feature_contrib_outputs = visualizer.plot_feature_contributions(
    model=explanation_model,
    dataset=train_dataset,
    numeric_cols=numeric_cols,
    device=device,
    sample_index=local_explain_index,
    global_sample_size=min(len(train_dataset), 256),
    filename_prefix="feature_contributions",
)

metrics_plot_path, metrics_summary = visualizer.plot_validation_metrics(
    kfold_results.get('all_metrics')
)

diagnostics_summary = {
    'probability_distribution': probability_plot_path,
    'fold_comparison': fold_comparison_path,
    'validation_metrics': metrics_plot_path,
    'feature_local_plot': feature_contrib_outputs.get('local_plot') if feature_contrib_outputs else None,
    'feature_global_plot': feature_contrib_outputs.get('global_plot') if feature_contrib_outputs else None,
    'feature_local_top': feature_contrib_outputs.get('local_series') if feature_contrib_outputs else None,
    'feature_global_scores': feature_contrib_outputs.get('global_series') if feature_contrib_outputs else None,
    'inspected_sample_index': inspected_sample_index,
    'local_explain_index': local_explain_index,
    'local_probability': feature_contrib_outputs.get('probability') if feature_contrib_outputs else None,
}

print("\n4. Diagnostics artifacts saved:")
print(f"   - Probability distribution: {probability_plot_path}")
print(f"   - Fold comparison (sample {inspected_sample_index}): {fold_comparison_path}")
if metrics_plot_path:
    print(f"   - Validation metrics: {metrics_plot_path}")
else:
    print("   - Validation metrics plot skipped (no data)")

if feature_contrib_outputs:
    local_prob = feature_contrib_outputs.get('probability')
    if isinstance(local_prob, (int, float)):
        prob_display = f"{local_prob:.3f}"
    else:
        prob_display = 'n/a'
    print(
        f"   - Feature contributions (local idx {local_explain_index}, "
        f"p={prob_display}):"
    )
    local_series = feature_contrib_outputs['local_series'].head(5)
    for feature, value in local_series.items():
        print(f"       • {feature}: {value:+.4f}")
    global_series = feature_contrib_outputs['global_series'].head(5)
    print("   - Top global numeric features:")
    for feature, value in global_series.items():
        print(f"       • {feature}: {value:.4f}")
else:
    print("   - Feature contribution analysis skipped (no numeric features)")

# 5. 제출 파일 생성
os.makedirs(CFG['OUTPUT_PATH'], exist_ok=True)
submission_filename = "dcn_submission0926.csv"
submission_path = os.path.join(CFG['OUTPUT_PATH'], submission_filename)
submission = create_submission(
    test_preds=ensemble_preds,
    sample_submission_path='../data/sample_submission.csv',
    output_path=submission_path
)

print(f"\n5. Pipeline completed successfully!")
print(f"   - Submission file: {submission_path}")
print(f"   - Ensemble predictions shape: {ensemble_preds.shape}")
print(
    f"   - Prediction range: ["
    f"{ensemble_preds.min():.4f}, {ensemble_preds.max():.4f}]"
)

#return ensemble_models, ensemble_preds, submission
