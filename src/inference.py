import os
from typing import Optional
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import DCNModel, DCNModelEnhanced, DCNModelV3
from data_loader import ClickDataset, collate_fn_infer


def load_model(model_path, numeric_cols, categorical_info, device, model_kwargs=None):
    """저장된 모델 로드"""
    categorical_info = categorical_info or {
        'columns': [],
        'cardinalities': [],
        'embedding_dims': [],
    }

    state = torch.load(model_path, map_location=device)
    state_dict = state.get('state_dict') if isinstance(state, dict) and 'state_dict' in state else state

    defaults = {
        'lstm_hidden': 64,
        'deep_hidden': [768, 512, 256, 128],
        'dropout': 0.25,
        'embedding_dropout': 0.05,
        'cross_layers': 4,
        'cross_num_experts': 4,
        'cross_low_rank': 64,
        'cross_gating_hidden': 128,
        'cross_dropout': 0.1,
    }
    build_kwargs = {**defaults, **(model_kwargs or {})}

    sample_key = 'cross_net.layers.0.U'
    if sample_key in state_dict:
        tensor = state_dict[sample_key]
        build_kwargs['cross_num_experts'] = tensor.shape[0]
        build_kwargs['cross_low_rank'] = tensor.shape[-1]
        prefix = 'cross_net.layers.'
        build_kwargs['cross_layers'] = len(
            {
                key.split('.')[2]
                for key in state_dict
                if key.startswith(prefix) and key.endswith('.U')
            }
        )

    if build_kwargs.get('deep_hidden') is None:
        build_kwargs['deep_hidden'] = defaults['deep_hidden']

    model = DCNModelV3(
        num_numeric_features=len(numeric_cols),
        categorical_cardinalities=categorical_info.get('cardinalities', []),
        embedding_dims=categorical_info.get('embedding_dims', []),
        **build_kwargs,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_models(model_paths, numeric_cols, categorical_info, device, model_kwargs=None):
    """Load multiple fold models for ensemble inference."""
    if not model_paths:
        raise ValueError("model_paths must contain at least one checkpoint path")

    models = []
    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        models.append(load_model(path, numeric_cols, categorical_info, device, model_kwargs=model_kwargs))

    return models

def predict(
        model,
        test_df,
        numeric_cols,
        categorical_info,
        seq_col,
        batch_size,
        device,
        temperature: Optional[float] = None,
    ):
    """테스트 데이터에 대한 예측 수행 (ensemble-aware)."""
    test_dataset = ClickDataset(
        test_df,
        numeric_cols,
        seq_col,
        categorical_info=categorical_info,
        has_target=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_infer,
    )

    models: list[torch.nn.Module]
    if isinstance(model, Sequence):
        models = list(model)
    else:
        models = [model]

    if not models:
        raise ValueError("At least one model instance is required for prediction")

    for mdl in models:
        mdl.eval()

    ensemble_predictions: list[torch.Tensor] = []
    fold_predictions: list[list[torch.Tensor]] = [[] for _ in models]

    with torch.no_grad():
        for x_num, x_cat, seqs, lens in tqdm(test_loader, desc="Inference"):
            x_num = x_num.to(device)
            seqs = seqs.to(device)
            lens = lens.to(device)
            x_cat = x_cat.to(device) if x_cat is not None else None

            fold_logits = []
            for mdl in models:
                logits = mdl(x_num, x_cat, seqs, lens)
                fold_logits.append(logits)

            stacked_logits = torch.stack(fold_logits)
            stacked_probs = torch.sigmoid(stacked_logits)

            for idx, probs in enumerate(stacked_probs):
                fold_predictions[idx].append(probs.cpu())

            mean_logits = stacked_logits.mean(dim=0)
            if temperature is not None:
                mean_logits = mean_logits / temperature
            ensemble_probs = torch.sigmoid(mean_logits).cpu()
            ensemble_predictions.append(ensemble_probs)

    ensemble_array = torch.cat(ensemble_predictions).numpy()
    fold_arrays = [torch.cat(preds).numpy() for preds in fold_predictions]
    fold_matrix = np.vstack(fold_arrays).T if len(fold_arrays) > 1 else fold_arrays[0][:, None]

    return ensemble_array, fold_matrix

def create_submission(test_preds, sample_submission_path, output_path):
    """제출 파일 생성"""
    submit = pd.read_csv(sample_submission_path)
    submit['clicked'] = test_preds
    submit.to_csv(output_path, index=False)
    print(f"Submission file saved to: {output_path}")
    
    return submit
