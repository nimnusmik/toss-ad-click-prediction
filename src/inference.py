import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import DCNModel
from data_loader import ClickDataset, collate_fn_infer


def load_model(model_path, numeric_cols, categorical_info, device):
    """저장된 모델 로드"""
    categorical_info = categorical_info or {
        'columns': [],
        'cardinalities': [],
        'embedding_dims': [],
    }
    model = DCNModel(
        num_numeric_features=len(numeric_cols),
        categorical_cardinalities=categorical_info.get('cardinalities', []), 
        embedding_dims=categorical_info.get('embedding_dims', []),
        lstm_hidden=64,
        cross_layers=3,
        deep_hidden=[512, 256, 128],
        dropout=0.3,
        embedding_dropout=0.05,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def predict(model, test_df, numeric_cols, categorical_info, seq_col, batch_size, device):
    """테스트 데이터에 대한 예측 수행"""
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

    model.eval()
    predictions = []

    with torch.no_grad():
        for x_num, x_cat, seqs, lens in tqdm(test_loader, desc="Inference"):
            x_num = x_num.to(device)
            seqs = seqs.to(device)
            lens = lens.to(device)
            x_cat = x_cat.to(device) if x_cat is not None else None
            preds = torch.sigmoid(model(x_num, x_cat, seqs, lens)).cpu()
            predictions.append(preds)

    test_preds = torch.cat(predictions).numpy()

    return test_preds

def create_submission(test_preds, sample_submission_path, output_path):
    """제출 파일 생성"""
    submit = pd.read_csv(sample_submission_path)
    submit['clicked'] = test_preds
    submit.to_csv(output_path, index=False)
    print(f"Submission file saved to: {output_path}")
    
    return submit
