import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import DCNModel
from data_loader import ClickDataset, collate_fn_infer

def load_model(model_path, d_features, device):
    """저장된 모델 로드"""
    model = DCNModel(
        d_features=d_features,
        lstm_hidden=64,
        cross_layers=3,
        deep_hidden=[512, 256, 128],
        dropout=0.3
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def predict(model, test_df, feature_cols, seq_col, batch_size, device):
    """테스트 데이터에 대한 예측 수행"""
    # Dataset과 DataLoader 생성
    test_dataset = ClickDataset(test_df, feature_cols, seq_col, has_target=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_infer)
    
    # 예측 수행
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for xs, seqs, lens in tqdm(test_loader, desc="Inference"):
            xs, seqs, lens = xs.to(device), seqs.to(device), lens.to(device)
            preds = torch.sigmoid(model(xs, seqs, lens)).cpu()
            predictions.append(preds)
    
    # 모든 예측값 합치기
    test_preds = torch.cat(predictions).numpy()
    
    return test_preds

def create_submission(test_preds, sample_submission_path, output_path):
    """제출 파일 생성"""
    submit = pd.read_csv(sample_submission_path)
    submit['clicked'] = test_preds
    submit.to_csv(output_path, index=False)
    print(f"Submission file saved to: {output_path}")
    
    return submit
