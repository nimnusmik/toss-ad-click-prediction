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
from train import train_dcn_model
from inference import load_model, predict, create_submission

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("DCN Click Prediction Pipeline")
    print("=" * 80)
    
    # 1. 데이터 로딩 및 전처리
    print("\n1. Loading and preprocessing data...")
    train_df, test_df = load_data(CFG['DATA_PATH'])
    feature_cols, seq_col, target_col = get_feature_columns(train_df)
    
    # 2. 모델 훈련
    print(f"\n2. Training DCN model...")
    print(f"   - Batch size: {CFG['BATCH_SIZE']}")
    print(f"   - Epochs: {CFG['EPOCHS']}")
    print(f"   - Learning rate: {CFG['LEARNING_RATE']}")
    print(f"   - Device: {device}")
    
    model = train_dcn_model(
        train_df=train_df,
        feature_cols=feature_cols,
        seq_col=seq_col,
        target_col=target_col,
        batch_size=CFG['BATCH_SIZE'],
        epochs=CFG['EPOCHS'],
        lr=CFG['LEARNING_RATE'],
        device=device,
        alpha = 0.7, # WLL 중시
        margin = 1.0 # Ranking margin
    )

    # GPU 메모리 정리
    torch.cuda.empty_cache()
    
    # 3. 추론 및 제출 파일 생성
    print(f"\n3. Making predictions and creating submission...")
    
    # 테스트 데이터를 pandas로 변환
    test_pd = test_df.to_pandas()
    
    # 모델 로드 (훈련된 모델 사용)
    model = load_model(CFG['MODEL_PATH'], len(feature_cols), device)
    
    # 예측 수행
    test_preds = predict(
        model=model,
        test_df=test_pd,
        feature_cols=feature_cols,
        seq_col=seq_col,
        batch_size=CFG['BATCH_SIZE'],
        device=device
    )
    
    # 제출 파일 생성
    os.makedirs(CFG['OUTPUT_PATH'], exist_ok=True)
    submission = create_submission(
        test_preds=test_preds,
        sample_submission_path='../data/sample_submission.csv',
        output_path=f"{CFG['OUTPUT_PATH']}dcn_submission2.csv"
    )
    
    print(f"\n4. Pipeline completed successfully!")
    print(f"   - Submission file: {CFG['OUTPUT_PATH']}dcn_submission.csv")
    print(f"   - Predictions shape: {test_preds.shape}")
    print(f"   - Prediction range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    
    return model, test_preds, submission

if __name__ == "__main__":
    model, predictions, submission = main()