#%%
#!/usr/bin/env python3
"""
DCN 모델을 사용한 클릭 예측 메인 스크립트
데이터 로딩, 훈련, 추론, 제출 파일 생성까지 전체 파이프라인을 실행합니다.
"""
import os
import sys
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, log_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG, device
from data_loader import load_data, get_feature_columns, ClickDataset, collate_fn_train
from train import train_dcn_model
from inference import load_model, predict, create_submission


# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("DCN Click Prediction Pipeline")
    print("=" * 80)
    
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
    
    model = train_dcn_model(
        train_df=train_df,
        numeric_cols=numeric_cols,
        categorical_info=categorical_info,
        seq_col=seq_col,
        target_col=target_col,
        batch_size=CFG['BATCH_SIZE'],
        epochs=20,
        lr=CFG['LEARNING_RATE'],
        device=device
    )
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()
    
    # 3. 추론 및 제출 파일 생성
    print(f"\n3. Making predictions and creating submission...")
    
    # 테스트 데이터를 pandas로 변환
    test_pd = test_df.to_pandas()
    
    # 모델 로드 (훈련된 모델 사용)
    model = load_model(CFG['MODEL_PATH'], numeric_cols, categorical_info, device)
    
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
    #submission = create_submission(
    #    test_preds=test_preds,
    #    sample_submission_path='../data/sample_submission.csv',
    #    output_path=f"{CFG['OUTPUT_PATH']}dcn_submission.csv"
    #)
    
    print(f"\n4. Pipeline completed successfully!")
    #print(f"   - Submission file: {CFG['OUTPUT_PATH']}dcn_submission.csv")
    print(f"   - Predictions shape: {test_preds.shape}")
    print(f"   - Prediction range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    
    return model, test_preds, submission, train_df, numeric_cols, categorical_info, seq_col, target_col


    
#%%
model, predictions, submission, train_df, numeric_cols, categorical_info, seq_col, target_col = main()


#%%
# 간단한 모델 분석 실행
def simple_model_analysis(model, train_df, numeric_cols, categorical_info, seq_col, target_col):
    """간단한 모델 해석 분석"""
    print("\n" + "=" * 80)
    print("🔍 SIMPLE MODEL ANALYSIS")
    print("=" * 80)
    
    # 1. 검증 데이터 준비 (간단하게)
    print("📊 검증 데이터 준비...")
    train_pd = train_df.to_pandas() if hasattr(train_df, 'to_pandas') else train_df
    
    # 샘플링으로 빠르게 (전체 데이터 대신 일부만)
    sample_df = train_pd.sample(n=min(5000000, len(train_pd)), random_state=42)
    _, val_df = train_test_split(sample_df, test_size=0.3, random_state=42)
    
    val_dataset = ClickDataset(val_df, numeric_cols, seq_col, target_col, categorical_info=categorical_info, has_target=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, collate_fn=collate_fn_train)
    
    print(f"   분석 데이터: {len(val_df):,}개 샘플")
    
    # 2. 기본 성능 평가
    print("\n🎯 모델 성능 평가...")
    model.eval()
    all_probs, all_targets = [], []
    correct, total = 0, 0
    
    with torch.no_grad():
        for x_num, x_cat, seqs, seq_lens, ys in tqdm(val_loader, desc="성능 평가"):
            x_num = x_num.to(device)
            x_cat = x_cat.to(device) if x_cat is not None else None
            seqs = seqs.to(device)
            seq_lens = seq_lens.to(device)
            ys = ys.to(device)

            logits = model(x_num, x_cat, seqs, seq_lens)
            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).float()
            
            total += ys.size(0)
            correct += (predicted == ys).sum().item()
            
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(ys.cpu().numpy())
    
    accuracy = correct / total
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # 대회 평가지표 계산
    ap_score = average_precision_score(all_targets, all_probs)
    
    # 가중 로그 손실
    class_counts = np.bincount(all_targets.astype(int))
    total_samples = len(all_targets)
    weight_0 = 0.5 / (class_counts[0] / total_samples) if class_counts[0] > 0 else 1.0
    weight_1 = 0.5 / (class_counts[1] / total_samples) if class_counts[1] > 0 else 1.0
    sample_weights = np.where(all_targets, weight_1, weight_0)
    wll_score = log_loss(all_targets, all_probs, sample_weight=sample_weights)
    
    final_score = (ap_score + (1 - wll_score)) / 2
    
    print(f"\n📈 성능 결과:")
    print(f"   정확도: {accuracy:.4f}")
    print(f"   평균 예측 확률: {np.mean(all_probs):.4f}")
    print(f"   실제 클릭률: {np.mean(all_targets):.4f}")
    print(f"\n🏆 대회 평가지표:")
    print(f"   AP (Average Precision): {ap_score:.4f}")
    print(f"   WLL (Weighted LogLoss): {wll_score:.4f}")
    print(f"   Final Score: {final_score:.4f}")
    
    # 3. 간단한 피처 중요도 (빠른 버전)
    print(f"\n🔍 피처 중요도 분석 (Top 10)...")
    
    # 원래 성능
    original_score = log_loss(all_targets, all_probs)
    feature_importance = []
    
    # 상위 30개 피처만 빠르게 테스트
    important_features = numeric_cols[:30]  
    
    for feat_idx in tqdm(range(len(important_features)), desc="피처 중요도"):
        # 작은 샘플로 빠르게 테스트
        test_probs = []
        
        with torch.no_grad():
            for batch_idx, (x_num, x_cat, seqs, seq_lens, ys) in enumerate(val_loader):
                if batch_idx >= 5:  # 5개 배치만
                    break
                x_num = x_num.to(device)
                x_cat = x_cat.to(device) if x_cat is not None else None
                seqs = seqs.to(device)
                seq_lens = seq_lens.to(device)

                # 피처 permutation
                x_perm = x_num.clone()
                perm_indices = torch.randperm(x_perm.size(0))
                x_perm[:, feat_idx] = x_perm[perm_indices, feat_idx]

                logits = model(x_perm, x_cat, seqs, seq_lens)
                probs = torch.sigmoid(logits)
                test_probs.extend(probs.cpu().numpy())
        
        if len(test_probs) > 0:
            test_targets = all_targets[:len(test_probs)]
            permuted_score = log_loss(test_targets, test_probs)
            importance = permuted_score - original_score
        else:
            importance = 0
            
        feature_importance.append((important_features[feat_idx], importance))
    
    # 중요도 순으로 정렬
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Top 10 중요 피처:")
    for i, (feat_name, importance) in enumerate(feature_importance[:10]):
        print(f"   {i+1:2d}. {feat_name:<25}: {importance:.6f}")
    
    # 4. 예측 분포 시각화
    print(f"\n📊 예측 분포 시각화...")
    
    plt.figure(figsize=(20, 5))
    
    # 전체 분포
    plt.subplot(1, 4, 1)
    plt.hist(all_probs, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Prediction Probability Distribution')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Frequency')
    
    # 클래스별 분포
    plt.subplot(1, 4, 2)
    clicked_probs = all_probs[all_targets == 1]
    not_clicked_probs = all_probs[all_targets == 0]
    
    plt.hist(not_clicked_probs, bins=20, alpha=0.7, label='Not Clicked (0)', color='red')
    plt.hist(clicked_probs, bins=20, alpha=0.7, label='Clicked (1)', color='green')
    plt.title('Class-wise Prediction Distribution')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 고신뢰도 예측 vs 실제 클릭 비교
    plt.subplot(1, 4, 3)
    
    # 0.6 이상 예측한 샘플들
    high_pred_mask = all_probs >= 0.6
    high_pred_samples = all_probs[high_pred_mask]
    high_pred_actual = all_targets[high_pred_mask]
    
    # 실제 clicked=1인 샘플들의 예측 확률
    actual_clicked_probs = all_probs[all_targets == 1]
    
    # 히스토그램 비교
    plt.hist(actual_clicked_probs, bins=20, alpha=0.7, label=f'Actual Clicked=1 ({len(actual_clicked_probs)})', 
            color='green', density=True)
    plt.hist(high_pred_samples, bins=20, alpha=0.7, label=f'Predicted ≥0.6 ({len(high_pred_samples)})', 
            color='orange', density=True)
    
    plt.axvline(x=0.6, color='red', linestyle='--', alpha=0.8, label='Threshold 0.6')
    plt.title('High Confidence Predictions vs Actual Clicks')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.legend()
    
    # 성능 요약
    plt.subplot(1, 4, 4)
    
    # 구간별 분석
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    precision_scores = []
    recall_scores = []
    
    for threshold in thresholds:
        pred_mask = all_probs >= threshold
        if np.sum(pred_mask) > 0:
            # Precision: 예측한 것 중 실제 맞은 비율
            precision = np.mean(all_targets[pred_mask])
            # Recall: 실제 클릭 중 찾아낸 비율  
            recall = np.sum(all_targets[pred_mask]) / np.sum(all_targets)
        else:
            precision = 0
            recall = 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    x_pos = np.arange(len(thresholds))
    width = 0.35
    
    bars1 = plt.bar(x_pos - width/2, precision_scores, width, label='Precision', color='lightblue')
    bars2 = plt.bar(x_pos + width/2, recall_scores, width, label='Recall', color='lightcoral')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision vs Recall by Threshold')
    plt.xticks(x_pos, [f'{t:.1f}' for t in thresholds])
    plt.legend()
    plt.ylim(0, 1)
    
    # 값 표시
    for bar, score in zip(bars1, precision_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    for bar, score in zip(bars2, recall_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # 상세 분석 결과 출력
    print(f"\n🎯 고신뢰도 예측 분석 (임계값 0.6):")
    high_pred_count = np.sum(high_pred_mask)
    high_pred_correct = np.sum(high_pred_actual)
    actual_clicked_count = np.sum(all_targets)
    
    if high_pred_count > 0:
        precision_06 = high_pred_correct / high_pred_count
        recall_06 = high_pred_correct / actual_clicked_count
        print(f"   예측 ≥0.6인 샘플: {high_pred_count:,}개")
        print(f"   이 중 실제 클릭: {high_pred_correct:,}개")
        print(f"   정밀도(Precision): {precision_06:.4f}")
        print(f"   재현율(Recall): {recall_06:.4f}")
    else:
        print(f"   예측 ≥0.6인 샘플이 없습니다.")
    
    print(f"\n📊 실제 클릭(1) 샘플들의 예측 분포:")
    print(f"   총 실제 클릭: {actual_clicked_count:,}개")
    print(f"   이 중 0.6+ 예측: {np.sum(actual_clicked_probs >= 0.6):,}개 ({np.mean(actual_clicked_probs >= 0.6):.3f})")
    print(f"   이 중 0.8+ 예측: {np.sum(actual_clicked_probs >= 0.8):,}개 ({np.mean(actual_clicked_probs >= 0.8):.3f})")
    print(f"   평균 예측 확률: {np.mean(actual_clicked_probs):.4f}")
    
    # 5. 최적 임계값 찾기
    from sklearn.metrics import f1_score
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        pred_binary = (all_probs >= threshold).astype(int)
        f1 = f1_score(all_targets, pred_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n🎯 최적화 결과:")
    print(f"   최적 임계값: {best_threshold:.3f}")
    print(f"   최적 F1 점수: {best_f1:.4f}")
    
    print(f"\n✅ 간단 분석 완료!")
    
    return {
        'accuracy': accuracy,
        'ap_score': ap_score,
        'wll_score': wll_score,
        'final_score': final_score,
        'best_threshold': best_threshold,
        'top_features': [f[0] for f in feature_importance[:5]]
    }

print(f"\n🤔 모델이 어떻게 판단하고 있는지 간단히 분석해보겠습니다...")

analysis_results = simple_model_analysis(
    model=model,
    train_df=train_df,
    numeric_cols=numeric_cols,
    categorical_info=categorical_info,
    seq_col=seq_col,
    target_col=target_col
)

if analysis_results:
    print(f"\n🎉 분석 완료!")
    print(f"   📊 Final Score: {analysis_results['final_score']:.4f}")
    print(f"   🎯 최적 임계값: {analysis_results['best_threshold']:.3f}")
    print(f"   🏆 Top 5 중요 피처: {', '.join(analysis_results['top_features'])}")
    
else:
    print(f"\n⚠️ 분석을 완료하지 못했지만 모델 훈련과 제출 파일은 생성되었습니다.")

print(f"\n🏁 전체 파이프라인 완료!")
print(f"   📁 제출 파일: {CFG['OUTPUT_PATH']}dcn_submission.csv")
print(f"   💾 모델 파일: {CFG['MODEL_PATH']}")


# %%
