import numpy as np
from sklearn.metrics import average_precision_score, log_loss

def calculate_metrics(y_true, y_pred_proba):
    """
    대회 평가지표 계산
    AP (Average Precision): 예측 확률에 대한 평균 정밀도
    WLL (Weighted LogLoss): 클래스 불균형을 고려한 가중 로그 손실
    """
    
    # 1. Average Precision 계산
    ap_score = average_precision_score(y_true, y_pred_proba)
    
    # 2. Weighted LogLoss 계산 (50:50 비율로 가중치 조정)
    # 클래스별 샘플 수 계산
    unique_classes = np.unique(y_true)
    class_counts = np.bincount(y_true.astype(int))
    
    # 50:50 비율로 가중치 계산
    total_samples = len(y_true)
    weight_0 = 0.5 / (class_counts[0] / total_samples)  # class 0에 대한 가중치
    weight_1 = 0.5 / (class_counts[1] / total_samples)  # class 1에 대한 가중치
    
    # 각 샘플에 대한 가중치 배열 생성
    sample_weights = np.where(y_true == 0, weight_0, weight_1)
    
    # 가중 로그 손실 계산
    wll_score = log_loss(y_true, y_pred_proba, sample_weight=sample_weights)
    
    # 최종 점수: AP와 WLL의 평균 (대회 규칙에 따라)
    final_score = (ap_score + (1 - wll_score)) / 2
    
    return {
        'AP': ap_score,
        'WLL': wll_score, 
        'Final_Score': final_score
    }
