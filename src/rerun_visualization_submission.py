#!/usr/bin/env python3
"""
학습 완료 후 visualization만 다시 실행하는 스크립트
이미 훈련된 모델과 예측 결과를 사용합니다.
"""
import json
import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = Path(__file__).resolve().parent

from config import CFG, device
from data_loader import load_data, get_feature_columns, ClickDataset
from inference import load_models, predict, create_submission
from visualization import EnsembleVisualizer

def analyze_low_importance_features(
    global_series: pd.Series,
    *,
    bottom_percentile: float = 10.0,
    min_threshold: float = None,
    min_features_to_keep: int = 50,
    safe_mode: bool = True,
) -> dict:
    """
    보수적으로 낮은 중요도 변수들을 식별합니다.
    
    Args:
        global_series: 글로벌 feature importance Series
        bottom_percentile: 제거할 하위 퍼센트 (5.0-15.0 권장)
        min_threshold: 절대적 임계값 (이 값 이하는 제거 후보)
        min_features_to_keep: 최소 유지할 변수 개수
        safe_mode: 안전 모드 (극단적 제거 방지)
        
    Returns:
        dict: 분석 결과 및 제거 후보 변수들
    """
    import pandas as pd
    import numpy as np
    
    if global_series.empty:
        return {'candidates': [], 'summary': 'No features to analyze'}
    
    total_features = len(global_series)
    
    # 안전 모드에서 extreme 설정 방지
    if safe_mode:
        bottom_percentile = min(bottom_percentile, 15.0)  # 최대 15%까지만
        min_features_to_keep = max(min_features_to_keep, total_features // 2)
    
    # Percentile 기반 임계값 계산
    percentile_threshold = np.percentile(global_series.values, bottom_percentile)
    
    # 최종 임계값 결정
    if min_threshold is not None:
        final_threshold = max(percentile_threshold, min_threshold)
    else:
        final_threshold = percentile_threshold
    
    # 제거 후보 식별
    candidates = global_series[global_series <= final_threshold].sort_values()
    
    # 안전장치: 너무 많이 제거되는 것 방지
    max_to_remove = total_features - min_features_to_keep
    if len(candidates) > max_to_remove:
        candidates = candidates.head(max_to_remove)
    
    # 통계 계산
    kept_features = total_features - len(candidates)
    
    return {
        'candidates': candidates.index.tolist(),
        'candidates_series': candidates,
        'total_features': total_features,
        'candidates_count': len(candidates),
        'kept_features': kept_features,
        'removal_percentage': len(candidates) / total_features * 100,
        'threshold_used': final_threshold,
        'safe_to_remove': len(candidates) <= max_to_remove and len(candidates) < total_features * 0.15
    }

def main():
    print("🎯 Visualization 재실행 시작...")
    
    # 1. 기본 데이터 로딩 (feature 정보만 필요)
    print("1. 데이터 및 feature 정보 로딩...")
    train_df, test_df = load_data(CFG['DATA_PATH'])
    numeric_cols, categorical_info, seq_col, target_col = get_feature_columns(train_df)
    
    # 2. 학습된 모델들 로드
    print("2. 학습된 모델들 로딩...")
    
    # 모델 경로들을 자동으로 찾기
    model_dir = Path(CFG['CHECKPOINT_DIR'])
    fold_model_paths = []
    
    # best_dcn_model_fold_*.pth 파일들 찾기
    for i in range(1, CFG['FOLDS'] + 1):
        model_path = model_dir / f"best_dcn_model_fold_{i}.pth"
        if model_path.exists():
            fold_model_paths.append(str(model_path))
            print(f"   ✓ Fold {i} 모델 발견: {model_path}")
        else:
            print(f"   ⚠️  Fold {i} 모델 없음: {model_path}")
    
    if not fold_model_paths:
        print("❌ 학습된 모델을 찾을 수 없습니다!")
        return
    
    # 모델들을 메모리에 로드
    ensemble_models = load_models(
        model_paths=fold_model_paths,
        numeric_cols=numeric_cols,
        categorical_info=categorical_info,
        device=device,
    )
    print(f"   ✓ {len(ensemble_models)}개 모델 로드 완료")
    
    # 3. 테스트 예측 수행 (visualization에 필요한 데이터 생성)
    print("3. 테스트 예측 수행...")
    
    # Calibration 설정
    calibration_temperature = None
    calibration_path_str = CFG.get('CALIBRATION_PATH')
    if calibration_path_str:
        calibration_path = (BASE_DIR / calibration_path_str).resolve()
        if calibration_path.exists():
            try:
                with calibration_path.open('r', encoding='utf-8') as f:
                    calibration_payload = json.load(f)
                calibration_temperature = float(calibration_payload.get('temperature'))
                print(f"   ✓ Temperature calibration: T={calibration_temperature:.4f}")
            except Exception as e:
                print(f"   ⚠️  Calibration 파일 오류: {e}")
    
    # 예측 실행
    test_pd = test_df.to_pandas()
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
    print(f"   ✓ 예측 완료: {ensemble_preds.shape}")
    
    # 4. Visualization 실행
    print("4. Visualization 생성...")
    
    diagnostics_dir = Path(CFG['OUTPUT_PATH']) / "diagnostics"
    visualizer = EnsembleVisualizer(diagnostics_dir)
    
    # 4-1. 확률 분포 그래프
    probability_plot_path = visualizer.plot_probability_distribution(ensemble_preds)
    print(f"   ✓ 확률 분포: {probability_plot_path}")
    
    # 4-2. Fold 비교 그래프
    fold_labels = [f"Fold {i+1}" for i in range(len(fold_model_paths))]
    fold_comparison_path, inspected_sample_index = visualizer.plot_fold_comparison(
        fold_predictions=fold_preds,
        ensemble_predictions=ensemble_preds,
        fold_labels=fold_labels,
    )
    print(f"   ✓ Fold 비교 (sample {inspected_sample_index}): {fold_comparison_path}")
    
    # 4-3. Feature contribution 분석 (주요 수정 부분)
    print("   📊 Feature contribution 분석 시작...")
    
    # 훈련 데이터셋 생성
    train_dataset = ClickDataset(
        train_df,
        numeric_cols,
        seq_col,
        target_col=target_col,
        categorical_info=categorical_info,
        has_target=True,
    )
    
    # 가장 좋은 모델 선택 (첫 번째 모델 사용)
    explanation_model = ensemble_models[0]
    
    # 양성 샘플 중에서 설명할 샘플 선택
    positive_indices = (
        train_dataset.df.index[train_dataset.df[target_col] == 1].tolist()
        if target_col in train_dataset.df.columns
        else []
    )
    local_explain_index = int(positive_indices[0]) if positive_indices else 0
    
    # Feature contribution 분석 실행
    try:
        feature_contrib_outputs = visualizer.plot_feature_contributions(
            model=explanation_model,
            dataset=train_dataset,
            numeric_cols=numeric_cols,
            device=device,
            sample_index=local_explain_index,
            global_sample_size=min(len(train_dataset), 256),
            filename_prefix="feature_contributions",
        )
        
        if feature_contrib_outputs:
            print("   ✓ Feature contribution 분석 완료!")
            
            # 결과 출력
            local_prob = feature_contrib_outputs.get('probability')
            prob_display = f"{local_prob:.3f}" if isinstance(local_prob, (int, float)) else 'n/a'
            
            print(f"   📍 로컬 분석 (sample {local_explain_index}, p={prob_display}):")
            local_series = feature_contrib_outputs['local_series'].head(5)
            for feature, value in local_series.items():
                print(f"       • {feature}: {value:+.4f}")
            
            print("   🌍 글로벌 중요 features:")
            global_series = feature_contrib_outputs['global_series'].head(5)
            for feature, value in global_series.items():
                print(f"       • {feature}: {value:.4f}")
            
            # 6. 낮은 중요도 변수 식별 분석
            print("\n6. 낮은 중요도 변수 식별...")
            try:
                full_global_series = feature_contrib_outputs['global_series']
                
                # 보수적 접근: 5%, 10% 두 가지 시나리오 분석
                scenarios = [
                    {'percentile': 5.0, 'name': '매우 보수적'},
                    {'percentile': 10.0, 'name': '보수적'},
                ]
                
                for scenario in scenarios:
                    analysis = analyze_low_importance_features(
                        full_global_series, 
                        bottom_percentile=scenario['percentile']
                    )
                    
                    print(f"\n   🔍 {scenario['name']} 시나리오 (하위 {scenario['percentile']}%):")
                    print(f"       • 전체 변수: {analysis['total_features']}개")
                    print(f"       • 제거 후보: {analysis['candidates_count']}개 ({analysis['removal_percentage']:.1f}%)")
                    print(f"       • 유지 변수: {analysis['kept_features']}개")
                    print(f"       • 임계값: {analysis['threshold_used']:.6f}")
                    
                    if analysis['candidates_count'] > 0:
                        candidates_series = analysis['candidates_series']
                        print(f"       • 제거 후보 중요도 범위: {candidates_series.min():.6f} ~ {candidates_series.max():.6f}")
                        
                        if analysis['candidates_count'] >= 5:
                            print("       • 제거 후보 변수들:")
                            for var_name, importance in candidates_series.items():
                                print(f"           - {var_name}: {importance:.6f}")
                        else:
                            print(f"       • 가장 낮은 5개 변수:")
                            for var_name, importance in candidates_series.head(5).items():
                                print(f"           - {var_name}: {importance:.6f}")
                    
                    # 안전성 평가
                    safety_status = "✅ 안전" if analysis['safe_to_remove'] else "⚠️ 주의"
                    print(f"       • 제거 안전성: {safety_status}")
                
                
            except Exception as e:
                print(f"   ❌ 낮은 중요도 변수 분석 실패: {e}")
                
        else:
            print("   ⚠️ Feature contribution 결과 없음")
            
    except Exception as e:
        print(f"   ❌ Feature contribution 분석 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 제출 파일 생성
    print("5. 제출 파일 생성...")
    
    os.makedirs(CFG['OUTPUT_PATH'], exist_ok=True)
    submission_filename = "dcn_submission0925.csv"
    submission_path = os.path.join(CFG['OUTPUT_PATH'], submission_filename)
    
    try:
        submission = create_submission(
            test_preds=ensemble_preds,
            sample_submission_path='../data/sample_submission.csv',
            output_path=submission_path
        )
        
        print(f"   ✓ 제출 파일 저장: {submission_path}")
        print(f"   ✓ 예측 결과 shape: {ensemble_preds.shape}")
        print(
            f"   ✓ 예측값 범위: ["
            f"{ensemble_preds.min():.4f}, {ensemble_preds.max():.4f}]"
        )
        
    except Exception as e:
        print(f"   ❌ 제출 파일 생성 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()
    
    print("\n🎉 전체 재실행 완료!")
    print("=" * 50)
    print("📂 생성된 파일들:")
    print(f"   • 제출 파일: {submission_path}")
    print(f"   • 진단 그래프들: {diagnostics_dir}/")
    print("=" * 50)

if __name__ == "__main__":
    main()