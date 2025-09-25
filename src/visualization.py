"""Visualization utilities for ensemble diagnostics and feature analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


class EnsembleVisualizer:
    """Helper class for producing diagnostic plots for model predictions."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Probability diagnostics
    # ------------------------------------------------------------------
    def plot_probability_distribution(
        self,
        predictions: Sequence[float] | np.ndarray,
        *,
        bins: int = 50,
        filename: str = "probability_distribution.png",
    ) -> Path:
        preds = np.asarray(predictions, dtype=float).reshape(-1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(preds, bins=bins, color="#2563eb", alpha=0.8, edgecolor="black")
        ax.set_title("Prediction Probability Distribution")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", alpha=0.3)
        return self._save_figure(fig, filename)

    # ------------------------------------------------------------------
    # Fold comparison diagnostics
    # ------------------------------------------------------------------
    def plot_fold_comparison(
        self,
        fold_predictions: np.ndarray,
        ensemble_predictions: Sequence[float] | np.ndarray,
        *,
        sample_index: int | None = None,
        fold_labels: Sequence[str] | None = None,
        filename: str = "fold_prediction_comparison.png",
    ) -> tuple[Path, int]:
        fold_preds = np.asarray(fold_predictions, dtype=float)
        if fold_preds.ndim != 2:
            raise ValueError("fold_predictions must be a 2D array")

        ensemble_preds = np.asarray(ensemble_predictions, dtype=float).reshape(-1)
        if fold_preds.shape[0] != ensemble_preds.shape[0]:
            raise ValueError("Ensemble predictions length must match fold predictions rows")

        num_samples, num_folds = fold_preds.shape
        if sample_index is None:
            variances = fold_preds.var(axis=1)
            sample_index = int(np.argmax(variances))
        if not 0 <= sample_index < num_samples:
            raise IndexError("sample_index out of bounds for provided predictions")

        fold_values = fold_preds[sample_index]
        ensemble_value = float(ensemble_preds[sample_index])

        if fold_labels is None:
            fold_labels = [f"Fold {i + 1}" for i in range(num_folds)]
        else:
            fold_labels = list(fold_labels)
            if len(fold_labels) != num_folds:
                raise ValueError("fold_labels length must match number of folds")

        x = np.arange(num_folds)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x, fold_values, color="#6366f1", alpha=0.85, label="Fold prediction")
        ax.axhline(
            ensemble_value,
            color="#ef4444",
            linestyle="--",
            linewidth=2,
            label=f"Ensemble ({ensemble_value:.3f})",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels, rotation=35, ha="right")
        ax.set_ylabel("Predicted probability")
        ax.set_title(f"Fold vs Ensemble Prediction (sample {sample_index})")
        ax.set_ylim(0, 1)
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.2)

        path = self._save_figure(fig, filename)
        return path, sample_index

    # ------------------------------------------------------------------
    # Validation metric diagnostics
    # ------------------------------------------------------------------
    def plot_validation_metrics(
        self,
        metrics_df: pd.DataFrame | Sequence[dict],
        *,
        filename: str = "fold_validation_metrics.png",
    ) -> tuple[Path, pd.DataFrame] | tuple[None, pd.DataFrame]:
        if metrics_df is None:
            return None, pd.DataFrame()

        df = pd.DataFrame(metrics_df)
        if df.empty or 'fold' not in df.columns:
            return None, df

        df = df.copy()
        df['fold'] = df['fold'].astype(int)
        df.sort_values(['fold', 'val_final'], ascending=[True, False], inplace=True)
        best_per_fold = df.groupby('fold', as_index=False).first()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

        axes[0].bar(best_per_fold['fold'], best_per_fold['val_ap'], color="#0ea5e9")
        axes[0].set_title('Validation AP per Fold')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Average Precision')
        axes[0].axhline(
            best_per_fold['val_ap'].mean(),
            color="#f59e0b",
            linestyle="--",
            label='Fold mean',
        )
        axes[0].legend(loc='best')
        axes[0].grid(True, linestyle='--', alpha=0.2)

        axes[1].bar(best_per_fold['fold'], best_per_fold['val_wll'], color="#22c55e")
        axes[1].set_title('Validation WLL per Fold')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('Weighted Log Loss')
        axes[1].axhline(
            best_per_fold['val_wll'].mean(),
            color="#f59e0b",
            linestyle='--',
            label='Fold mean',
        )
        axes[1].legend(loc='best')
        axes[1].grid(True, linestyle='--', alpha=0.2)

        fig.suptitle('Validation Metrics by Fold (best epoch per fold)')

        path = self._save_figure(fig, filename)
        return path, best_per_fold

    # ------------------------------------------------------------------
    # Feature contribution diagnostics
    # ------------------------------------------------------------------
    def plot_feature_contributions(
        self,
        model: torch.nn.Module,
        dataset,
        numeric_cols: Sequence[str],
        *,
        device: torch.device,
        sample_index: int | None = None,
        global_sample_size: int = 128,
        filename_prefix: str = "feature_contributions",
        random_state: int = 42,
    ) -> dict:
        numeric_cols = list(numeric_cols)
        if not numeric_cols:
            return {}

        model.eval()
        dataset_size = len(dataset)
        if dataset_size == 0:
            return {}

        rng = np.random.default_rng(random_state)
        if sample_index is None or not 0 <= sample_index < dataset_size:
            sample_index = int(rng.integers(low=0, high=dataset_size))

        global_sample_size = min(global_sample_size, dataset_size)
        global_indices = rng.choice(dataset_size, size=global_sample_size, replace=False)

        global_contribs = []
        for idx in global_indices:
            contrib, _ = self._numeric_saliency(model, dataset, idx, device)
            global_contribs.append(np.abs(contrib))
        global_scores = np.mean(global_contribs, axis=0)

        local_contrib, probability = self._numeric_saliency(model, dataset, sample_index, device)

        global_series = pd.Series(global_scores, index=numeric_cols)
        local_series = pd.Series(local_contrib, index=numeric_cols)

        global_plot = self._plot_series(
            global_series,
            title="Global Numeric Feature Importance (|grad * input|)",
            filename=f"{filename_prefix}_global.png",
            allow_negative=False,
        )

        local_title = (
            f"Local Numeric Contributions (sample {sample_index}, "
            f"p={probability:.3f})"
        )
        local_plot = self._plot_series(
            local_series,
            title=local_title,
            filename=f"{filename_prefix}_local.png",
            allow_negative=True,
        )

        return {
            'local_plot': local_plot,
            'global_plot': global_plot,
            'local_series': local_series.sort_values(key=np.abs, ascending=False),
            'global_series': global_series.sort_values(ascending=False),
            'sample_index': sample_index,
            'probability': probability,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _numeric_saliency(
        self,
        model: torch.nn.Module,
        dataset,
        index: int,
        device: torch.device,
    ) -> tuple[np.ndarray, float]:
    # 현재 모델의 상태를 저장
        original_training_mode = model.training
        
        # BatchNorm 모듈들의 원래 상태 저장
        original_bn_modes = {}
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                original_bn_modes[name] = module.training
        
        try:
            # 전체 모델을 training 모드로 전환 (RNN backward를 위해)
            model.train()
            
            # BatchNorm 모듈들만 evaluation 모드로 설정 (단일 배치 문제 해결)
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    module.eval()
            
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                x_num, x_cat, seq, *_ = dataset[index]

                x_num = x_num.unsqueeze(0).to(device)
                x_num = x_num.clone().detach().requires_grad_(True)

                seq_tensor = seq.unsqueeze(0).to(device)
                seq_length = torch.tensor([seq_tensor.shape[1]], dtype=torch.long, device=device)

                if x_cat is not None:
                    x_cat_tensor = x_cat.unsqueeze(0).to(device)
                else:
                    x_cat_tensor = None

                # 이제 RNN은 training 모드, BatchNorm은 eval 모드로 실행
                logits = model(x_num, x_cat_tensor, seq_tensor, seq_length)
                prob = torch.sigmoid(logits)
                
                # backward pass 실행 (RNN은 training 모드이므로 가능)
                prob.backward()

                gradient = x_num.grad.detach().cpu().numpy()[0]
                values = x_num.detach().cpu().numpy()[0]
                contribution = gradient * values

        finally:
            # 모든 모듈의 원래 모드로 복원
            model.train(original_training_mode)
            
            # BatchNorm 모듈들의 원래 상태 복원
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    original_mode = original_bn_modes.get(name, False)
                    module.train(original_mode)
            
        model.zero_grad(set_to_none=True)
        return contribution, float(prob.detach().cpu().item())

    def _plot_series(
        self,
        series: pd.Series,
        *,
        title: str,
        filename: str,
        allow_negative: bool,
    ) -> Path | None:
        if series.empty:
            return None

        sorted_series = series.sort_values(key=np.abs, ascending=True)
        fig_height = max(4, 0.4 * len(sorted_series) + 1)
        fig, ax = plt.subplots(figsize=(8, fig_height))
        colors = (
            ["#22c55e" if v >= 0 else "#ef4444" for v in sorted_series.values]
            if allow_negative
            else "#0ea5e9"
        )
        ax.barh(sorted_series.index, sorted_series.values, color=colors, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Contribution")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.grid(True, axis="x", linestyle="--", alpha=0.2)
        fig.tight_layout()
        return self._save_figure(fig, filename)

    def _save_figure(self, fig: plt.Figure, filename: str) -> Path:
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return path
