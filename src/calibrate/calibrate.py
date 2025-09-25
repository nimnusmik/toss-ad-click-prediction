#!/usr/bin/env python3
"""Run post-training temperature scaling and report calibration metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from calibration import TemperatureCalibrationResult, TemperatureScaler
from config import CFG, device
from data_loader import ClickDataset, collate_fn_train, get_feature_columns, load_data
from evaluate import calculate_metrics
from train import load_best_kfold_model


def _compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    class_counts = np.bincount(labels.astype(int), minlength=2).astype(np.float64)
    total = float(labels.shape[0])

    # Guard against degenerate splits
    class_counts = np.clip(class_counts, 1.0, None)

    weight_0 = 0.5 / (class_counts[0] / total)
    weight_1 = 0.5 / (class_counts[1] / total)
    return np.where(labels == 0, weight_0, weight_1).astype(np.float32)


def _select_holdout(
    train_df,
    holdout_fraction: float,
    random_state: int,
    max_samples: int | None,
):
    indices = np.arange(len(train_df))
    _, holdout_idx = train_test_split(
        indices,
        test_size=holdout_fraction,
        stratify=train_df["clicked"],
        random_state=random_state,
    )

    if max_samples is not None and len(holdout_idx) > max_samples:
        rng = np.random.default_rng(random_state)
        holdout_idx = rng.choice(holdout_idx, size=max_samples, replace=False)

    holdout_df = train_df.iloc[holdout_idx].reset_index(drop=True)
    return holdout_df


def _collect_logits(
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    logits_list = []
    targets_list = []

    model.eval()
    with torch.no_grad():
        for x_num, x_cat, seqs, seq_lens, ys in dataloader:
            x_num = x_num.to(device)
            seqs = seqs.to(device)
            seq_lens = seq_lens.to(device)
            ys = ys.to(device)
            x_cat = x_cat.to(device) if x_cat is not None else None

            logits = model(x_num, x_cat, seqs, seq_lens)

            logits_list.append(logits.cpu())
            targets_list.append(ys.cpu())

    logits_tensor = torch.cat(logits_list)
    targets_tensor = torch.cat(targets_list)

    return logits_tensor.numpy(), targets_tensor.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate DCN logits with temperature scaling")
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.1,
        help="Fraction of training data for calibration holdout",
    )
    parser.add_argument(
        "--max-holdout",
        type=int,
        default=250000,
        help="Cap the number of holdout samples for calibration",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Batch size for forward passes during calibration"
    )
    parser.add_argument(
        "--random-state", type=int, default=CFG.get("SEED", 42), help="Random seed for holdout split"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(CFG.get("CHECKPOINT_DIR", "../models/"))
        / "temperature_calibration.json",
        help="Path to store calibration summary",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    def _resolve_path(path_str: str) -> Path:
        return (base_dir / path_str).resolve()

    print("[1/5] Loading data and model metadata...")
    data_path = _resolve_path(CFG["DATA_PATH"])
    data_path_str = str(data_path)
    if not data_path_str.endswith("/"):
        data_path_str += "/"
    train_df, _ = load_data(data_path_str)
    numeric_cols, categorical_info, seq_col, target_col = get_feature_columns(train_df)

    print("[2/5] Preparing calibration holdout slice...")
    holdout_df = _select_holdout(
        train_df,
        holdout_fraction=args.holdout_fraction,
        random_state=args.random_state,
        max_samples=args.max_holdout,
    )
    print(f"   Holdout size: {len(holdout_df):,} rows")

    holdout_dataset = ClickDataset(
        holdout_df,
        numeric_cols,
        seq_col,
        target_col=target_col,
        categorical_info=categorical_info,
        has_target=True,
    )

    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_train,
    )

    print("[3/5] Loading best trained model...")
    model_path = _resolve_path(CFG["MODEL_PATH"])
    model = load_best_kfold_model(
        numeric_cols=numeric_cols,
        categorical_info=categorical_info,
        best_fold_path=str(model_path),
        device=device,
    )

    print("[4/5] Collecting logits and metrics before calibration...")
    logits, targets = _collect_logits(model, holdout_loader)
    probs_before = torch.sigmoid(torch.from_numpy(logits.astype(np.float32))).numpy()
    metrics_before = calculate_metrics(targets, probs_before)
    print(
        f"   Pre-calibration — AP: {metrics_before['AP']:.4f}, "
        f"WLL: {metrics_before['WLL']:.4f}, Final: {metrics_before['Final_Score']:.4f}"
    )

    print("[5/5] Fitting temperature scaler...")
    sample_weights = _compute_sample_weights(targets)
    scaler = TemperatureScaler()
    temperature, loss, iterations = scaler.fit(
        logits=logits,
        labels=targets,
        sample_weights=sample_weights,
    )
    scaled_logits = scaler.transform(logits)
    probs_after = torch.sigmoid(torch.from_numpy(scaled_logits.astype(np.float32))).numpy()
    metrics_after = calculate_metrics(targets, probs_after)

    print(
        f"   Post-calibration — AP: {metrics_after['AP']:.4f}, "
        f"WLL: {metrics_after['WLL']:.4f}, Final: {metrics_after['Final_Score']:.4f}"
    )
    print(f"   Optimised temperature: {temperature:.4f} (loss={loss:.6f}, iterations={iterations})")

    result = TemperatureCalibrationResult(
        temperature=temperature,
        loss=loss,
        iterations=iterations,
        holdout_size=len(holdout_df),
        wll_before=float(metrics_before["WLL"]),
        wll_after=float(metrics_after["WLL"]),
        ap_before=float(metrics_before["AP"]),
        ap_after=float(metrics_after["AP"]),
    )

    output_path = (
        args.output if args.output.is_absolute() else (base_dir / args.output)
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.to_json(), encoding="utf-8")
    print(f"Calibration summary saved to {output_path}")


if __name__ == "__main__":
    main()
