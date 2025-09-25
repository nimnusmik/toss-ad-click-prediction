"""Calibration utilities for post-training probability adjustment."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemperatureCalibrationResult:
    """Container for fitted temperature-scaling statistics."""

    temperature: float
    loss: float
    iterations: int
    holdout_size: int
    wll_before: float
    wll_after: float
    ap_before: float
    ap_after: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


class TemperatureScaler(nn.Module):
    """Applies temperature scaling to logits and optimises the temperature."""

    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        if init_temperature <= 0:
            raise ValueError("init_temperature must be positive")
        self.log_temperature = nn.Parameter(torch.log(torch.tensor([init_temperature], dtype=torch.float32)))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return logits / self.temperature

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        max_iter: int = 50,
        tolerance: float = 1e-6,
        ) -> Tuple[float, float, int]:
        """Optimise the temperature to minimise weighted BCE on logits.

        Returns:
            temperature: Optimised temperature value.
            loss: Final weighted BCE achieved during optimisation.
            iterations: LBFGS iteration count reported by the optimiser.
        """
        if logits.ndim != 1:
            raise ValueError("logits must be 1-D array")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("logits and labels must have the same length")

        logits_t = torch.from_numpy(logits.astype(np.float32))
        labels_t = torch.from_numpy(labels.astype(np.float32))

        if sample_weights is not None:
            weights_t = torch.from_numpy(sample_weights.astype(np.float32))
        else:
            weights_t = None

        optimizer = torch.optim.LBFGS(
            [self.log_temperature], lr=0.01, max_iter=max_iter, tolerance_grad=tolerance
        )

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled_logits = self.forward(logits_t)
            if weights_t is not None:
                loss = F.binary_cross_entropy_with_logits(scaled_logits, labels_t, weight=weights_t)
            else:
                loss = F.binary_cross_entropy_with_logits(scaled_logits, labels_t)
            loss.backward()
            return loss

        optimizer.step(_closure)

        with torch.no_grad():
            final_logits = self.forward(logits_t)
            if weights_t is not None:
                final_loss = F.binary_cross_entropy_with_logits(
                    final_logits, labels_t, weight=weights_t
                ).item()
            else:
                final_loss = F.binary_cross_entropy_with_logits(final_logits, labels_t).item()

        state = optimizer.state.get(self.log_temperature, {})
        iterations = int(state.get("n_iter", 0))
        return float(self.temperature.item()), float(final_loss), iterations

    def transform(self, logits: np.ndarray) -> np.ndarray:
        logits_t = torch.from_numpy(logits.astype(np.float32))
        scaled = self.forward(logits_t)
        return scaled.detach().numpy()

    def transform_proba(self, probs: np.ndarray) -> np.ndarray:
        logits = np.log(probs / np.clip(1.0 - probs, 1e-12, 1.0))
        scaled_logits = self.transform(logits)
        return 1.0 / (1.0 + np.exp(-scaled_logits))

    def summary(self) -> str:
        return f"TemperatureScaler(temperature={self.temperature.item():.4f})"
