"""Wrap the ruptures PELT algorithm for behavioural change-point detection."""

from __future__ import annotations

import numpy as np
import ruptures as rpt

from .models import DriftMetrics


class ChangePointDetector:
    """Fits PELT to per-session metric time series, returns change-point indices."""

    def __init__(self, penalty: float = 3.0, min_size: int = 2) -> None:
        self.penalty = penalty
        self.min_size = min_size

    def detect(self, metrics: list[DriftMetrics]) -> list[int]:
        """Return 0-based session indices where step changes occur."""
        if len(metrics) < self.min_size * 2:
            return []

        signal = self._to_signal(metrics)
        algo = rpt.Pelt(model="rbf", min_size=self.min_size).fit(signal)
        raw = algo.predict(pen=self.penalty)

        # ruptures returns 1-based indices; last element == len(signal) (terminal)
        result = sorted({cp - 1 for cp in raw if cp < len(metrics)})
        return result

    @staticmethod
    def _to_signal(metrics: list[DriftMetrics]) -> np.ndarray:
        return np.array(
            [
                [m.topic_shift, m.disclosure_depth_delta, m.style_shift, m.steering_ratio]
                for m in metrics
            ]
        )
