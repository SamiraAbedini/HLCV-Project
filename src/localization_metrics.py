"""
Consistent tamper-localization metrics for experiment comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class BinaryLocalizationMeter:
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.probs: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []

    def update(self, probability: np.ndarray, target: np.ndarray) -> None:
        prob = np.asarray(probability, dtype=np.float32)
        gt = np.asarray(target) > 0
        pred = prob >= self.threshold
        self.tp += int((pred & gt).sum())
        self.fp += int((pred & ~gt).sum())
        self.fn += int((~pred & gt).sum())
        self.probs.append(prob.reshape(-1))
        self.labels.append(gt.reshape(-1).astype(np.uint8))

    def compute(self) -> Dict[str, float]:
        eps = 1e-9
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        auc = float("nan")
        if self.probs:
            y = np.concatenate(self.labels)
            s = np.concatenate(self.probs)
            if y.min() != y.max():
                try:
                    from sklearn.metrics import roc_auc_score

                    auc = float(roc_auc_score(y, s))
                except ImportError:
                    auc = float("nan")
        return {
            "pixel_f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "foreground_iou": float(iou),
            "auc": auc,
        }
