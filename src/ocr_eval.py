"""
OCR module evaluation -- the metric the TA emphasized.

The OCR text prior is effectively a REGION-PROPOSAL stage. If it MISSES text
that is actually tampered, the localizer can never recover that region. So the
headline metric is the RECALL of the OCR text mask with respect to the tampered
regions:

  * pixel_recall_coverage   = |tampered AND text| / |tampered|
  * component_recall        = fraction of connected tampered components whose
                              area is at least `coverage_thresh` covered by the
                              text mask
  * text_area_ratio         = |text| / |all pixels|  (search-space size kept by
                              the prior; smaller is better at equal recall --
                              this is the precision/efficiency side of the
                              region-proposal trade-off)

Ground truth here is the DocTamper tampering mask. We deliberately do NOT assume
GT text-detection boxes, because the dataset does not provide them; we measure
how well the OCR prior covers the *forged* regions, which is what matters.

Use this both quantitatively (aggregate over a split) and qualitatively (dump
masks where component_recall fails) for the report the TA asked for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _connected_components(binary: np.ndarray) -> List[np.ndarray]:
    """Return a boolean mask per connected foreground component."""
    if not binary.any():
        return []
    if cv2 is not None:
        num, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        return [labels == i for i in range(1, num)]
    # Fallback (no cv2 on the dev machine): treat all foreground as one region.
    return [binary.astype(bool)]


@dataclass
class OCRCoverageMeter:
    """Accumulate OCR-vs-tamper coverage statistics over a dataset split.

    Call ``update`` once per image with the OCR text mask and the GT tamper
    mask, then ``compute`` for the aggregated metrics.
    """

    coverage_thresh: float = 0.5  # component counts as covered if >= this fraction inside text

    def __post_init__(self) -> None:
        self._tamper_pixels = 0
        self._covered_tamper_pixels = 0
        self._text_pixels = 0
        self._total_pixels = 0
        self._components = 0
        self._covered_components = 0

    def update(self, text_mask: np.ndarray, tamper_mask: np.ndarray) -> None:
        text = np.asarray(text_mask) > 0
        tamper = np.asarray(tamper_mask) > 0
        if text.shape != tamper.shape:
            raise ValueError(
                f"shape mismatch: text {text.shape} vs tamper {tamper.shape}"
            )
        self._tamper_pixels += int(tamper.sum())
        self._covered_tamper_pixels += int((tamper & text).sum())
        self._text_pixels += int(text.sum())
        self._total_pixels += int(tamper.size)
        for comp in _connected_components(tamper):
            area = int(comp.sum())
            if area == 0:
                continue
            covered = int((comp & text).sum()) / area
            self._components += 1
            if covered >= self.coverage_thresh:
                self._covered_components += 1

    def compute(self) -> Dict[str, float]:
        eps = 1e-9
        return {
            "pixel_recall_coverage": self._covered_tamper_pixels / (self._tamper_pixels + eps),
            "component_recall": self._covered_components / (self._components + eps),
            "text_area_ratio": self._text_pixels / (self._total_pixels + eps),
            "num_tampered_components": float(self._components),
            "num_covered_components": float(self._covered_components),
        }
