"""
OCR-derived text prior (M_text).

Turns OCR text-detection boxes into a binary text-region mask that marks WHERE
text is (not where tampering is). The mask is fused into the DTD decoder as a
weak structural prior (see src.fusion).

Detector-agnostic: pass any callable that maps an image to a list of boxes.
This lets the OCR backbone be PRETRAINED (per the TA's feedback), so the limited
tampering data is not a constraint for the OCR part. Adapters for common
pretrained detectors (EasyOCR, PaddleOCR / DBNet) are described in
docs/INTEGRATION.md.

`dilate` slightly grows the text mask. Because a missed tampered region is
unrecoverable downstream, a small dilation trades a little precision for higher
recall (the region-proposal "high recall" target the TA referenced). Tune it
together with src.ocr_eval.
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np

try:  # cv2 is present on Colab; guard so the module imports on the dev machine
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def boxes_to_mask(
    boxes: Sequence[Sequence[float]], height: int, width: int
) -> np.ndarray:
    """Axis-aligned boxes [x1, y1, x2, y2] -> binary uint8 mask (H, W)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(width, int(round(x2)))
        y2 = min(height, int(round(y2)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def polygons_to_mask(
    polygons: Sequence[Sequence[Sequence[float]]], height: int, width: int
) -> np.ndarray:
    """Quadrilateral / polygon text boxes -> binary uint8 mask (H, W).

    Most modern OCR detectors (DBNet, PaddleOCR) return quadrilaterals, which
    fit rotated text more tightly than axis-aligned boxes.
    """
    if cv2 is None:
        raise ImportError("polygons_to_mask requires opencv (available on Colab).")
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


class OCRTextMasker:
    """Wraps a pretrained OCR detector into an M_text producer.

    Parameters
    ----------
    detector : Callable[[np.ndarray], list]
        Returns a list of boxes for an image. Each box is either
        [x1, y1, x2, y2] (axis-aligned) or a polygon [[x, y], ...]
        (set ``polygon=True``).
    polygon : bool
        Whether ``detector`` returns polygons rather than axis-aligned boxes.
    dilate : int
        Optional square-kernel dilation (pixels) to grow the mask for recall.
    """

    def __init__(
        self,
        detector: Callable[[np.ndarray], List],
        polygon: bool = False,
        dilate: int = 0,
    ) -> None:
        self.detector = detector
        self.polygon = polygon
        self.dilate = dilate

    def __call__(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        boxes = self.detector(image)
        if self.polygon:
            mask = polygons_to_mask(boxes, height, width)
        else:
            mask = boxes_to_mask(boxes, height, width)
        if self.dilate > 0:
            if cv2 is None:
                raise ImportError("dilate>0 requires opencv (available on Colab).")
            kernel = np.ones((self.dilate, self.dilate), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask
