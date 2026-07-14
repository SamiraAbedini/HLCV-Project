"""
JSON cache for OCR detections and masks.

The cache key includes backend, language, confidence threshold, Tesseract PSM/OEM
and dilation, so EasyOCR and Tesseract results cannot accidentally reuse each
other's files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from .ocr_backends import OCRConfig, OCRDetection, detections_to_mask


def safe_cache_name(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


class OCRDetectionCache:
    def __init__(self, root: str | Path, config: OCRConfig) -> None:
        self.config = config
        self.root = Path(root) / f"{config.backend}_{config.cache_key()}"
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, sample_id: str) -> Path:
        return self.root / f"{safe_cache_name(sample_id)}.json"

    def load(self, sample_id: str) -> list[OCRDetection] | None:
        path = self.path_for(sample_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("cache_key") != self.config.cache_key():
            return None
        return [OCRDetection.from_json(item) for item in payload.get("detections", [])]

    def save(self, sample_id: str, detections: Sequence[OCRDetection]) -> None:
        payload = {
            "sample_id": sample_id,
            "backend": self.config.backend,
            "cache_key": self.config.cache_key(),
            "config": self.config.__dict__,
            "detections": [det.to_json() for det in detections],
        }
        with self.path_for(sample_id).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    def get_or_compute(self, sample_id: str, image: np.ndarray, backend) -> tuple[list[OCRDetection], bool]:
        detections = self.load(sample_id)
        if detections is not None:
            return detections, True
        detections = backend.detect(image)
        self.save(sample_id, detections)
        return detections, False

    def mask_from_detections(self, detections: Sequence[OCRDetection], image_shape) -> np.ndarray:
        h, w = image_shape[:2]
        return detections_to_mask(detections, h, w, self.config.dilation)
