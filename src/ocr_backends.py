"""
Optional OCR backends for text-prior masks.

EasyOCR remains supported exactly as an external pretrained detector. Tesseract
is added as a separate backend through pytesseract's bounding-box API. Both
adapters return the same lightweight ``OCRDetection`` objects, which can be
converted into the binary text mask expected by the DTD fusion code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from typing import Iterable, List, Literal, Sequence

import numpy as np

from .text_prior import polygons_to_mask


BackendName = Literal["none", "easyocr", "tesseract"]


@dataclass(frozen=True)
class OCRConfig:
    backend: BackendName = "easyocr"
    languages: tuple[str, ...] = ("en",)
    confidence_threshold: float = 0.0
    dilation: int = 0
    easyocr_gpu: bool = True
    easyocr_paragraph: bool = False
    tesseract_cmd: str | None = None
    tesseract_psm: int = 6
    tesseract_oem: int = 3

    def cache_key(self) -> str:
        payload = asdict(self)
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:12]


@dataclass
class OCRDetection:
    points: list[list[float]]
    confidence: float | None = None
    text: str = ""

    def to_json(self) -> dict:
        return {"points": self.points, "confidence": self.confidence, "text": self.text}

    @classmethod
    def from_json(cls, payload: dict) -> "OCRDetection":
        return cls(
            points=[[float(x), float(y)] for x, y in payload["points"]],
            confidence=None if payload.get("confidence") is None else float(payload["confidence"]),
            text=str(payload.get("text", "")),
        )


def _clip_detection(det: OCRDetection, height: int, width: int) -> OCRDetection:
    pts = []
    for x, y in det.points:
        pts.append([float(min(max(x, 0.0), width - 1)), float(min(max(y, 0.0), height - 1))])
    return OCRDetection(points=pts, confidence=det.confidence, text=det.text)


def detections_to_mask(
    detections: Sequence[OCRDetection],
    height: int,
    width: int,
    dilation: int = 0,
) -> np.ndarray:
    """Convert OCR detections to a binary uint8 text mask."""
    if not detections:
        return np.zeros((height, width), dtype=np.uint8)
    clipped = [_clip_detection(det, height, width) for det in detections]
    mask = polygons_to_mask([det.points for det in clipped], height, width)
    if dilation > 0:
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("dilation requires opencv-python-headless") from exc
        kernel = np.ones((int(dilation), int(dilation)), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(np.uint8)


class NoOCRBackend:
    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig(backend="none")

    def detect(self, image: np.ndarray) -> list[OCRDetection]:
        return []

    def mask(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)


class EasyOCRBackend:
    def __init__(self, config: OCRConfig) -> None:
        try:
            import easyocr  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("EasyOCR backend requires `pip install easyocr`.") from exc
        self.config = config
        self.reader = easyocr.Reader(list(config.languages), gpu=config.easyocr_gpu)

    def detect(self, image: np.ndarray) -> list[OCRDetection]:
        results = self.reader.readtext(
            image,
            detail=1,
            paragraph=self.config.easyocr_paragraph,
        )
        detections: list[OCRDetection] = []
        for points, text, conf in results:
            confidence = float(conf)
            if confidence < self.config.confidence_threshold:
                continue
            detections.append(
                OCRDetection(
                    points=[[float(x), float(y)] for x, y in points],
                    confidence=confidence,
                    text=str(text),
                )
            )
        return detections

    def mask(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return detections_to_mask(self.detect(image), h, w, self.config.dilation)


class TesseractOCRBackend:
    def __init__(self, config: OCRConfig) -> None:
        try:
            import pytesseract  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Tesseract backend requires `pip install pytesseract`.") from exc
        self.config = config
        self.pytesseract = pytesseract
        cmd = config.tesseract_cmd or os.environ.get("TESSERACT_CMD")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

    def detect(self, image: np.ndarray) -> list[OCRDetection]:
        output = self.pytesseract.Output.DICT
        lang = "+".join(self.config.languages)
        tess_config = f"--oem {int(self.config.tesseract_oem)} --psm {int(self.config.tesseract_psm)}"
        data = self.pytesseract.image_to_data(
            image,
            lang=lang,
            config=tess_config,
            output_type=output,
        )
        detections: list[OCRDetection] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = str(data["text"][i] or "").strip()
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if not text or conf < self.config.confidence_threshold:
                continue
            x = float(data["left"][i])
            y = float(data["top"][i])
            w = float(data["width"][i])
            h = float(data["height"][i])
            if w <= 0 or h <= 0:
                continue
            detections.append(
                OCRDetection(
                    points=[[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                    confidence=conf,
                    text=text,
                )
            )
        return detections

    def mask(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return detections_to_mask(self.detect(image), h, w, self.config.dilation)


def create_ocr_backend(config: OCRConfig):
    """Create an OCR backend from a config."""
    if config.backend == "none":
        return NoOCRBackend(config)
    if config.backend == "easyocr":
        return EasyOCRBackend(config)
    if config.backend == "tesseract":
        return TesseractOCRBackend(config)
    raise ValueError(f"Unsupported OCR backend: {config.backend}")
