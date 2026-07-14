"""
OCR-Guided and Boundary-Aware Tampered Text Localization.

Novel components that extend the DTD (DocTamper) baseline with:
  * an OCR-derived text prior (src.text_prior) fused into the decoder
    (src.fusion), and
  * imbalance-aware + boundary-aware supervision (src.losses).

The OCR module is evaluated as a region-proposal stage (src.ocr_eval),
where recall of tampered regions is the headline metric.

These modules are dependency-light and plug into the DocTamper model, which is
cloned separately on Colab. Nothing here downloads, executes, or bundles the
DocTamper checkpoints / pickle files.
"""

from .text_prior import OCRTextMasker, boxes_to_mask, polygons_to_mask
from .ocr_eval import OCRCoverageMeter
from .ocr_backends import OCRConfig, OCRDetection, create_ocr_backend, detections_to_mask
from .doctamper_lmdb import ManifestSubset

try:
    from .doctamper_dataset import ManifestDocTamperDataset
except ImportError:  # pragma: no cover - optional DTD runtime deps
    ManifestDocTamperDataset = None

try:
    from .losses import CombinedTamperLoss, soft_dice_loss, boundary_loss
    from .fusion import TextPriorFusion
except ImportError:  # pragma: no cover - allows data/OCR scripts without torch
    CombinedTamperLoss = None
    soft_dice_loss = None
    boundary_loss = None
    TextPriorFusion = None

__all__ = [
    "CombinedTamperLoss",
    "soft_dice_loss",
    "boundary_loss",
    "TextPriorFusion",
    "OCRTextMasker",
    "boxes_to_mask",
    "polygons_to_mask",
    "OCRCoverageMeter",
    "OCRConfig",
    "OCRDetection",
    "create_ocr_backend",
    "detections_to_mask",
    "ManifestSubset",
    "ManifestDocTamperDataset",
]
