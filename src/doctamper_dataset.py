"""
Manifest-aware DocTamper dataset for controlled Colab fine-tuning.

The public DocTamper repository mainly exposes inference/evaluation loaders.
Those loaders expect compression-record pickles for the split being loaded; in
the public tree, records are present for FCD/SCD/TestingSet but not necessarily
for TrainingSet. This class keeps the DTD tensor interface while using a fixed,
reproducible JPEG quality value instead of random compression records.

It is intended for controlled student ablations, not as a reproduction of the
full DTD training curriculum.
"""
from __future__ import annotations

import io
from pathlib import Path
import pickle
import tempfile

import numpy as np

from .doctamper_lmdb import load_manifest, open_lmdb


class ManifestDocTamperDataset:
    """Read samples listed in a manifest and return DTD-compatible tensors."""

    def __init__(
        self,
        data_root: str | Path,
        manifest: str | Path | dict,
        qt_table_path: str | Path,
        jpeg_quality: int = 75,
    ) -> None:
        try:
            import cv2  # type: ignore
            import jpegio  # type: ignore
            import torch
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - Colab/runtime dependency
            raise ImportError(
                "ManifestDocTamperDataset requires torch, pillow, opencv, lmdb, and dwgoon/jpegio."
            ) from exc

        self.cv2 = cv2
        self.jpegio = jpegio
        self.torch = torch
        self.Image = Image
        self.manifest = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
        self.samples = list(self.manifest["samples"])
        self.source_split = str(self.manifest["source_split"])
        self.env = open_lmdb(Path(data_root) / self.source_split)
        self.jpeg_quality = int(jpeg_quality)

        with Path(qt_table_path).open("rb") as f:
            qt_tables = pickle.load(f)
        if self.jpeg_quality not in qt_tables:
            raise KeyError(f"JPEG quality {self.jpeg_quality} missing from {qt_table_path}")
        self.qtb = torch.LongTensor(qt_tables[self.jpeg_quality])
        self.mean = np.asarray([0.485, 0.455, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int):
        sample = self.samples[item]
        image_key = sample["image_key"].encode("utf-8")
        label_key = sample["label_key"].encode("utf-8")
        with self.env.begin(write=False) as txn:
            imgbuf = txn.get(image_key)
            lblbuf = txn.get(label_key)
        if imgbuf is None or lblbuf is None:
            raise KeyError(f"Missing LMDB pair for {sample['sample_id']}")

        image = self.Image.open(io.BytesIO(imgbuf)).convert("L")
        mask = self.cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
        if mask is None:
            raise ValueError(f"Could not decode mask for {sample['sample_id']}")
        mask = (mask != 0).astype(np.int64)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            image.save(tmp.name, "JPEG", quality=self.jpeg_quality)
            jpg = self.jpegio.read(tmp.name)
            dct = jpg.coef_arrays[0].copy()
            image_rgb = self.Image.open(tmp.name).convert("RGB")

        arr = np.asarray(image_rgb).astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        image_tensor = self.torch.from_numpy(arr.transpose(2, 0, 1)).float()
        label_tensor = self.torch.from_numpy(mask).long().unsqueeze(0)

        return {
            "image": image_tensor,
            "label": label_tensor,
            "rgb": np.clip(np.abs(dct), 0, 20).astype(np.int64),
            "q": self.qtb.clone(),
            "sample_id": sample["sample_id"],
            "index": int(sample["index"]),
        }
