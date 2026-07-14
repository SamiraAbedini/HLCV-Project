"""
Small, safe helpers for DocTamper LMDB splits and manifest-based subsets.

The upstream DocTamper dataset stores samples in LMDB using integer-indexed keys:

    image-000000000, label-000000000

The DTD training loader also depends on the same integer index to look up JPEG
compression records. For reduced experiments, the least disruptive approach is
therefore to keep the original LMDB unchanged and save the chosen integer IDs in
human-readable manifests.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import numpy as np


def sample_id(source_split: str, index: int) -> str:
    """Stable ID used in manifests and caches."""
    return f"{source_split}:{int(index):09d}"


def lmdb_keys(index: int) -> tuple[str, str]:
    """Return DocTamper image/label keys for an integer sample index."""
    idx = int(index)
    return f"image-{idx:09d}", f"label-{idx:09d}"


def load_manifest(path: str | Path) -> Mapping:
    """Load a JSON manifest."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def manifest_indices(manifest: Mapping) -> List[int]:
    """Extract integer indices from a manifest dictionary."""
    return [int(s["index"]) for s in manifest.get("samples", [])]


def manifest_sample_ids(manifest: Mapping) -> List[str]:
    """Extract stable sample IDs from a manifest dictionary."""
    return [str(s["sample_id"]) for s in manifest.get("samples", [])]


class ManifestSubset:
    """Map a manifest onto an existing PyTorch-style dataset.

    This wrapper deliberately does not know about DTD tensors, DCT features, or
    quantization tables. It only remaps ``__getitem__`` from compact subset
    positions to original DocTamper indices, so the upstream loader remains the
    source of truth.
    """

    def __init__(self, dataset, manifest: Mapping | str | Path) -> None:
        self.dataset = dataset
        self.manifest = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
        self.indices = manifest_indices(self.manifest)
        self.samples = list(self.manifest.get("samples", []))
        if len(self.indices) != len(self.samples):
            raise ValueError("manifest samples are malformed")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        return self.dataset[self.indices[item]]


def _require_lmdb():
    try:
        import lmdb  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("Reading DocTamper LMDB files requires `pip install lmdb`.") from exc
    return lmdb


def get_lmdb_num_samples(lmdb_dir: str | Path) -> int:
    """Read the ``num-samples`` key from one DocTamper LMDB directory."""
    lmdb = _require_lmdb()
    env = lmdb.open(
        str(lmdb_dir),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    try:
        with env.begin(write=False) as txn:
            value = txn.get(b"num-samples")
            if value is None:
                raise KeyError(f"{lmdb_dir} does not contain LMDB key num-samples")
            return int(value)
    finally:
        env.close()


def open_lmdb(lmdb_dir: str | Path):
    """Open a DocTamper LMDB read-only."""
    lmdb = _require_lmdb()
    return lmdb.open(
        str(lmdb_dir),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=64,
    )


def lmdb_pair_exists(env, index: int) -> bool:
    """Check that both image and label keys exist."""
    image_key, label_key = lmdb_keys(index)
    with env.begin(write=False) as txn:
        return (
            txn.get(image_key.encode("utf-8")) is not None
            and txn.get(label_key.encode("utf-8")) is not None
        )


def read_lmdb_image_and_mask(env, index: int):
    """Return ``(rgb_uint8, tamper_mask_bool)`` for a DocTamper LMDB sample."""
    try:
        import cv2  # type: ignore
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional runtime deps
        raise ImportError("Reading image/mask pairs requires pillow and opencv-python-headless.") from exc

    image_key, label_key = lmdb_keys(index)
    with env.begin(write=False) as txn:
        imgbuf = txn.get(image_key.encode("utf-8"))
        lblbuf = txn.get(label_key.encode("utf-8"))
    if imgbuf is None or lblbuf is None:
        raise KeyError(f"Missing image or label key for index {index}")
    image = np.asarray(Image.open(io.BytesIO(imgbuf)).convert("RGB"))
    mask = cv2.imdecode(np.frombuffer(lblbuf, dtype=np.uint8), 0)
    if mask is None:
        raise ValueError(f"Could not decode label mask for index {index}")
    return image, mask > 0


def validate_manifest_against_lmdb(manifest: Mapping, data_root: str | Path) -> List[str]:
    """Validate keys, dimensions, duplicates, and expected count for one manifest."""
    errors: List[str] = []
    source_split = manifest.get("source_split")
    samples = list(manifest.get("samples", []))
    if not source_split:
        errors.append("manifest is missing source_split")
        return errors

    ids = [str(s.get("sample_id")) for s in samples]
    if len(ids) != len(set(ids)):
        errors.append("manifest contains duplicate sample IDs")
    expected_size = manifest.get("size")
    if expected_size is not None and int(expected_size) != len(samples):
        errors.append(f"manifest size={expected_size} but contains {len(samples)} samples")

    lmdb_dir = Path(data_root) / str(source_split)
    if not (lmdb_dir / "data.mdb").exists():
        errors.append(f"LMDB data.mdb not found: {lmdb_dir}")
        return errors

    env = open_lmdb(lmdb_dir)
    try:
        for sample in samples:
            idx = int(sample["index"])
            sid = sample_id(str(source_split), idx)
            if sample.get("sample_id") != sid:
                errors.append(f"sample index {idx} has non-standard sample_id {sample.get('sample_id')}")
            if not lmdb_pair_exists(env, idx):
                errors.append(f"missing image/label pair for {sid}")
                continue
            try:
                image, mask = read_lmdb_image_and_mask(env, idx)
            except Exception as exc:  # pragma: no cover - reports external data issues
                errors.append(f"failed to read {sid}: {exc}")
                continue
            if image.shape[:2] != mask.shape:
                errors.append(f"image/mask shape mismatch for {sid}: {image.shape[:2]} vs {mask.shape}")
    finally:
        env.close()
    return errors


def assert_no_overlap(manifests: Sequence[Mapping]) -> None:
    """Raise if any stable sample ID appears in more than one manifest."""
    seen: dict[str, str] = {}
    overlaps: list[str] = []
    for manifest in manifests:
        split_name = str(manifest.get("split", "unknown"))
        for sid in manifest_sample_ids(manifest):
            if sid in seen:
                overlaps.append(f"{sid} in {seen[sid]} and {split_name}")
            else:
                seen[sid] = split_name
    if overlaps:
        raise ValueError("Manifest split overlap detected: " + "; ".join(overlaps[:10]))


def write_json(path: str | Path, payload: Mapping) -> None:
    """Write compact, human-readable JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
