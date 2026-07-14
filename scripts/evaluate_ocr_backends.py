#!/usr/bin/env python
"""Evaluate OCR-prior coverage for one or more DocTamper manifests."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.doctamper_lmdb import load_manifest, open_lmdb, read_lmdb_image_and_mask
from src.ocr_backends import OCRConfig, create_ocr_backend
from src.ocr_cache import OCRDetectionCache
from src.ocr_eval import OCRCoverageMeter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["none", "easyocr", "tesseract"], default="tesseract")
    parser.add_argument("--languages", default="en", help="EasyOCR comma list or Tesseract plus/comma list, e.g. en or eng+deu.")
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--dilation", type=int, default=0)
    parser.add_argument("--tesseract-cmd")
    parser.add_argument("--tesseract-psm", type=int, default=6)
    parser.add_argument("--tesseract-oem", type=int, default=3)
    parser.add_argument("--easyocr-cpu", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--qualitative", type=int, default=6)
    return parser.parse_args()


def _languages(raw: str) -> tuple[str, ...]:
    return tuple(part for part in raw.replace("+", ",").split(",") if part)


def _save_visual(out_path: Path, image, text_mask, tamper):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    missed = tamper & ~(text_mask > 0)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[1].imshow(text_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("OCR mask")
    axes[2].imshow(tamper, cmap="gray")
    axes[2].set_title("GT tamper")
    axes[3].imshow(image)
    axes[3].imshow(missed, cmap="Reds", alpha=0.55)
    axes[3].set_title("missed tamper")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    samples = list(manifest["samples"])
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    config = OCRConfig(
        backend=args.backend,
        languages=_languages(args.languages),
        confidence_threshold=args.confidence_threshold,
        dilation=args.dilation,
        easyocr_gpu=not args.easyocr_cpu,
        tesseract_cmd=args.tesseract_cmd,
        tesseract_psm=args.tesseract_psm,
        tesseract_oem=args.tesseract_oem,
    )
    backend = create_ocr_backend(config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = OCRDetectionCache(out_dir / "ocr_cache", config)
    env = open_lmdb(Path(args.data_root) / manifest["source_split"])
    meter = OCRCoverageMeter(coverage_thresh=0.5)
    rows = []
    total_runtime = 0.0

    try:
        for i, sample in enumerate(samples):
            image, tamper = read_lmdb_image_and_mask(env, int(sample["index"]))
            start = time.perf_counter()
            detections, cache_hit = cache.get_or_compute(sample["sample_id"], image, backend)
            text_mask = cache.mask_from_detections(detections, image.shape)
            elapsed = 0.0 if cache_hit else time.perf_counter() - start
            total_runtime += elapsed
            if tamper.any():
                meter.update(text_mask, tamper)
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "num_boxes": len(detections),
                    "runtime_sec": elapsed,
                    "cache_hit": cache_hit,
                    "text_area_ratio": float((text_mask > 0).mean()),
                    "tampered_pixels": int(tamper.sum()),
                }
            )
            if i < args.qualitative:
                _save_visual(out_dir / f"qual_{i:03d}_{args.backend}.png", image, text_mask, tamper)
    finally:
        env.close()

    metrics = meter.compute()
    metrics.update(
        {
            "backend": args.backend,
            "config": config.__dict__,
            "manifest": str(args.manifest),
            "num_images": len(samples),
            "runtime_per_image_sec": total_runtime / max(1, len(samples)),
            "mean_num_boxes": float(np.mean([r["num_boxes"] for r in rows])) if rows else 0.0,
        }
    )
    with (out_dir / f"ocr_metrics_{args.backend}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    with (out_dir / f"ocr_per_image_{args.backend}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
