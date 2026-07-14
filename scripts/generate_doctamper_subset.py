#!/usr/bin/env python
"""Generate reproducible DocTamper reduced-dataset manifests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import pickle
from pathlib import Path
import random
import sys
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.doctamper_lmdb import (
    assert_no_overlap,
    get_lmdb_num_samples,
    lmdb_keys,
    sample_id,
    validate_manifest_against_lmdb,
    write_json,
)


EVAL_ONLY_NAMES = {"DocTamperV1-FCD", "DocTamperV1-SCD", "DocTamperV1-TestingSet"}


def _load_tamper_types(path: Path | None, allow_unsafe_pickle: bool) -> dict:
    if path is None:
        return {}
    if not allow_unsafe_pickle:
        raise ValueError(
            "Tampering-type metadata is stored as pickle. Re-run with "
            "--allow-unsafe-pickle only if you trust this file."
        )
    with path.open("rb") as f:
        return pickle.load(f)


def _build_manifest(split: str, source_split: str, indices: list[int], seed: int, metadata: dict) -> dict:
    samples = []
    for idx in indices:
        image_key, label_key = lmdb_keys(idx)
        sid = sample_id(source_split, idx)
        item = {
            "sample_id": sid,
            "index": idx,
            "source_split": source_split,
            "image_key": image_key,
            "label_key": label_key,
        }
        if sid in metadata:
            item["tamper_type"] = metadata[sid]
        elif idx in metadata:
            item["tamper_type"] = metadata[idx]
        elif str(idx) in metadata:
            item["tamper_type"] = metadata[str(idx)]
        samples.append(item)
    return {
        "format": "doctamper_subset_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "source_split": source_split,
        "seed": seed,
        "size": len(samples),
        "samples": samples,
    }


def _shuffled_indices(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Folder containing DocTamperV1-* LMDB directories.")
    parser.add_argument("--output-dir", required=True, help="Where train/val/test manifests will be written.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-source", default="DocTamperV1-TrainingSet")
    parser.add_argument("--test-source", default="DocTamperV1-TestingSet")
    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--allow-eval-source-for-training", action="store_true")
    parser.add_argument("--tamper-types-pickle", type=Path)
    parser.add_argument("--allow-unsafe-pickle", action="store_true")
    parser.add_argument("--skip-file-validation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)

    if args.train_source in EVAL_ONLY_NAMES and not args.allow_eval_source_for_training:
        raise SystemExit(
            f"{args.train_source} looks like an official evaluation/domain split. "
            "Use DocTamperV1-TrainingSet for train/val, or pass "
            "--allow-eval-source-for-training only for a clearly labeled prototype."
        )

    train_n = get_lmdb_num_samples(data_root / args.train_source)
    test_n = get_lmdb_num_samples(data_root / args.test_source)
    if args.train_size + args.val_size > train_n:
        raise SystemExit(f"Requested train+val={args.train_size + args.val_size}, but {args.train_source} has {train_n}")
    if args.test_size > test_n:
        raise SystemExit(f"Requested test={args.test_size}, but {args.test_source} has {test_n}")

    metadata = _load_tamper_types(args.tamper_types_pickle, args.allow_unsafe_pickle)
    train_pool = _shuffled_indices(train_n, args.seed)
    if args.train_source == args.test_source:
        needed = args.train_size + args.val_size + args.test_size
        if needed > train_n:
            raise SystemExit(f"Requested train+val+test={needed}, but {args.train_source} has {train_n}")
        train_idx = train_pool[: args.train_size]
        val_idx = train_pool[args.train_size : args.train_size + args.val_size]
        test_idx = train_pool[args.train_size + args.val_size : needed]
    else:
        test_pool = _shuffled_indices(test_n, args.seed + 1)
        train_idx = train_pool[: args.train_size]
        val_idx = train_pool[args.train_size : args.train_size + args.val_size]
        test_idx = test_pool[: args.test_size]

    manifests = [
        _build_manifest("train", args.train_source, train_idx, args.seed, metadata),
        _build_manifest("val", args.train_source, val_idx, args.seed, metadata),
        _build_manifest("test", args.test_source, test_idx, args.seed, metadata),
    ]
    assert_no_overlap(manifests)

    if not args.skip_file_validation:
        for manifest in manifests:
            errors = validate_manifest_against_lmdb(manifest, data_root)
            if errors:
                raise SystemExit("\n".join(errors[:20]))

    out_dir.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        write_json(out_dir / f"{manifest['split']}.json", manifest)
    summary = {
        "seed": args.seed,
        "data_root": str(data_root),
        "train_source": args.train_source,
        "test_source": args.test_source,
        "manifests": {m["split"]: f"{m['split']}.json" for m in manifests},
        "sizes": {m["split"]: m["size"] for m in manifests},
        "note": "Original LMDB files are unchanged. Manifests select deterministic sample IDs.",
    }
    write_json(out_dir / "subset_summary.json", summary)
    print(f"Wrote manifests to {out_dir}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
