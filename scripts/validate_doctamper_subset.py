#!/usr/bin/env python
"""Validate DocTamper subset manifests."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.doctamper_lmdb import assert_no_overlap, load_manifest, validate_manifest_against_lmdb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("manifests", nargs="+")
    args = parser.parse_args()

    manifests = [load_manifest(path) for path in args.manifests]
    assert_no_overlap(manifests)
    all_errors: list[str] = []
    for path, manifest in zip(args.manifests, manifests):
        errors = validate_manifest_against_lmdb(manifest, args.data_root)
        if errors:
            all_errors.extend([f"{path}: {err}" for err in errors])
    if all_errors:
        print("\n".join(all_errors[:50]))
        return 1
    print("Manifest validation passed: no duplicate IDs, no split overlap, all LMDB pairs exist, shapes match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
