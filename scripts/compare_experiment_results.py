#!/usr/bin/env python
"""Create a compact comparison report from experiment metric files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LOCALIZATION_KEYS = ["pixel_f1", "precision", "recall", "foreground_iou", "iou", "auc"]
OCR_KEYS = ["pixel_recall_coverage", "component_recall", "text_area_ratio", "runtime_per_image_sec", "mean_num_boxes"]
TRAIN_KEYS = ["train_size", "val_size", "test_size", "epochs", "steps", "batch_size", "learning_rate", "seed", "ocr_backend"]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _experiment_row(exp_dir: Path) -> dict:
    metrics = _load_json(exp_dir / "metrics.json")
    ocr = _load_json(exp_dir / "ocr_metrics.json")
    config = _load_json(exp_dir / "config.json")
    row = {"experiment": exp_dir.name}
    for key in LOCALIZATION_KEYS:
        if key in metrics:
            row[key] = metrics[key]
    if "foreground_iou" not in row and "iou" in row:
        row["foreground_iou"] = row["iou"]
    for key in OCR_KEYS:
        if key in ocr:
            row[key] = ocr[key]
    for key in TRAIN_KEYS:
        if key in config:
            row[key] = config[key]
    return row


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def _summary(rows: list[dict]) -> str:
    lines = [
        "# OCR Backend Comparison",
        "",
        "This summary separates OCR-mask coverage from downstream tamper-localization quality.",
        "",
    ]
    if not rows:
        lines.append("No experiments were found.")
        return "\n".join(lines)

    best_f1 = max(rows, key=lambda r: float(r.get("pixel_f1", -1)))
    lines.append(f"- Best Pixel-F1 in these files: **{best_f1['experiment']}** ({_fmt(best_f1.get('pixel_f1'))}).")
    for row in rows:
        backend = row.get("ocr_backend", row.get("backend", "unknown"))
        coverage = row.get("pixel_recall_coverage")
        area = row.get("text_area_ratio")
        if coverage is not None and area is not None:
            lines.append(
                f"- {row['experiment']}: OCR backend `{backend}` covers {_fmt(coverage)} of tampered pixels "
                f"with text-area ratio {_fmt(area)}."
            )
            if float(area) > 0.5:
                lines.append(
                    f"  The OCR prior for {row['experiment']} is broad; high coverage alone does not mean it is selective."
                )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Small differences may not be statistically meaningful without multiple seeds.",
            "- OCR coverage is not true OCR precision/recall because DocTamper does not provide text-box ground truth.",
            "- AUC is threshold-independent; F1 and IoU depend on the chosen binary threshold.",
            "",
            "## Recommended Next Experiment",
            "",
            "Repeat the most promising configuration with at least three seeds, keeping the same manifests and checkpoint.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", required=True, help="Directories containing metrics.json/config.json.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = [_experiment_row(Path(path)) for path in args.experiments]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = ["experiment"] + LOCALIZATION_KEYS + OCR_KEYS + TRAIN_KEYS
    keys = list(dict.fromkeys(keys))
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (out / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")

    table = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        table.append("| " + " | ".join(_fmt(row.get(k)) for k in keys) + " |")
    report = _summary(rows) + "\n\n## Compact Table\n\n" + "\n".join(table) + "\n"
    (out / "comparison.md").write_text(report, encoding="utf-8")
    print(f"Wrote comparison report to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
