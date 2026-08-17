"""Build the qualitative figure with an added tamper-probability heatmap.

This script is meant to be run in the same Colab/Drive environment used for
`qualitative_redundancy_colab.ipynb` and `final_results_colab.ipynb`. It does
not modify either notebook. It recomputes the baseline and +prior probability
maps, selects the same kind of examples used in the presentation, and writes a
publication-ready PDF.

Example:
    python scripts/make_qualitative_with_probability_heatmap.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

import numpy as np


MEAN = np.array([0.485, 0.455, 0.406])
STD = np.array([0.229, 0.224, 0.225])
IMG = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default="/content/HLCV-Project")
    parser.add_argument("--doctamper-dir", default="/content/DocTamper")
    parser.add_argument("--data-root", default="/content/doctamper_train_test")
    parser.add_argument(
        "--manifest-dir",
        default="/content/drive/MyDrive/HLCV_samira/manifests/train800_val200_test200_seed42",
    )
    parser.add_argument("--checkpoint-dir", default="/content/drive/MyDrive/HLCV/checkpoints")
    parser.add_argument("--final-out", default="/content/drive/MyDrive/HLCV_samira/final_seed42")
    parser.add_argument("--ta-out", default="/content/drive/MyDrive/HLCV_samira/ta_design")
    parser.add_argument(
        "--out",
        default="/content/drive/MyDrive/HLCV_samira/qualitative/qualitative_with_probability_heatmap.pdf",
    )
    parser.add_argument("--prior-arm", default="easyocr")
    parser.add_argument("--init-checkpoint", default="dtd_doctamper.pth")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--text-feat-idx", type=int, default=0)
    parser.add_argument("--ocr-confidence-threshold", type=float, default=30.0)
    parser.add_argument("--ocr-dilation", type=int, default=2)
    parser.add_argument("--ocr-languages", default="eng")
    parser.add_argument("--n-top-diff", type=int, default=4)
    parser.add_argument("--explicit-indices", default="")
    parser.add_argument("--scan-limit", type=int, default=0)
    return parser.parse_args()


def require(path: Path, note: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {note}: {path}")
    return path


def denorm(t):
    arr = t.permute(1, 2, 0).cpu().numpy() * STD + MEAN
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def batch_from_sample(sample):
    import torch

    batch = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        elif isinstance(value, np.ndarray):
            batch[key] = torch.from_numpy(value).unsqueeze(0)
        else:
            batch[key] = [value]
    return batch


def copy_if_needed(src: Path, dst: Path) -> None:
    if not dst.exists():
        shutil.copy2(src, dst)


def setup_runtime(args: argparse.Namespace):
    project_dir = require(Path(args.project_dir), "project checkout")
    doctamper_models = require(Path(args.doctamper_dir) / "models", "DocTamper/models")
    sys.path.insert(0, str(project_dir))
    os.chdir(doctamper_models)

    checkpoint_dir = require(Path(args.checkpoint_dir), "checkpoint directory")
    copy_if_needed(require(Path(args.doctamper_dir) / "qt_table.pk", "qt_table.pk"), Path("qt_table.pk"))
    for name in ["vph_imagenet.pt", "swin_imagenet.pt", args.init_checkpoint]:
        copy_if_needed(require(checkpoint_dir / name, name), Path(name))


def build_components(args: argparse.Namespace):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast
    from torch.utils.data import DataLoader

    from dtd import seg_dtd
    from src.doctamper_dataset import ManifestDocTamperDataset
    from src.fusion import TextPriorFusion
    from src.ocr_backends import OCRConfig, create_ocr_backend
    from src.ocr_cache import OCRDetectionCache

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    data_root = require(Path(args.data_root), "DocTamper data root")
    manifest = require(Path(args.manifest_dir) / "test.json", "test manifest")
    test_ds = ManifestDocTamperDataset(data_root, manifest, "qt_table.pk", args.jpeg_quality)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    def build_model():
        model = seg_dtd("", 2).to(device)
        for mod in model.modules():
            if isinstance(mod, nn.GELU) and not hasattr(mod, "approximate"):
                mod.approximate = "none"
        state = torch.load(args.init_checkpoint, map_location="cpu")["state_dict"]
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
        return model

    def forward_dtd(model, batch):
        return model(batch["image"].to(device), batch["rgb"].to(device), batch["q"].unsqueeze(1).to(device))

    class HeadWithPrior(nn.Module):
        def __init__(self, head, in_ch):
            super().__init__()
            self.fusion = TextPriorFusion(in_ch)
            self.head = head
            self.text_mask = None

        def forward(self, feat):
            return self.head(self.fusion(feat, self.text_mask))

    def wire_prior(model):
        core = model.model
        head = core.segmentation_head
        while hasattr(head, "head"):
            head = head.head
        core.segmentation_head = HeadWithPrior(head, head[0].in_channels).to(device)
        return core

    class TextHead(nn.Module):
        def __init__(self, in_ch, mid=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(True),
                nn.Conv2d(mid, mid, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(True),
                nn.Conv2d(mid, 1, 1),
            )

        def forward(self, x):
            return self.net(x)

    languages = tuple(x.strip() for x in args.ocr_languages.split(",") if x.strip())
    backend_langs = tuple("en" if x == "eng" else x for x in languages) if args.prior_arm == "easyocr" else languages
    conf = 0.0 if args.prior_arm == "easyocr" and args.ocr_confidence_threshold > 1 else args.ocr_confidence_threshold
    ocr_cfg = OCRConfig(
        backend=args.prior_arm,
        languages=backend_langs,
        confidence_threshold=conf,
        dilation=args.ocr_dilation,
        easyocr_gpu=(device == "cuda"),
    )
    ocr_backend = create_ocr_backend(ocr_cfg)
    ocr_cache = OCRDetectionCache(Path(args.final_out) / "ocr_cache", ocr_cfg)

    @torch.no_grad()
    def predict(model, idx, use_prior):
        sample = test_ds[idx]
        batch = batch_from_sample(sample)
        if use_prior:
            image = denorm(sample["image"])
            det, _ = ocr_cache.get_or_compute(sample["sample_id"], image, ocr_backend)
            mask = ocr_cache.mask_from_detections(det, image.shape)
            model.model.segmentation_head.text_mask = torch.from_numpy(mask[None, None]).float().to(device)
        with autocast(enabled=(device == "cuda")):
            logits = forward_dtd(model, batch)
        logits = F.interpolate(logits.float(), size=(IMG, IMG), mode="bilinear", align_corners=False)
        return torch.softmax(logits, 1)[0, 1].cpu().numpy()

    @torch.no_grad()
    def capture_row(base_model, text_head, idx):
        sample = test_ds[idx]
        batch = batch_from_sample(sample)
        grabbed = {}
        hook = base_model.model.vph.register_forward_hook(lambda _m, _i, out: grabbed.update(out=out))
        with autocast(enabled=(device == "cuda")):
            _ = forward_dtd(base_model, batch)
        hook.remove()
        feats = list(grabbed["out"]) if isinstance(grabbed["out"], (list, tuple)) else [grabbed["out"]]
        feat = feats[args.text_feat_idx].float()
        text_prob = None
        if text_head is not None:
            text_logits = text_head(feat)
            text_logits = F.interpolate(text_logits, size=(IMG, IMG), mode="bilinear", align_corners=False)
            text_prob = torch.sigmoid(text_logits)[0, 0].cpu().numpy()
        image = denorm(sample["image"])
        det, _ = ocr_cache.get_or_compute(sample["sample_id"], image, ocr_backend)
        ocr_mask = ocr_cache.mask_from_detections(det, image.shape).astype(np.float32)
        gt = sample["label"][0].numpy() > 0
        return {"image": image, "ocr": ocr_mask, "text": text_prob, "gt": gt, "sample_id": sample["sample_id"]}

    def load_text_head():
        channels = [96, 192, 384, 768][args.text_feat_idx]
        for candidate in sorted(Path(args.ta_out).glob("**/*.pth")):
            try:
                blob = torch.load(candidate, map_location="cpu")
            except Exception:
                continue
            if isinstance(blob, dict) and "text_head" in blob:
                head = TextHead(channels).to(device)
                head.load_state_dict(blob["text_head"])
                head.eval()
                print(f"Loaded text head: {candidate}")
                return head
        print("No saved text head found; the text-head column will be blank.")
        return None

    return {
        "torch": torch,
        "F": F,
        "device": device,
        "test_ds": test_ds,
        "test_loader": test_loader,
        "build_model": build_model,
        "wire_prior": wire_prior,
        "predict": predict,
        "capture_row": capture_row,
        "load_text_head": load_text_head,
    }


def component_stats(mask: np.ndarray) -> tuple[int, int]:
    import cv2

    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = [int(a) for a in stats[1:, cv2.CC_STAT_AREA] if int(a) > 0]
    return len(areas), min(areas) if areas else 0


def select_indices(args: argparse.Namespace, ctx: dict) -> tuple[list[int], list[dict]]:
    torch = ctx["torch"]
    test_ds = ctx["test_ds"]
    build_model = ctx["build_model"]
    wire_prior = ctx["wire_prior"]
    predict = ctx["predict"]

    if args.explicit_indices.strip():
        indices = [int(x) for x in args.explicit_indices.split(",") if x.strip()]
        return indices, [{"idx": i, "label": f"selected #{j + 1}"} for j, i in enumerate(indices)]

    prior_path = Path(args.final_out) / "arms" / f"{args.prior_arm}.pth"
    require(prior_path, "+prior checkpoint")

    base_model = build_model()
    base_model.eval()
    prior_model = build_model()
    wire_prior(prior_model)
    state = torch.load(prior_path, map_location="cpu")["state_dict"]
    prior_model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    prior_model.eval()

    rows = []
    n_scan = min(len(test_ds), args.scan_limit) if args.scan_limit else len(test_ds)
    for idx in range(n_scan):
        sample = test_ds[idx]
        gt = sample["label"][0].numpy() > 0
        if not gt.any():
            continue
        p_base = predict(base_model, idx, False)
        p_prior = predict(prior_model, idx, True)
        diff = np.abs(p_prior - p_base)
        changed = int(((p_base > args.threshold) != (p_prior > args.threshold)).sum())
        n_comp, min_comp = component_stats(gt)
        rows.append(
            {
                "idx": idx,
                "p_base": p_base,
                "p_prior": p_prior,
                "maxdiff": float(diff.max()),
                "meandiff": float(diff.mean()),
                "changed": changed,
                "n_comp": n_comp,
                "min_comp": min_comp,
                "area": int(gt.sum()),
            }
        )
        if (idx + 1) % 25 == 0:
            print(f"scanned {idx + 1}/{n_scan}")

    chosen = sorted(rows, key=lambda r: (-r["maxdiff"], -r["changed"]))[: args.n_top_diff]
    used = {r["idx"] for r in chosen}

    medium = [
        r
        for r in rows
        if r["idx"] not in used and r["n_comp"] == 1 and 200 <= r["min_comp"] <= 700 and r["changed"] > 0
    ]
    if medium:
        chosen.append(sorted(medium, key=lambda r: (abs(r["min_comp"] - 365), -r["changed"]))[0])
        used.add(chosen[-1]["idx"])

    small = [
        r
        for r in rows
        if r["idx"] not in used and r["n_comp"] <= 3 and 1 <= r["min_comp"] <= 30 and r["changed"] > 0
    ]
    if small:
        chosen.append(sorted(small, key=lambda r: (abs(r["min_comp"] - 10), -r["changed"]))[0])

    labels = []
    for rank, row in enumerate(chosen):
        if rank < args.n_top_diff:
            name = f"max-diff #{rank + 1}"
        elif row["n_comp"] == 1:
            name = f"medium\\n{row['n_comp']} comp min {row['min_comp']}px"
        else:
            name = f"single-small\\n{row['n_comp']} comps min {row['min_comp']}px"
        labels.append({**row, "label": name})

    del base_model, prior_model
    if ctx["device"] == "cuda":
        torch.cuda.empty_cache()
    return [r["idx"] for r in labels], labels


def overlay_contours(image: np.ndarray, gt=None, pred=None, pred_color=(255, 190, 0)) -> np.ndarray:
    import cv2

    out = image.copy()
    if gt is not None:
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 210, 255), 2)
    if pred is not None:
        contours, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, pred_color, 2)
    return out


def save_figure(args: argparse.Namespace, ctx: dict, labels: list[dict]) -> None:
    import matplotlib.pyplot as plt

    build_model = ctx["build_model"]
    wire_prior = ctx["wire_prior"]
    predict = ctx["predict"]
    capture_row = ctx["capture_row"]
    load_text_head = ctx["load_text_head"]
    torch = ctx["torch"]

    prior_path = Path(args.final_out) / "arms" / f"{args.prior_arm}.pth"
    require(prior_path, "+prior checkpoint")

    base_model = build_model()
    base_model.eval()
    prior_model = build_model()
    wire_prior(prior_model)
    state = torch.load(prior_path, map_location="cpu")["state_dict"]
    prior_model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    prior_model.eval()
    text_head = load_text_head()

    n_rows = len(labels)
    titles = [
        "Document + GT",
        "OCR mask M_text",
        "Text head (frozen VPH)",
        "P(ours + prior)",
        "GT + baseline",
        "GT + ours (+prior)",
    ]
    fig, axes = plt.subplots(n_rows, len(titles), figsize=(18.8, 3.0 * n_rows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for r, info in enumerate(labels):
        row = capture_row(base_model, text_head, info["idx"])
        if "p_base" in info and "p_prior" in info:
            p_base = info["p_base"]
            p_prior = info["p_prior"]
        else:
            p_base = predict(base_model, info["idx"], False)
            p_prior = predict(prior_model, info["idx"], True)
            diff = np.abs(p_prior - p_base)
            info["maxdiff"] = float(diff.max())
            info["changed"] = int(((p_base > args.threshold) != (p_prior > args.threshold)).sum())

        pred_base = p_base > args.threshold
        pred_prior = p_prior > args.threshold
        panels = [
            overlay_contours(row["image"], gt=row["gt"]),
            row["ocr"],
            row["text"] if row["text"] is not None else np.zeros_like(row["gt"], dtype=np.float32),
            p_prior,
            overlay_contours(row["image"], gt=row["gt"], pred=pred_base, pred_color=(220, 40, 40)),
            overlay_contours(row["image"], gt=row["gt"], pred=pred_prior, pred_color=(255, 190, 0)),
        ]
        cmaps = [None, "viridis", "viridis", "magma", None, None]

        for c, (panel, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[r, c]
            if cmap is None:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)

        axes[r, 0].set_ylabel(info["label"], fontsize=8)
        axes[r, 5].set_xlabel(
            f"{int(info.get('changed', 0)):,} px changed   max|d|={float(info.get('maxdiff', 0.0)):.3f}",
            fontsize=8,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"saved {out}")
    print(f"saved {out.with_suffix('.png')}")


def main() -> None:
    args = parse_args()
    setup_runtime(args)
    ctx = build_components(args)
    _indices, labels = select_indices(args, ctx)
    print("selected rows:")
    for row in labels:
        print(
            f"  idx={row['idx']:>4} {row['label'].replace(chr(10), ' '):<28} "
            f"changed={int(row.get('changed', 0)):>5} max|d|={float(row.get('maxdiff', 0.0)):.3f}"
        )
    save_figure(args, ctx, labels)


if __name__ == "__main__":
    main()
