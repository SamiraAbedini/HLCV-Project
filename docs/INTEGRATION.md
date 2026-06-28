# Integration guide: wiring `src/` into DocTamper (on Colab)

This repo holds the **novel pieces** of the project. The DTD backbone and the
DocTamper dataloaders live in the upstream repo
(<https://github.com/qcf-568/DocTamper>), which is cloned **on Colab** — it is
license-restricted and contains untrusted serialized files, so it is **not**
vendored here (see `.gitignore`).

## Where each module plugs in

The DTD forward pass is, schematically:

```
image --> Visual Perception Head (RGB) -.
                                          >-- Multi-Modality Transformer --> Decoder --> logits
DCT   --> Frequency Perception Head -----'
```

| Our module | Hook point in DocTamper |
|---|---|
| `src.text_prior.OCRTextMasker` | Build `M_text` from a pretrained OCR detector at data-load time (or cache to disk per image). |
| `src.fusion.TextPriorFusion` | Insert before/within the iterative decoder: `feat = fusion(feat, M_text)`. Match `in_channels` to that decoder stage. |
| `src.losses.CombinedTamperLoss` | Replace the training loss. It already uses `CrossEntropyLoss(ignore_index=100)` to match DocTamper's label convention (`gt<8 -> 0`, `gt>160 -> 1`, else `100`). |
| `src.ocr_eval.OCRCoverageMeter` | Standalone evaluation of the OCR prior vs the GT tamper mask. |

## Two implementation phases

**Phase 1 (proposal as written) — start here.** Use an *external* pretrained
OCR detector (EasyOCR or PaddleOCR) to produce `M_text`, fuse it into the
decoder, and train with the combined loss. This is runnable immediately and
already addresses the TA's "use a pretrained OCR" and "measure recall" points.

**Phase 2 (TA's architecture note) — stretch goal.** Avoid a second backbone:
add a lightweight **OCR/text-detection head on top of the existing Visual
Perception Head features** instead of running OCR on raw RGB. The shared
features feed two heads (tampering + text). `TextPriorFusion` is unchanged; only
the *source* of `M_text` moves from an external model to an internal head. Keep
the OCR head's recall as the selection metric (see below).

## Reporting the OCR module (TA's headline ask)

A missed tampered region is unrecoverable, so the OCR prior is a region-proposal
stage with a **high-recall** target. Evaluate it on its own:

```python
from src.ocr_eval import OCRCoverageMeter

meter = OCRCoverageMeter(coverage_thresh=0.5)
for image, tamper_gt in loader:        # tamper_gt: binary HxW
    text_mask = masker(image)          # src.text_prior.OCRTextMasker
    meter.update(text_mask, tamper_gt)
print(meter.compute())
# -> pixel_recall_coverage, component_recall, text_area_ratio, ...
```

Report `pixel_recall_coverage` and `component_recall` (quantitative) plus a few
qualitative masks where coverage fails. Tune `OCRTextMasker(dilate=...)` to push
recall up; watch `text_area_ratio` so the prior does not collapse to "everything
is text".

## Safety / hygiene

* Never `git add` datasets, `.mdb`, or `*.pt/*.pth/*.pk` — `.gitignore` blocks
  them. Keep them in Colab/Drive only.
* Do not `pickle.load` or `torch.load` an upstream file to "inspect" it; use
  `python -m pickletools` for pickles. Run the DocTamper checkpoints only inside
  Colab, never on your laptop.
