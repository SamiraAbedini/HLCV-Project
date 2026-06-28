# HLCV-Project — OCR-Guided and Boundary-Aware Tampered Text Localization

Course project (High-Level Computer Vision). We extend the **DTD** document
tampering detector (DocTamper) with document-specific priors and
imbalance/boundary-aware supervision, to better localize **small** forged text
regions.

Team: Samira Abedini (7072848), Pardis Rahbarsooreh (7059149).

## Idea

Keep DTD's RGB + frequency backbone unchanged, and add:

1. an **OCR-derived text prior** `M_text` (where text is) fused into the decoder
   as soft guidance toward regions where tampering is likely;
2. **imbalance-aware** (Dice) and **boundary-aware** supervision on top of the
   cross-entropy loss, for small regions and sharp glyph edges.

Following TA feedback, the OCR branch is treated as a **region-proposal** stage
with a high-recall target, and is evaluated on its own (a missed tampered region
cannot be recovered downstream). The stretch goal shares one backbone — an OCR
head on the Visual Perception Head features instead of a second network.

## Repo layout

```
src/
  losses.py      # CE + Dice + Boundary combined loss (CombinedTamperLoss)
  text_prior.py  # OCR boxes -> binary M_text mask (OCRTextMasker)
  fusion.py      # TextPriorFusion: F_hat_l = phi_l([F_l, M_l])
  ocr_eval.py    # OCRCoverageMeter: recall/coverage of the OCR prior
docs/
  INTEGRATION.md # how to wire src/ into DocTamper, on Colab
requirements.txt
```

## How it runs

These modules are self-contained and plug into the DocTamper model, which is
cloned **on Colab** (it is license-restricted and ships untrusted serialized
files, so it is not vendored here). See [docs/INTEGRATION.md](docs/INTEGRATION.md).

## Data & weights — do not commit

DocTamper datasets and checkpoints are license-restricted (no redistribution)
**and** are untrusted executable content (pickle / `torch.load`). They are kept
out of git via `.gitignore` and stay in Colab/Drive only. Inspect upstream
pickles with `python -m pickletools`, never by loading them.
