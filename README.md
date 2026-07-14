# Text-Aware Tampered Text Localization

Document forgery localization that extends the **DTD** detector (DocTamper) with
document-specific priors and imbalance/boundary-aware supervision, to better
localize small forged text regions.

Team: Samira Abedini (7072848), Pardis Rahbarsooreh (7059149).

## Method

Keep DTD's RGB + DCT-frequency backbone, and add:

1. an **OCR-derived text prior** `M_text` fused into the decoder as soft guidance
   toward text regions, where tampering is most likely;
2. **imbalance-aware** (Dice) and **boundary-aware** losses on top of
   cross-entropy, for small regions and sharp glyph edges.

The OCR prior is also evaluated on its own as a region-proposal stage — its
recall / coverage of the tampered regions — since a missed region cannot be
recovered downstream.

## Repo layout

```
src/
  losses.py      CombinedTamperLoss (CE + Dice + boundary)
  text_prior.py  OCR boxes -> binary M_text mask (OCRTextMasker)
  fusion.py      TextPriorFusion: F_hat_l = phi_l([F_l, M_l])
  ocr_eval.py    OCRCoverageMeter: recall / coverage of the OCR prior
notebooks/
  ocr_prior_recall_check.ipynb   evaluate the OCR prior (recall, dilate sweep, misses)
  phase1_train.ipynb             train (baseline / +OCR) and evaluate Pixel-F1 / IoU / AUC
  fcd_debug_tesseract_colab.ipynb quick Colab smoke test; not final science
  main_train_test_tesseract_colab.ipynb main reduced TrainingSet/TestingSet workflow
docs/
  INTEGRATION.md
  COLAB_WORKFLOWS.md
requirements.txt
```

## Running

Run the notebooks on Colab (GPU). They clone the DocTamper repo, build the data
pipeline, and plug in the modules from `src/`. The training notebook reports
Pixel-F1, precision, recall, IoU, and AUC; the OCR notebook reports the prior's
recall / coverage. See [docs/INTEGRATION.md](docs/INTEGRATION.md) for how `src/`
attaches to the DocTamper model.

For the current Tesseract workflow, start with
[docs/COLAB_WORKFLOWS.md](docs/COLAB_WORKFLOWS.md). Use the FCD notebook only as
a smoke test, then use the main TrainingSet/TestingSet notebook for reportable
reduced experiments.

## Data & weights — do not commit

DocTamper datasets and checkpoints are license-restricted (no redistribution)
and are untrusted serialized content (pickle / `torch.load`). They are kept out
of git via `.gitignore` and stay in Colab/Drive only. Inspect upstream pickles
with `python -m pickletools`, never by loading them.
