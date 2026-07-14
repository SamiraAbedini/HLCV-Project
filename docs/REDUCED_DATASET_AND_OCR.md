# Reduced Dataset and OCR Backend Workflow

This project should keep the original DocTamper LMDB folders unchanged. The
recommended reduced setup is a set of JSON manifests containing stable sample
IDs and integer LMDB indices.

## Why Manifests Instead of Copying Data?

Option A, copying files into a smaller directory, is convenient to browse and
transfer, but DocTamper is stored as LMDB plus compression-record pickles. Copying
samples out would either duplicate many gigabytes or require rebuilding LMDB and
matching DCT/compression metadata.

Option B, saving deterministic manifests, leaves the original dataset untouched,
uses little disk space, and keeps the upstream DTD loader intact. The trade-off is
that training code must wrap the dataset with `ManifestSubset`.

For this repository, manifests are safer and more reproducible.

## Generate Manifests

Windows local example:

```powershell
.\.venv\Scripts\python.exe scripts\generate_doctamper_subset.py `
  --data-root data\doctamper_data `
  --output-dir data\manifests\colab_2400_seed42 `
  --seed 42 `
  --train-size 1600 `
  --val-size 400 `
  --test-size 400
```

Google Colab example:

```python
!python scripts/generate_doctamper_subset.py \
  --data-root /content/drive/MyDrive/DocTamper/data \
  --output-dir /content/drive/MyDrive/HLCV/manifest_colab_2400_seed42 \
  --seed 42 --train-size 1600 --val-size 400 --test-size 400
```

The default train/validation source is `DocTamperV1-TrainingSet`. The default
test source is `DocTamperV1-TestingSet`. `DocTamperV1-FCD` and `DocTamperV1-SCD`
should be kept for evaluation unless an experiment is explicitly marked as a
prototype.

## Use Manifests With the Existing Loader

```python
from src.doctamper_lmdb import ManifestSubset, load_manifest

dataset = TamperDataset("DocTamperV1-TrainingSet", mode="train", minq=75)
train_subset = ManifestSubset(dataset, load_manifest("train.json"))
```

Use separate `train.json`, `val.json`, and `test.json` files. This makes it easy
to guarantee that the baseline, EasyOCR, and Tesseract experiments use exactly
the same samples.

If the public DocTamper `pks/` folder does not contain compression records for
`DocTamperV1-TrainingSet`, use `src.doctamper_dataset.ManifestDocTamperDataset`
in Colab. It returns the same DTD tensor keys but uses a fixed JPEG quality
table, which is reproducible and avoids training on FCD/SCD/TestingSet.

## Validate Manifests

```powershell
.\.venv\Scripts\python.exe scripts\validate_doctamper_subset.py `
  --data-root data\doctamper_data `
  data\manifests\colab_2400_seed42\train.json `
  data\manifests\colab_2400_seed42\val.json `
  data\manifests\colab_2400_seed42\test.json
```

Validation checks duplicate sample IDs, split overlap, missing image/label keys,
image/mask dimensions, and expected manifest size.

## OCR Backends

Backends are selected by config:

- `none`: baseline, all-zero text prior.
- `easyocr`: existing EasyOCR-style external prior.
- `tesseract`: pytesseract word boxes converted to the same binary text mask.

Colab Tesseract install:

```python
!apt-get update -qq
!apt-get install -y -qq tesseract-ocr tesseract-ocr-eng
!pip install -q pytesseract
!tesseract --version
!tesseract --list-langs
```

Windows local install:

1. Install Tesseract from a trusted Windows build.
2. Either add the install folder to `PATH`, set environment variable
   `TESSERACT_CMD`, or pass `--tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"`.

Do not hard-code one teammate's Windows path in committed code.

## OCR Coverage Evaluation

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_ocr_backends.py `
  --data-root data\doctamper_data `
  --manifest data\manifests\colab_2400_seed42\val.json `
  --output-dir runs\ocr_tesseract_val `
  --backend tesseract `
  --languages eng `
  --confidence-threshold 30 `
  --dilation 2
```

The reported OCR numbers are forged-region coverage and selectivity proxies,
not true OCR text-detection precision/recall.

## Realistic Colab Sizes

Start small: `train=800`, `val=200`, `test=200` for debugging. For a T4 GPU,
`train=1600`, `val=400`, `test=400` is a practical first controlled experiment.
Increase only after the notebook runs end-to-end and checkpoints resume cleanly.
