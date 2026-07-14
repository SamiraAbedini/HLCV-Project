# Colab Workflows

Use these notebooks from the `colab-tesseract-workflow` branch.

## 1. FCD Debug Notebook

Notebook:

```text
notebooks/fcd_debug_tesseract_colab.ipynb
```

Purpose:

- quick smoke test that Colab, Tesseract, OCR caching, DTD checkpoint loading,
  training, evaluation, qualitative figures, and reports all work;
- uses a tiny split from `DocTamperV1-FCD`;
- should not be reported as a final scientific result.

Open in Colab:

```text
https://colab.research.google.com/github/SamiraAbedini/HLCV-Project/blob/colab-tesseract-workflow/notebooks/fcd_debug_tesseract_colab.ipynb
```

## 2. Main Reduced TrainingSet/TestingSet Notebook

Notebook:

```text
notebooks/main_train_test_tesseract_colab.ipynb
```

Purpose:

- main reduced experiment;
- train/validation samples come from `DocTamperV1-TrainingSet`;
- test samples come from `DocTamperV1-TestingSet`;
- original LMDB files are not modified;
- deterministic JSON manifests are saved to Drive;
- the Kaggle zip is downloaded to Colab temporary disk, only TrainingSet and
  TestingSet are extracted, then the zip is deleted.

Open in Colab:

```text
https://colab.research.google.com/github/SamiraAbedini/HLCV-Project/blob/colab-tesseract-workflow/notebooks/main_train_test_tesseract_colab.ipynb
```

## Required Drive Files

Put checkpoints here:

```text
MyDrive/HLCV/checkpoints/
  vph_imagenet.pt
  swin_imagenet.pt
  dtd_doctamper.pth
```

Do not commit checkpoints, LMDB files, Kaggle zips, or trained model files to
Git.

## Recommended Experiment Order

1. Run the FCD debug notebook once.
2. Run the main notebook with `OCR_BACKEND = "tesseract"`.
3. Re-run the main notebook with the same manifests and `OCR_BACKEND = "none"`.
4. Re-run the main notebook with the same manifests and `OCR_BACKEND = "easyocr"`.
5. Generate the comparison report from the saved run folders.

The final comparison should use the main notebook outputs, not the FCD debug
outputs.
