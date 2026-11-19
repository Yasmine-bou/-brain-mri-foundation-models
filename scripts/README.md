# FeTA Preprocessing — SSL3D / nnU-Net

This guide explains how to preprocess the **FeTA brain MRI dataset** using the official SSL3D/nnU-Net pipeline.

All commands are executed from the **project root in Windows PowerShell**.

## 1. Dataset preparation

We use 80 FeTA subjects with predefined splits:

| Split | Subjects |
|---|---:|
| Train | 48 |
| Validation | 16 |
| Test | 16 |

Splits are stored in `data/splits/feta_splits.json`.

Original data must be placed in `data/FeTA/feta_2.4/`.

Convert the data to nnU-Net format:

```powershell
python scripts/convert_feta_to_nnunet.py
```

Output: `data/nnUNet_raw/Dataset101_FeTA/`

Only the 48 training subjects are stored in `imagesTr`, ensuring that validation and test do not influence the fingerprint and planning.

## 2. Set up Docker

Make sure Docker Desktop is running, then download the SSL3D image:

```powershell
docker pull petermcgor/nnunetv2:1.1.0-nnssl
```

Create output directories:

```powershell
New-Item -ItemType Directory -Force -Path "data/nnUNet_preprocessed" | Out-Null
New-Item -ItemType Directory -Force -Path "data/nnUNet_results" | Out-Null
```

## 3. Fingerprint + Planning

Analyze the 48 training MRIs and generate nnU-Net plans:

```powershell
docker run --rm `
  -v "${PWD}/data/nnUNet_raw:/opt/nnunet_resources/nnUNet_raw" `
  -v "${PWD}/data/nnUNet_preprocessed:/opt/nnunet_resources/nnUNet_preprocessed" `
  -v "${PWD}/data/nnUNet_results:/opt/nnunet_resources/nnUNet_results" `
  petermcgor/nnunetv2:1.1.0-nnssl `
  nnUNetv2_plan_and_preprocess -d 101 --no_pp -np 2
```

**Fingerprint** analyzes the image properties. **Planning** generates the preprocessing configuration.

`--no_pp` means the images are not preprocessed yet.

## 4. SSL3D preprocessing — Train

We use the SSL3D pretrained checkpoint and `like_pretrained` adaptation mode.

On CPU-only machines, `scripts/docker_fixes/like_nnssl.py` contains a small fix (`map_location="cpu"`) to load the checkpoint correctly.

Run:

```powershell
docker run --rm `
  -v "${PWD}/data/nnUNet_raw:/opt/nnunet_resources/nnUNet_raw" `
  -v "${PWD}/data/nnUNet_preprocessed:/opt/nnunet_resources/nnUNet_preprocessed" `
  -v "${PWD}/data/nnUNet_results:/opt/nnunet_resources/nnUNet_results" `
  -v "${PWD}/scripts/docker_fixes/like_nnssl.py:/opt/nnunet_adaptation/nnunetv2/experiment_planning/like_nnssl.py:ro" `
  petermcgor/nnunetv2:1.1.0-nnssl `
  nnUNetv2_preprocess_like_nnssl `
  -d 101 `
  -n SSL3D_FeTA `
  -pc /opt/nnunet_resources/models/BaseMAETrainerExtendedHealth__nnsslPlans__onemmiso/fold_all/checkpoint_best.pth `
  -am like_pretrained `
  -np 1
```

This preprocesses the **48 training MRIs** using SSL3D-compatible plans.

Final spacing: **1 × 1 × 1 mm**, with Z-score normalization.

## 5. Preprocess Validation + Test

Reuse the same SSL3D plans without recomputing the fingerprint:

```powershell
docker run --rm `
  -v "${PWD}/data/nnUNet_raw:/opt/nnunet_resources/nnUNet_raw" `
  -v "${PWD}/data/nnUNet_preprocessed:/opt/nnunet_resources/nnUNet_preprocessed" `
  -v "${PWD}/data/splits:/opt/project_splits:ro" `
  -v "${PWD}/scripts/preprocess_feta_val_test.py:/tmp/preprocess_feta_val_test.py:ro" `
  petermcgor/nnunetv2:1.1.0-nnssl `
  python /tmp/preprocess_feta_val_test.py
```

Expected: `VALIDATION COMPLETE: 16/16` and `TEST COMPLETE: 16/16`.

## 6. Verify preprocessing

Run the automatic quality checks:

```powershell
docker run --rm `
  -v "${PWD}/data/nnUNet_preprocessed:/opt/nnunet_resources/nnUNet_preprocessed:ro" `
  -v "${PWD}/data/splits:/opt/project_splits:ro" `
  -v "${PWD}/scripts/check_feta_preprocessing.py:/tmp/check_feta_preprocessing.py:ro" `
  petermcgor/nnunetv2:1.1.0-nnssl `
  python /tmp/check_feta_preprocessing.py
```

Expected result:

```text
TRAIN: 48/48
VAL: 16/16
TEST: 16/16
TOTAL: 80/80
ALL CHECKS PASSED!
```

For visual inspection, use `scripts/visualize_feta_preprocessing.py`.

## 7. Final output

```text
data/nnUNet_preprocessed/Dataset101_FeTA/
├── Spacing__1.00_1.00_1.00___Norm__Z/  # 48 Train
├── SSL3D_val/                           # 16 Validation
├── SSL3D_test/                          # 16 Test
├── ptPlans__SSL3D_FeTA...json
├── nnUNetPlans.json
└── dataset.json
```

Each subject contains:

- `.b2nd` — preprocessed MRI
- `_seg.b2nd` — segmentation mask
- `.pkl` — preprocessing metadata

**Important:** MRI data and preprocessing outputs are excluded from GitHub. Transfer the preprocessed folder separately to Izar for model training. The same data and splits are used for both the pretrained and random frozen-encoder experiments.
