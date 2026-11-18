
"""Validate preprocessed FeTA or BOBs MRI and segmentation files.

Checks spatial consistency, intensity normalization, and segmentation
labels. Prints a summary of passed and failed subjects.

Usage (from the project root):
    python scripts/check_preprocessing.py                  # FeTA (default)
    python scripts/check_preprocessing.py --dataset bobs
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where each processed dataset lives, how its masks are named,
# and which label values are allowed.
DATASETS = {
    "feta": {
        "dir": PROJECT_ROOT / "data" / "processed" / "feta_fomo",
        "seg_suffix": "_dseg",
        "labels": set(range(8)),                 # 0-7
    },
    "bobs": {
        "dir": PROJECT_ROOT / "data" / "processed" / "bobs_fomo",
        "seg_suffix": "_desc-bobs9_dseg",
        "labels": {0, 2, 3, 5, 6, 7, 8, 9},      # values found in BOBs9 during EDA
    },
}

EXPECTED_SPACING = (1.0, 1.0, 1.0)


def check_subject(
    img_path: Path,
    seg_suffix: str = "_dseg",
    expected_labels: set[int] = DATASETS["feta"]["labels"],
) -> list[str]:
    """Return validation errors for one preprocessed MRI/mask pair."""
    errors = []

    seg_path = img_path.with_name(
        img_path.name.replace("_T2w.nii.gz", f"{seg_suffix}.nii.gz")
    )

    if not seg_path.exists():
        return ["Missing segmentation mask"]

    img = nib.load(img_path)
    seg = nib.load(seg_path)

    data = img.get_fdata(dtype=np.float32)
    mask = np.asarray(seg.dataobj)

    # Spatial consistency
    if img.shape != seg.shape:
        errors.append("Shape mismatch")

    if not np.allclose(img.affine, seg.affine):
        errors.append("Affine mismatch")

    if nib.aff2axcodes(img.affine) != ("R", "A", "S"):
        errors.append("MRI orientation is not RAS")

    if nib.aff2axcodes(seg.affine) != ("R", "A", "S"):
        errors.append("Mask orientation is not RAS")

    if not np.allclose(nib.affines.voxel_sizes(img.affine), EXPECTED_SPACING):
        errors.append("Incorrect voxel spacing")

    # MRI intensity checks
    if not np.isfinite(data).all():
        errors.append("NaN or Inf intensities")

    foreground = data != 0

    if not foreground.any():
        errors.append("Empty MRI foreground")
    else:
        mean = data[foreground].mean()
        std = data[foreground].std()

        if not np.isclose(mean, 0, atol=1e-3):
            errors.append("Incorrect Z-score mean")

        if not np.isclose(std, 1, atol=1e-3):
            errors.append("Incorrect Z-score standard deviation")

    # Segmentation checks
    if not np.isfinite(mask).all():
        errors.append("NaN or Inf segmentation values")
    elif not np.all(np.isin(np.unique(mask), list(expected_labels))):
        errors.append("Unexpected segmentation labels")

    if not np.any(mask > 0):
        errors.append("Empty segmentation")

    # NumPy copies: must exist and contain the same arrays as the NIfTI files
    img_npy = img_path.with_suffix("").with_suffix(".npy")
    seg_npy = seg_path.with_suffix("").with_suffix(".npy")

    if not img_npy.exists() or not seg_npy.exists():
        errors.append("Missing .npy file")
    else:
        if not np.array_equal(np.load(img_npy), data):
            errors.append("Image .npy differs from NIfTI")
        if not np.array_equal(np.load(seg_npy), mask):
            errors.append("Mask .npy differs from NIfTI")

    return errors


def main():
    """Check all processed subjects of one dataset and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="feta")
    parser.add_argument("--data-dir", type=Path, help="Override the processed data directory.")
    args = parser.parse_args()

    settings = DATASETS[args.dataset]
    data_dir = args.data_dir or settings["dir"]

    image_paths = sorted(data_dir.glob("sub-*/**/anat/*_T2w.nii.gz"))

    if not image_paths:
        raise FileNotFoundError(f"No preprocessed MRIs found in {data_dir}")

    passed = 0
    failed = 0

    for img_path in image_paths:
        errors = check_subject(img_path, settings["seg_suffix"], settings["labels"])

        if errors:
            failed += 1
            print(f"FAIL: {img_path.name}: {', '.join(errors)}")
        else:
            passed += 1
            print(f"PASS: {img_path.name}")

    print("\n--- Preprocessing quality control ---")
    print(f"Total subjects: {len(image_paths)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
