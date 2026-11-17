
"""Validate preprocessed FeTA MRI and segmentation files.

Checks spatial consistency, intensity normalization, and segmentation
labels. Prints a summary of passed and failed subjects.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "feta_fomo"

EXPECTED_LABELS = set(range(8))
EXPECTED_SPACING = (1.0, 1.0, 1.0)


def check_subject(img_path: Path) -> list[str]:
    """Return validation errors for one preprocessed MRI/mask pair."""
    errors = []

    seg_path = img_path.with_name(
        img_path.name.replace("_T2w.nii.gz", "_dseg.nii.gz")
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
    elif not np.all(np.isin(np.unique(mask), list(EXPECTED_LABELS))):
        errors.append("Unexpected segmentation labels")

    if not np.any(mask > 0):
        errors.append("Empty segmentation")

    return errors


def main():
    """Check all processed FeTA subjects and print a summary."""
    image_paths = sorted(DATA_DIR.glob("sub-*/**/anat/*_T2w.nii.gz"))

    if not image_paths:
        raise FileNotFoundError(f"No preprocessed MRIs found in {DATA_DIR}")

    passed = 0
    failed = 0

    for img_path in image_paths:
        errors = check_subject(img_path)

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
