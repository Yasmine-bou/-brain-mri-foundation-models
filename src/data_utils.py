
from pathlib import Path

import nibabel as nib
import numpy as np


def find_subjects(dataset_dir: Path) -> list[Path]:
    """Return sorted subject directories in a dataset."""
    return sorted(
        path for path in dataset_dir.glob("sub-*")
        if path.is_dir()
    )


def load_nifti(file_path: Path, is_segmentation: bool = False):
    """Load a NIfTI file and return its image object and NumPy array."""
    image = nib.load(str(file_path))

    if is_segmentation:
        data = np.asarray(image.dataobj)
    else:
        data = image.get_fdata(dtype=np.float32)

    return image, data


def inspect_volume(image, data: np.ndarray) -> dict:
    """Return basic properties of a 3D NIfTI volume."""
    return {
        "shape": data.shape,
        "voxel_spacing": image.header.get_zooms()[:3],
        "orientation": nib.aff2axcodes(image.affine),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
    }

def count_segmentation_labels(seg: np.ndarray) -> dict:
    """Count the number of voxels belonging to each segmentation label."""
    labels, counts = np.unique(seg, return_counts=True)

    return {
        int(label): int(count)
        for label, count in zip(labels, counts)
    }



def check_volume_quality(
    mri: np.ndarray,
    seg: np.ndarray,
    expected_labels: set[int],
) -> dict:
    """Check MRI and segmentation arrays for basic data quality issues."""

    # MRI checks
    finite_values = np.isfinite(mri)

    # Segmentation checks
    seg_finite = np.isfinite(seg)
    seg_values = seg[seg_finite]
    unique_labels = np.unique(seg_values)

    return {
        "mri_nan": int(np.isnan(mri).sum()),
        "mri_inf": int(np.isinf(mri).sum()),
        "mri_constant": bool(
            np.all(finite_values)
            and np.min(mri) == np.max(mri)
        ),
        "seg_nan": int(np.isnan(seg).sum()),
        "seg_inf": int(np.isinf(seg).sum()),
        "seg_non_integer": bool(
            np.any(seg_values != np.round(seg_values))
        ),
        "seg_empty": bool(
            not np.any(seg_values > 0)
        ),
        "unexpected_labels": sorted(
            float(label)
            for label in unique_labels
            if label not in expected_labels
        ),
    }
