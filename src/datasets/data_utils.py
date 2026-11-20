"""Utilities for exploring and validating fetal and infant brain MRI data.

Provides helpers for loading NIfTI images, inspecting MRI properties,
counting segmentation labels, and detecting basic data quality issues.
Used during FeTA and BOBs dataset exploration and quality assessment.
"""

from pathlib import Path

import nibabel as nib
import numpy as np


def find_subjects(dataset_dir: Path) -> list[Path]:
    """Find subject directories in a dataset.

    Args:
        dataset_dir: Root directory containing BIDS-style subject folders.

    Returns:
        Sorted list of directories whose names start with 'sub-'.
    """
    return sorted(path for path in dataset_dir.glob("sub-*") if path.is_dir())


def load_nifti(file_path: Path, is_segmentation: bool = False):
    """Load a NIfTI MRI or segmentation mask.

    Args:
        file_path: Path to the NIfTI file (.nii or .nii.gz).
        is_segmentation: If True, preserve the original label values.
            Otherwise, load MRI intensities as float32.

    Returns:
        Tuple containing the NIfTI image object and its NumPy array.
    """
    image = nib.load(str(file_path))

    if is_segmentation:
        # Preserve discrete label values without unnecessary conversion.
        data = np.asarray(image.dataobj)
    else:
        data = image.get_fdata(dtype=np.float32)

    return image, data


def inspect_volume(image, data: np.ndarray) -> dict:
    """Extract basic spatial and intensity information from an MRI volume.

    Args:
        image: Loaded NIfTI image object.
        data: NumPy array containing image intensities.

    Returns:
        Dictionary containing volume shape, voxel spacing, orientation,
        and intensity statistics (min, max, mean, standard deviation).
    """
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
    """Count voxels belonging to each segmentation class.

    Args:
        seg: Segmentation mask containing discrete class labels.

    Returns:
        Dictionary mapping each class label to its voxel count.
    """
    labels, counts = np.unique(seg, return_counts=True)

    return {int(label): int(count) for label, count in zip(labels, counts)}


def check_volume_quality(
    mri: np.ndarray,
    seg: np.ndarray,
    expected_labels: set[int],
) -> dict:
    """Check MRI and segmentation arrays for basic data quality issues.

    Checks for non-finite values, constant MRI intensity, empty
    segmentations, non-integer labels, and unexpected class IDs.

    Args:
        mri: NumPy array containing MRI intensities.
        seg: NumPy array containing segmentation labels.
        expected_labels: Set of valid segmentation class IDs.

    Returns:
        Dictionary containing error counts and boolean quality flags.
        Boolean values indicate whether a potential issue was detected.

    Note:
        This function reports quality issues without modifying the data
        or raising exceptions.
    """
    finite_values = np.isfinite(mri)

    # Exclude NaN/Inf values when inspecting segmentation labels.
    seg_finite = np.isfinite(seg)
    seg_values = seg[seg_finite]
    unique_labels = np.unique(seg_values)

    return {
        "mri_nan": int(np.isnan(mri).sum()),
        "mri_inf": int(np.isinf(mri).sum()),
        "mri_constant": bool(np.all(finite_values) and np.min(mri) == np.max(mri)),
        "seg_nan": int(np.isnan(seg).sum()),
        "seg_inf": int(np.isinf(seg).sum()),
        "seg_non_integer": bool(np.any(seg_values != np.round(seg_values))),
        "seg_empty": bool(not np.any(seg_values > 0)),
        "unexpected_labels": sorted(
            float(label) for label in unique_labels if label not in expected_labels
        ),
    }
