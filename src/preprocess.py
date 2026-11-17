
"""MRI preprocessing utilities for brain segmentation.

Provides reusable functions to preprocess T2-weighted MRI volumes and their
segmentation masks, including orientation standardization, resampling,
foreground cropping, optional padding, and intensity normalization.

The current pipeline is configured for FeTA and FOMO25:
- RAS orientation
- 1 mm isotropic voxel spacing
- Foreground cropping with a 5-voxel margin
- Optional zero-padding
- Per-volume foreground Z-score normalization

Spatial transformations are applied to both the MRI and its segmentation.
Intensity normalization is applied only to the MRI.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output


def find_image_label_pairs(
    dataset_dir: Path,
    seg_suffix: str = "_dseg",
) -> list[tuple[Path, Path]]:
    """Find matching T2-weighted MRI and segmentation files.

    Searches the dataset using its BIDS-like folder structure and returns
    the image/segmentation pairs in sorted order.

    Args:
        dataset_dir: Root directory containing the MRI dataset.
        seg_suffix: Filename suffix identifying the segmentation mask.
            Defaults to "_dseg" for FeTA.

    Returns:
        A sorted list of (image_path, segmentation_path) tuples.

    Raises:
        FileNotFoundError: If an MRI is found without its corresponding
            segmentation file.
    """
    dataset_dir = Path(dataset_dir)
    pairs = []

    for img_path in sorted(dataset_dir.glob("sub-*/**/anat/*_T2w.nii.gz")):
        seg_path = img_path.with_name(
            img_path.name.replace("_T2w.nii.gz", f"{seg_suffix}.nii.gz")
        )
        if not seg_path.exists():
            raise FileNotFoundError(f"No segmentation for {img_path.name}")
        pairs.append((img_path, seg_path))

    return pairs


def reorient_to_ras(image: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient a NIfTI volume to RAS anatomical orientation.

    Standardizes the orientation to Right-Anterior-Superior (RAS) by
    rearranging or flipping voxel axes and updating the affine matrix.

    No interpolation is performed, so voxel intensities are preserved.

    Args:
        image: Input NIfTI volume.

    Returns:
        NIfTI volume in the closest canonical RAS orientation.
    """
    return nib.as_closest_canonical(image)


def fill_nan(
    image: nib.Nifti1Image,
    fill_value: float = 0.0,
) -> nib.Nifti1Image:
    """Replace invalid MRI intensity values.

    Replaces NaN, positive infinity, and negative infinity with a
    specified finite value to prevent numerical issues during training.

    Args:
        image: Input NIfTI MRI volume.
        fill_value: Replacement value for NaN and infinite intensities.
            Defaults to 0.

    Returns:
        NIfTI volume with invalid intensities replaced.
    """
    data = image.get_fdata(dtype=np.float32)
    data_filled = np.nan_to_num(data, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return nib.Nifti1Image(data_filled, image.affine, image.header)


def crop_to_foreground(
    image: nib.Nifti1Image,
    seg: nib.Nifti1Image,
    margin: int = 5,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Crop an MRI and its segmentation to the image foreground.

    Identifies MRI voxels with positive intensities and computes the
    smallest bounding box containing them.

    The same bounding box is applied to the MRI and segmentation to
    preserve their spatial correspondence.

    Args:
        image: Input NIfTI MRI volume.
        seg: Corresponding segmentation mask, assumed to share the
            same voxel grid as the MRI.
        margin: Additional voxels retained around the foreground
            along each axis. Defaults to 5.

    Returns:
        Tuple containing:
            - Cropped MRI volume.
            - Cropped segmentation mask.

        The NIfTI affine is updated by the slicing operation to
        preserve the physical coordinates of the retained voxels.

    Raises:
        ValueError: If the MRI contains no positive-intensity voxels.
    """
    data = image.get_fdata(dtype=np.float32)
    foreground = np.argwhere(data > 0)
    if foreground.size == 0:
        raise ValueError("Image has no non-zero voxels, cannot crop.")

    start = np.maximum(foreground.min(axis=0) - margin, 0)
    end = np.minimum(foreground.max(axis=0) + 1 + margin, data.shape)
    box = tuple(slice(s, e) for s, e in zip(start, end))

    return image.slicer[box], seg.slicer[box]


def resample_to_spacing(
    image: nib.Nifti1Image,
    seg: nib.Nifti1Image,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Resample an MRI and its segmentation to a target voxel spacing.

    Standardizes the physical resolution of MRI volumes while using
    different interpolation methods for images and segmentation masks.

    - MRI: linear interpolation (order=1).
    - Segmentation: nearest-neighbor interpolation (order=0), preserving
      discrete class labels.

    The MRI and segmentation are currently resampled independently.
    Their output grids must therefore be checked for alignment.

    Args:
        image: Input NIfTI MRI volume.
        seg: Corresponding NIfTI segmentation mask.
        spacing: Target voxel spacing in millimeters, specified as
            (x, y, z). Defaults to 1 mm isotropic spacing.

    Returns:
        Tuple containing:
            - Resampled MRI volume.
            - Resampled segmentation mask.
    """
    image_r = resample_to_output(image, voxel_sizes=spacing, order=1)
    seg_r = resample_to_output(seg, voxel_sizes=spacing, order=0)
    return image_r, seg_r


def pad_to_shape(
    image: nib.Nifti1Image,
    seg: nib.Nifti1Image,
    target_shape: tuple[int, int, int],
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Zero-pad an MRI and its segmentation to a specified shape.

    Adds zero-valued voxels symmetrically around each spatial axis,
    keeping the existing image content approximately centered.

    The affine translation is adjusted so that the original voxels
    retain their physical coordinates after padding.

    This function is optional in the preprocessing pipeline.
    For patch-based training, global padding is generally unnecessary.

    Args:
        image: Input NIfTI MRI volume.
        seg: Corresponding NIfTI segmentation mask.
        target_shape: Required output dimensions (x, y, z) in voxels.
            Every dimension must be at least as large as the input.

    Returns:
        Tuple containing:
            - Zero-padded MRI volume.
            - Zero-padded segmentation mask.

    Raises:
        ValueError: If any input dimension exceeds the target shape.
    """
    shape = np.array(image.shape[:3])
    target = np.array(target_shape)
    if np.any(shape > target):
        raise ValueError(f"Volume shape {tuple(shape)} is larger than target {tuple(target)}")

    total = target - shape

    # Distribute padding between both sides of each spatial axis.
    before = total // 2
    after = total - before
    pad_width = list(zip(before, after))

    img_data = np.pad(image.get_fdata(dtype=np.float32), pad_width, constant_values=0)
    seg_data = np.pad(np.asarray(seg.dataobj), pad_width, constant_values=0)

    # Shift the origin to preserve the original voxel coordinates.
    affine = image.affine.copy()
    affine[:3, 3] -= affine[:3, :3] @ before

    return (
        nib.Nifti1Image(img_data, affine, image.header),
        nib.Nifti1Image(seg_data, affine, seg.header),
    )




def zscore_intensity(image: nib.Nifti1Image) -> nib.Nifti1Image:
    """Apply per-volume foreground Z-score intensity normalization.

    Computes the mean and standard deviation using only nonzero MRI
    voxels, then normalizes those voxels independently for each volume.

    Background voxels remain zero.

    Because statistics are calculated separately for each MRI,
    this operation does not use information from other subjects
    or from validation/test datasets.

    Args:
        image: Input NIfTI MRI volume.

    Returns:
        NIfTI MRI volume with float32 intensities, where foreground
        voxels are Z-score normalized and background remains zero.

        If the foreground has zero standard deviation, the output
        remains zero-valued.

    Raises:
        ValueError: If the MRI has no nonzero voxels.
    """
    data = image.get_fdata(dtype=np.float32)

    # Exclude zero-valued background from normalization statistics.
    foreground = data != 0

    if not np.any(foreground):
        raise ValueError("Cannot normalize an empty MRI.")

    mean = data[foreground].mean()
    std = data[foreground].std()

    normalized = np.zeros_like(data)

    if std > 0:
        normalized[foreground] = (data[foreground] - mean) / std

    header = image.header.copy()
    header.set_data_dtype(np.float32)

    return nib.Nifti1Image(normalized, image.affine, header)


def resample_and_crop(
    img_path: Path,
    seg_path: Path,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    margin: int = 5,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Apply orientation correction, cleaning, resampling, and cropping.

    Loads an MRI and its segmentation mask, then performs:
        1. Reorientation to RAS.
        2. Replacement of invalid MRI intensities.
        3. Resampling to the target voxel spacing.
        4. Foreground cropping with the specified margin.

    The same spatial operations are applied to both volumes.

    Args:
        img_path: Path to the input T2-weighted MRI.
        seg_path: Path to its corresponding segmentation mask.
        spacing: Target voxel spacing in millimeters.
            Defaults to (1.0, 1.0, 1.0).
        margin: Number of voxels retained around the foreground.
            Defaults to 5.

    Returns:
        Tuple containing:
            - Reoriented, cleaned, resampled, and cropped MRI.
            - Correspondingly transformed segmentation mask.
    """
    img = fill_nan(reorient_to_ras(nib.load(img_path)))
    seg = reorient_to_ras(nib.load(seg_path))
    img, seg = resample_to_spacing(img, seg, spacing=spacing)
    img, seg = crop_to_foreground(img, seg, margin=margin)
    return img, seg


def preprocess_subject(
    img_path: Path,
    seg_path: Path,
    target_shape: tuple[int, int, int] | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    margin: int = 5,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Run the complete preprocessing pipeline for one MRI/mask pair.

    Processing steps:
        1. Reorient MRI and segmentation to RAS.
        2. Replace invalid MRI intensity values.
        3. Resample both volumes to the target spacing.
        4. Crop to the MRI foreground with a margin.
        5. Apply optional zero-padding.
        6. Apply per-volume foreground Z-score normalization.

    Segmentation labels are preserved without remapping.
    For FeTA, the original class IDs range from 0 to 7.

    Args:
        img_path: Path to the input T2-weighted MRI.
        seg_path: Path to the corresponding segmentation mask.
        target_shape: Optional target dimensions for zero-padding.
            If None, no padding is applied and the cropped shape
            is preserved.
        spacing: Target voxel spacing in millimeters.
            Defaults to 1 mm isotropic spacing.
        margin: Number of voxels retained around the foreground.
            Defaults to 5.

    Returns:
        Tuple containing:
            - Preprocessed MRI with normalized float32 intensities.
            - Corresponding segmentation mask with original labels.
    """
    img, seg = resample_and_crop(img_path, seg_path, spacing=spacing, margin=margin)

    # Global padding is optional; patch padding can be handled during training.
    if target_shape is not None:
        img, seg = pad_to_shape(img, seg, target_shape=target_shape)

    img = zscore_intensity(img)

    return img, seg
