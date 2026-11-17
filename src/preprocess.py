from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output


def find_image_label_pairs(
    dataset_dir: Path,
    seg_suffix: str = "_dseg",
) -> list[tuple[Path, Path]]:
    """Return sorted (T2w image, segmentation) path pairs for a dataset.
    Raises FileNotFoundError if an image has no matching segmentation.
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
    """Reorient a NIfTI image to RAS (Right, Anterior, Superior).

    Only flips the voxel axes and updates the affine accordingly: voxel values are unchanged.
    """
    return nib.as_closest_canonical(image)


def fill_nan(image: nib.Nifti1Image, fill_value: float = 0.0) -> nib.Nifti1Image:
    """Fill NaN values in a NIfTI image with a specified value.
    """
    data = image.get_fdata(dtype=np.float32)
    data_filled = np.nan_to_num(data, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return nib.Nifti1Image(data_filled, image.affine, image.header)

def crop_to_foreground(
    image: nib.Nifti1Image,
    seg: nib.Nifti1Image,
    margin: int = 5,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Crop image and mask to the bounding box of the image's non-zero voxels.

    `margin` adds extra voxels on each side, clipped to the volume edges.
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
    spacing: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Resample image and mask to a common voxel spacing (in mm).

    - image: linear interpolation (order=1), smooth intensities.
    - mask: nearest-neighbour (order=0), so every voxel keeps a valid label
    Both get the same output grid, so they stay aligned.
    """
    image_r = resample_to_output(image, voxel_sizes=spacing, order=1)
    seg_r = resample_to_output(seg, voxel_sizes=spacing, order=0)
    return image_r, seg_r


def pad_to_shape(
    image: nib.Nifti1Image,
    seg: nib.Nifti1Image,
    target_shape: tuple[int, int, int],
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Pad image and mask with zeros (background) to `target_shape`.

    Padding is split evenly on both sides of each axis, so the brain stays
    centred. The affine is shifted so every voxel keeps its position in mm.
    """
    shape = np.array(image.shape[:3])
    target = np.array(target_shape)
    if np.any(shape > target):
        raise ValueError(f"Volume shape {tuple(shape)} is larger than target {tuple(target)}")

    total = target - shape
    before = total // 2            # e.g. 7 extra voxels -> 3 before, 4 after
    after = total - before
    pad_width = list(zip(before, after))

    img_data = np.pad(image.get_fdata(dtype=np.float32), pad_width, constant_values=0)
    seg_data = np.pad(np.asarray(seg.dataobj), pad_width, constant_values=0)

    # The old voxel [0,0,0] is now at index `before`, so the new voxel [0,0,0]
    # sits `before` voxels earlier in space: move the origin back by that much.
    affine = image.affine.copy()
    affine[:3, 3] -= affine[:3, :3] @ before

    return (
        nib.Nifti1Image(img_data, affine, image.header),
        nib.Nifti1Image(seg_data, affine, seg.header),
    )


def scale_intensity(image: nib.Nifti1Image, minv: float = 0.0, maxv: float = 1.0) -> nib.Nifti1Image:
    """Min-max rescale the whole volume to [minv, maxv]. Use on images only.
    """
    data = image.get_fdata(dtype=np.float32)
    lo, hi = data.min(), data.max()
    if hi > lo:
        data = (data - lo) / (hi - lo) * (maxv - minv) + minv
    else:
        data = np.full_like(data, minv)    # constant image: nothing to scale

    header = image.header.copy()
    header.set_data_dtype(np.float32)      # some FeTA files are int16: keep decimals when saving
    return nib.Nifti1Image(data, image.affine, header)


def resample_and_crop(
    img_path: Path,
    seg_path: Path,
    spacing: tuple[float, float, float] = (0.5, 0.5, 0.5),
    margin: int = 5,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Steps 2-5: reorient to RAS, fill NaN, resample, crop to foreground."""
    img = fill_nan(reorient_to_ras(nib.load(img_path)))
    seg = reorient_to_ras(nib.load(seg_path))
    img, seg = resample_to_spacing(img, seg, spacing=spacing)
    img, seg = crop_to_foreground(img, seg, margin=margin)
    return img, seg


def preprocess_subject(
    img_path: Path,
    seg_path: Path,
    target_shape: tuple[int, int, int],
    spacing: tuple[float, float, float] = (0.5, 0.5, 0.5),
    margin: int = 5,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Full pipeline for one image/mask pair (steps 2-7).

    2. reorient to RAS   3. fill NaN (image)   4. resample   5. crop
    6. pad to target_shape   7. scale intensities to [0, 1] (image)
    8. remap labels: TODO, waiting for the label mapping
    """
    img, seg = resample_and_crop(img_path, seg_path, spacing=spacing, margin=margin)
    img, seg = pad_to_shape(img, seg, target_shape=target_shape)
    img = scale_intensity(img)
    # seg = remap_labels(seg, mapping)   # step 8, TODO
    return img, seg
