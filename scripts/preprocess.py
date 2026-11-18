
"""Run MRI preprocessing on an entire FeTA or BOBs dataset.

This script applies the preprocessing pipeline defined in src/preprocess.py
to every T2-weighted MRI and its corresponding segmentation mask.

The pipeline includes:
    1. Reorientation to RAS.
    2. Replacement of invalid MRI intensities (NaN and Inf).
    3. Resampling to the specified voxel spacing.
    4. Foreground cropping with a configurable margin.
    5. Optional zero-padding to a target shape.
    6. Per-volume foreground Z-score intensity normalization.

Processed files are saved under data/processed/<dataset>/ by default,
preserving the original BIDS-like folder structure and filenames.
Each volume is saved twice: as NIfTI (.nii.gz, keeps the affine, for
viewing and checks) and as NumPy (.npy, plain array for fast loading
during training; image float32, mask uint8).

A config.json file records the preprocessing parameters for
reproducibility.

For FeTA + FOMO25, the default voxel spacing is 1 mm isotropic,
and global padding is disabled. Patch extraction will be handled
separately during training.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np


# Resolve the project root so the script can import from src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocess import (  # noqa: E402
    find_image_label_pairs,
    preprocess_subject,
)


# Dataset locations and segmentation filename conventions.
DATASETS = {
    "feta": {
        "dir": PROJECT_ROOT / "data" / "FeTA" / "feta_2.4",
        "seg_suffix": "_dseg",
    },
    "bobs": {
        "dir": PROJECT_ROOT / "data" / "bobs" / "resampled75",
        "seg_suffix": "_desc-bobs9_dseg",
    },
}


def npy_path(nifti_path: Path) -> Path:
    """Return the .npy path next to a .nii.gz file (x_T2w.nii.gz -> x_T2w.npy)."""
    return nifti_path.with_suffix("").with_suffix(".npy")


def main():
    """Preprocess a dataset and save the resulting MRI/mask pairs.

    Parses command-line arguments, identifies matching MRI and
    segmentation files, and applies preprocess_subject() to each pair.

    Command-line arguments:
        --dataset:
            Dataset to preprocess: "feta" or "bobs" (required).

        --data-dir:
            Optional override for the input dataset directory.

        --out-dir:
            Optional override for the output directory.

        --spacing:
            Target voxel spacing in millimeters.
            Defaults to (1.0, 1.0, 1.0).

        --margin:
            Number of voxels kept around the foreground during cropping.
            Defaults to 5.

        --target-shape:
            Optional dimensions for zero-padding.
            If omitted, no global padding is applied.

        --limit:
            Optional maximum number of image/mask pairs to process,
            useful for testing.

    Outputs:
        Preprocessed NIfTI images and segmentation masks, preserving
        their relative directory structure.

        The same volumes as NumPy arrays (.npy) next to each NIfTI file:
        the image as float32 and the mask as uint8.

        config.json containing the dataset, preprocessing steps,
        and parameters used.
    """

    # ---------------------------------------------------------
    # 1. Parse command-line arguments
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
        help="Dataset to preprocess.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override the dataset directory.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Override the output directory.",
    )

    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Target voxel spacing in mm (default: 1 mm isotropic).",
    )

    parser.add_argument(
        "--margin",
        type=int,
        default=5,
        help="Foreground cropping margin in voxels (default: 5).",
    )

    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        help="Optional target shape for zero-padding.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N image/mask pairs for testing.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 2. Resolve paths and preprocessing parameters
    # ---------------------------------------------------------
    data_dir = args.data_dir or DATASETS[args.dataset]["dir"]

    out_dir = (
        args.out_dir
        or PROJECT_ROOT / "data" / "processed" / args.dataset
    )

    spacing = tuple(args.spacing)

    # No global padding unless explicitly requested.
    target_shape = (
        tuple(args.target_shape)
        if args.target_shape is not None
        else None
    )

    # ---------------------------------------------------------
    # 3. Find MRI and segmentation pairs
    # ---------------------------------------------------------
    pairs = find_image_label_pairs(
        data_dir,
        seg_suffix=DATASETS[args.dataset]["seg_suffix"],
    )

    if args.limit is not None:
        pairs = pairs[:args.limit]

    if not pairs:
        raise ValueError(f"No MRI/segmentation pairs found in {data_dir}")

    print(f"{args.dataset}: {len(pairs)} pairs in {data_dir}")

    # ---------------------------------------------------------
    # 4. Preprocess and save each MRI/segmentation pair
    # ---------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    for k, (img_path, seg_path) in enumerate(pairs, start=1):

        # Apply the same spatial preprocessing to the MRI and mask.
        img, seg = preprocess_subject(
            img_path,
            seg_path,
            target_shape=target_shape,
            spacing=spacing,
            margin=args.margin,
        )

        # Preserve the original BIDS-like directory structure.
        img_out = out_dir / img_path.relative_to(data_dir)
        seg_out = out_dir / seg_path.relative_to(data_dir)

        img_out.parent.mkdir(parents=True, exist_ok=True)

        # NIfTI: keeps the affine, can be opened in a viewer.
        nib.save(img, img_out)
        nib.save(seg, seg_out)

        # NumPy: plain arrays for fast loading during training.
        np.save(npy_path(img_out), img.get_fdata(dtype=np.float32))
        np.save(npy_path(seg_out), np.asarray(seg.dataobj).astype(np.uint8))

        print(
            f"  {k}/{len(pairs)} {img_path.name} "
            f"-> {img.shape} ({time.time() - t0:.0f}s)"
        )

    # ---------------------------------------------------------
    # 5. Save preprocessing configuration for reproducibility
    # ---------------------------------------------------------
    config = {
        "dataset": args.dataset,
        "source_dir": str(
            data_dir.relative_to(PROJECT_ROOT)
            if data_dir.is_relative_to(PROJECT_ROOT)
            else data_dir
        ),
        "n_subjects": len(pairs),
        "steps": [
            "reorient to RAS",
            "fill NaN/inf with 0 (image)",
            "resample (image: linear, mask: nearest)",
            "crop to image foreground",
            "optional zero-padding (if target_shape is specified)",
            "per-volume foreground Z-score normalization (image)",
            "save as NIfTI and NumPy (.npy: image float32, mask uint8)",
        ],
        "spacing_mm": list(spacing),
        "crop_margin_voxels": args.margin,
        "target_shape": (
            list(target_shape)
            if target_shape is not None
            else None
        ),
        "label_remapping": None,
    }

    config_path = out_dir / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"Done: {len(pairs)} pairs saved to {out_dir}")


if __name__ == "__main__":
    main()
