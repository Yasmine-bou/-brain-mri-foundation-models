"""Preprocess a whole dataset (FeTA or BOBs) and save the results.

Runs src.preprocess.preprocess_subject on every image/mask pair and writes
the outputs to data/processed/<dataset>/, keeping the BIDS folder layout
and filenames. Also writes config.json with the settings used.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocess import (  # noqa: E402
    find_image_label_pairs,
    preprocess_subject,
    resample_and_crop,
)

DATASETS = {
    "feta": {"dir": PROJECT_ROOT / "data" / "FeTA" / "feta_2.4", "seg_suffix": "_dseg"},
    "bobs": {"dir": PROJECT_ROOT / "data" / "bobs" / "resampled75", "seg_suffix": "_desc-bobs9_dseg"},
}


def measure_target_shape(pairs, spacing, margin, multiple=16):
    """Largest shape after resample + crop, rounded up to a multiple of 16."""
    shapes = []
    t0 = time.time()
    for k, (img_path, seg_path) in enumerate(pairs, start=1):
        img, _ = resample_and_crop(img_path, seg_path, spacing=spacing, margin=margin)
        shapes.append(img.shape)
        print(f"  measuring {k}/{len(pairs)} {img_path.name} {img.shape} ({time.time() - t0:.0f}s)")
    max_shape = np.array(shapes).max(axis=0)
    target = tuple(int(n) for n in np.ceil(max_shape / multiple) * multiple)
    print(f"max shape per axis: {tuple(int(n) for n in max_shape)} -> target shape: {target}")
    return target, tuple(int(n) for n in max_shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASETS.keys())
    parser.add_argument("--data-dir", type=Path, help="override the dataset folder")
    parser.add_argument("--out-dir", type=Path, help="default: data/processed/<dataset>")
    parser.add_argument("--spacing", type=float, nargs=3, default=[0.5, 0.5, 0.5], help="voxel size in mm")
    parser.add_argument("--margin", type=int, default=5, help="voxels kept around the brain when cropping")
    parser.add_argument("--target-shape", type=int, nargs=3, help="pad shape; measured from the data if omitted")
    parser.add_argument("--limit", type=int, help="only process the first N subjects (for testing)")
    args = parser.parse_args()

    data_dir = args.data_dir or DATASETS[args.dataset]["dir"]
    out_dir = args.out_dir or PROJECT_ROOT / "data" / "processed" / args.dataset
    spacing = tuple(args.spacing)

    pairs = find_image_label_pairs(data_dir, seg_suffix=DATASETS[args.dataset]["seg_suffix"])
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{args.dataset}: {len(pairs)} pairs in {data_dir}")

    max_shape = None
    if args.target_shape:
        target_shape = tuple(args.target_shape)
    else:
        print("No --target-shape given, measuring it (slow)...")
        target_shape, max_shape = measure_target_shape(pairs, spacing, args.margin)

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, (img_path, seg_path) in enumerate(pairs, start=1):
        img, seg = preprocess_subject(
            img_path, seg_path, target_shape=target_shape, spacing=spacing, margin=args.margin
        )
        # keep the BIDS layout: sub-XXX/(ses-YYY/)anat/<same filename>
        img_out = out_dir / img_path.relative_to(data_dir)
        seg_out = out_dir / seg_path.relative_to(data_dir)
        img_out.parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, img_out)
        nib.save(seg, seg_out)
        print(f"  {k}/{len(pairs)} {img_path.name} -> {img.shape} ({time.time() - t0:.0f}s)")

    config = {
        "dataset": args.dataset,
        "source_dir": str(data_dir.relative_to(PROJECT_ROOT) if data_dir.is_relative_to(PROJECT_ROOT) else data_dir),
        "n_subjects": len(pairs),
        "steps": [
            "reorient to RAS",
            "fill NaN/inf with 0 (image)",
            "resample (image: linear, mask: nearest)",
            "crop to image foreground",
            "pad with zeros (centred)",
            "min-max scale to [0, 1] (image)",
        ],
        "spacing_mm": list(spacing),
        "crop_margin_voxels": args.margin,
        "target_shape": list(target_shape),
        "measured_max_shape": list(max_shape) if max_shape else None,
        "label_remapping": None,  # TODO: step 8
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Done: {len(pairs)} subjects saved to {out_dir}")


if __name__ == "__main__":
    main()
