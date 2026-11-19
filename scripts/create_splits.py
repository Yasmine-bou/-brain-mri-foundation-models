
"""Create and save subject-level train/validation/test splits.

FeTA:
    Generate reproducible splits stratified by reconstruction method
    (MIAL and IRTK), using a fixed random seed.

BOBs:
    Preserve the dataset's predefined train, val, test_i, and test_ho
    splits.

Splits are saved as JSON files in data/splits/.
No MRI or segmentation files are moved or modified.

Usage (from the project root):
    python scripts/create_splits.py --dataset feta
    python scripts/create_splits.py --dataset bobs
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.splits import load_bobs_splits, split_feta, validate_splits


DATASETS = {
    "feta": PROJECT_ROOT / "data" / "processed" / "feta_fomo",
    "bobs": PROJECT_ROOT / "data" / "bobs" / "resampled75",
}


def main():
    """Create or load dataset splits and save them as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override the input dataset directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for FeTA splits (default: 42).",
    )

    args = parser.parse_args()
    data_dir = args.data_dir or DATASETS[args.dataset]

    # Create FeTA splits or read existing BOBs splits.
    if args.dataset == "feta":
        image_paths = sorted(
            data_dir.glob("sub-*/**/anat/*_T2w.nii.gz")
        )

        if not image_paths:
            raise FileNotFoundError(
                f"No FeTA MRI files found in {data_dir}"
            )

        splits = split_feta(image_paths, seed=args.seed)
    else:
        splits = load_bobs_splits(data_dir)

    # Check for duplicate subjects and overlap between splits.
    validate_splits(splits)

    # Save the split definitions.
    args.out_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.out_dir / f"{args.dataset}_splits.json"
    output_path.write_text(
        json.dumps(splits, indent=2),
        encoding="utf-8",
    )

    print(f"\nDataset: {args.dataset.upper()}")
    print(f"Source: {data_dir}")

    for name, subjects in splits.items():
        print(f"{name}: {len(subjects)} subjects")

    print(f"\nSaved splits to: {output_path}")


if __name__ == "__main__":
    main()
