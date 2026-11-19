
"""Create and save subject-level train/validation/test splits.

FeTA:
    Create reproducible 60/20/20 splits stratified by gestational age.
    Seed 367 balances age, pathology and reconstruction methods.

BOBs:
    Create participant-level splits stratified by age at first MRI.
    Seed 559 balances sex and age distributions across MRI sessions.
    All sessions from the same participant stay in the same split.

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

from src.splits import split_bobs, split_feta, validate_splits


DATASETS = {
    "feta": PROJECT_ROOT / "data" / "processed" / "feta_fomo",
    "bobs": PROJECT_ROOT / "data" / "bobs" / "resampled75",
}

FETA_METADATA = (
    PROJECT_ROOT / "data" / "FeTA" / "feta_2.4" / "participants.tsv"
)

BOBS_METADATA = (
    PROJECT_ROOT / "data" / "bobs" / "resampled75" / "participants.csv"
)


def main():
    """Create dataset splits and save them as JSON."""
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
        default=None,
        help="Override the dataset seed (FeTA: 367, BOBs: 559).",
    )

    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Override the dataset metadata file.",
    )

    args = parser.parse_args()

    data_dir = args.data_dir or DATASETS[args.dataset]

    # Create FeTA splits.
    if args.dataset == "feta":
        seed = 367 if args.seed is None else args.seed
        metadata_path = args.metadata_path or FETA_METADATA

        image_paths = sorted(
            data_dir.glob("sub-*/**/anat/*_T2w.nii.gz")
        )

        if not image_paths:
            raise FileNotFoundError(
                f"No FeTA MRI files found in {data_dir}"
            )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"FeTA metadata not found: {metadata_path}"
            )

        splits = split_feta(
            image_paths=image_paths,
            metadata_path=metadata_path,
            seed=seed,
        )

    # Create BOBs splits.
    else:
        seed = 559 if args.seed is None else args.seed

        metadata_path = (
            args.metadata_path
            or (
                data_dir / "participants.csv"
                if args.data_dir is not None
                else BOBS_METADATA
            )
        )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"BOBs metadata not found: {metadata_path}"
            )

        splits = split_bobs(
            metadata_path=metadata_path,
            seed=seed,
        )

    # Check that no participant appears in multiple splits.
    validate_splits(splits)

    # Save splits as JSON.
    args.out_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.out_dir / f"{args.dataset}_splits.json"

    output_path.write_text(
        json.dumps(splits, indent=2),
        encoding="utf-8",
    )

    print(f"\nDataset: {args.dataset.upper()}")
    print(f"Seed: {seed}")
    print(f"Metadata: {metadata_path}")

    for name, subjects in splits.items():
        print(f"{name}: {len(subjects)} participants")

    print(f"\nSaved splits to: {output_path}")


if __name__ == "__main__":
    main()
