"""
Create reproducible subject-level train/validation/test dataset splits.

This script generates and saves predefined splits for the FeTA and
Baby Open Brains (BOBs) datasets.

FeTA:
    - Create reproducible 60/20/20 subject-level splits.
    - Stratify by gestational age.
    - Use seed 367 by default.
    - The selected seed balances age, pathology, and reconstruction
      methods according to the project's split-generation procedure.

BOBs:
    - Create participant-level splits using age at first MRI.
    - Use seed 559 by default.
    - The selected seed balances sex and age distributions across
      MRI sessions according to the project's split-generation procedure.
    - Keep all sessions from the same participant in the same split.

Output:
    data/splits/feta_splits.json
    data/splits/bobs_splits.json

The split-generation algorithms are implemented in src.datasets.splits.

No MRI or segmentation files are moved or modified.

Usage (from the project root):
    python scripts/preprocessing/create_splits.py --dataset feta
    python scripts/preprocessing/create_splits.py --dataset bobs

Optional arguments:
    --data-dir        Override the input dataset directory.
    --out-dir         Override the output split directory.
    --seed            Override the default random seed.
    --metadata-path   Override the metadata file location.

Note:
    Existing JSON files at the selected output path are overwritten.
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# Project paths and dataset configuration
# ---------------------------------------------------------------------

# Locate the repository root and enable imports from src/.
# Note: this expression assumes the script's original location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import dataset-specific splitting and validation functions.
from src.datasets.splits import split_bobs, split_feta, validate_splits  # noqa: E402


# Default directories used to locate the source datasets.
DATASETS = {
    "feta": PROJECT_ROOT / "data" / "processed" / "feta_fomo",
    "bobs": PROJECT_ROOT / "data" / "bobs" / "resampled75",
}

# FeTA participant metadata, including information used for stratification.
FETA_METADATA = PROJECT_ROOT / "data" / "FeTA" / "feta_2.4" / "participants.tsv"

# BOBs participant metadata used for participant-level splitting.
BOBS_METADATA = PROJECT_ROOT / "data" / "bobs" / "resampled75" / "participants.csv"


def main():
    """
    Create dataset splits and save them as JSON.

    The selected dataset determines:
        - The default source directory.
        - The metadata file.
        - The splitting function.
        - The default random seed.

    All generated splits are checked for participant overlap
    before being written to disk.

    Raises
    ------
    FileNotFoundError
        If the required MRI files or metadata are unavailable.

    Notes
    -----
    The actual stratification logic is implemented in
    src.datasets.splits, not in this command-line script.
    """

    # -------------------------------------------------------------
    # 1. Parse command-line arguments
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(description=__doc__)

    # Select one of the supported developmental MRI datasets.
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
    )

    # Optionally specify a different source dataset directory.
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override the input dataset directory.",
    )

    # Output directory for reproducible JSON split definitions.
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits",
    )

    # Optional random seed override.
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the dataset seed (FeTA: 367, BOBs: 559).",
    )

    # Optional path to the dataset-specific participant metadata.
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Override the dataset metadata file.",
    )

    args = parser.parse_args()

    # Use the supplied data directory or the configured default.
    data_dir = args.data_dir or DATASETS[args.dataset]

    # -------------------------------------------------------------
    # 2. Generate FeTA splits
    # -------------------------------------------------------------

    # Create FeTA splits.
    if args.dataset == "feta":
        # Use the predefined FeTA seed unless explicitly overridden.
        seed = 367 if args.seed is None else args.seed

        # Locate participant metadata for stratified splitting.
        metadata_path = args.metadata_path or FETA_METADATA

        # Discover available FeTA T2-weighted MRI volumes.
        image_paths = sorted(data_dir.glob("sub-*/**/anat/*_T2w.nii.gz"))

        # Require at least one valid source MRI.
        if not image_paths:
            raise FileNotFoundError(f"No FeTA MRI files found in {data_dir}")

        # Ensure the stratification metadata is available.
        if not metadata_path.is_file():
            raise FileNotFoundError(f"FeTA metadata not found: {metadata_path}")

        # Generate subject-level splits using the FeTA splitting utility.
        splits = split_feta(
            image_paths=image_paths,
            metadata_path=metadata_path,
            seed=seed,
        )

    # -------------------------------------------------------------
    # 3. Generate BOBs splits
    # -------------------------------------------------------------

    # Create BOBs splits.
    else:
        # Use the predefined BOBs seed unless explicitly overridden.
        seed = 559 if args.seed is None else args.seed

        # Select the default or user-specified participant metadata.
        # When a custom data directory is supplied, its participants.csv
        # file is used unless --metadata-path overrides it.
        metadata_path = args.metadata_path or (
            data_dir / "participants.csv"
            if args.data_dir is not None
            else BOBS_METADATA
        )

        # Verify that the participant metadata exists.
        if not metadata_path.is_file():
            raise FileNotFoundError(f"BOBs metadata not found: {metadata_path}")

        # Generate participant-level splits from BOBs metadata.
        splits = split_bobs(
            metadata_path=metadata_path,
            seed=seed,
        )

    # -------------------------------------------------------------
    # 4. Validate split independence
    # -------------------------------------------------------------

    # Check that no participant appears in multiple splits.
    # This prevents participant-level data leakage during evaluation.
    validate_splits(splits)

    # -------------------------------------------------------------
    # 5. Save the generated splits
    # -------------------------------------------------------------

    # Create the output directory if necessary.
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Construct the dataset-specific output JSON filename.
    output_path = args.out_dir / f"{args.dataset}_splits.json"

    # Write the generated splits in a human-readable JSON format.
    output_path.write_text(
        json.dumps(splits, indent=2),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # 6. Display split-generation summary
    # -------------------------------------------------------------

    print(f"\nDataset: {args.dataset.upper()}")
    print(f"Seed: {seed}")
    print(f"Metadata: {metadata_path}")

    # Report the number of identifiers assigned to each split.
    for name, subjects in splits.items():
        print(f"{name}: {len(subjects)} participants")

    print(f"\nSaved splits to: {output_path}")


if __name__ == "__main__":
    main()
