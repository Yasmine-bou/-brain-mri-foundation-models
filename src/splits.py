
"""Utilities for creating and validating subject-level dataset splits.

FeTA:
    Create reproducible train/validation/test splits, stratified by
    reconstruction method (MIAL and IRTK).

BOBs:
    Read the existing dataset splits without modifying them.

All splitting is done at subject level to prevent data leakage
between training, validation, and test sets.
"""

import random
from collections import defaultdict
from pathlib import Path


def get_subject_id(path: Path) -> str:
    """Extract the subject ID from a BIDS-like file path."""
    for part in Path(path).parts:
        if part.startswith("sub-"):
            return part

    raise ValueError(f"No subject ID found in {path}")


def split_feta(
    image_paths: list[Path],
    seed: int = 42,
) -> dict[str, list[str]]:
    """Create reproducible FeTA splits stratified by reconstruction method.

    Uses 70% training, 15% validation and 15% testing within each
    reconstruction group (MIAL and IRTK).

    For 40 subjects per group, this gives:
        Train: 28 MIAL + 28 IRTK = 56
        Val:    6 MIAL +  6 IRTK = 12
        Test:   6 MIAL +  6 IRTK = 12

    Args:
        image_paths: Paths to the original or preprocessed FeTA MRIs.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with "train", "val" and "test" subject ID lists.
    """
    groups = defaultdict(list)

    for path in image_paths:
        subject = get_subject_id(path)
        name = Path(path).name.lower()

        if "mial" in name:
            groups["mial"].append(subject)
        elif "irtk" in name:
            groups["irtk"].append(subject)
        else:
            raise ValueError(f"Unknown reconstruction method: {path}")

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for method in ("mial", "irtk"):
        subjects = sorted(groups[method])

        if len(subjects) != len(set(subjects)):
            raise ValueError(f"Duplicate subjects in {method}")

        rng.shuffle(subjects)

        n_train = int(0.70 * len(subjects))
        n_val = int(0.15 * len(subjects))

        splits["train"].extend(subjects[:n_train])
        splits["val"].extend(subjects[n_train:n_train + n_val])
        splits["test"].extend(subjects[n_train + n_val:])

    for split in splits:
        splits[split] = sorted(splits[split])

    validate_splits(splits)

    return splits



def load_bobs_splits(
    dataset_dir: Path,
) -> dict[str, list[str]]:
    """Load predefined BOBs splits from the dataset CSV.

    The CSV contains one row per MRI session. Subjects may have
    multiple sessions, so we validate splits at participant level.

    Args:
        dataset_dir: Root of the BOBs resampled75 dataset.

    Returns:
        Dictionary mapping split names to participant IDs.
    """
    import pandas as pd

    csv_path = Path(dataset_dir) / "splits_40_10_15_35.csv"
    df = pd.read_csv(csv_path)

    splits = {}

    for split in ("train", "val", "test_i", "test_ho"):
        subjects = df.loc[
            df["splits"] == split, "participant_id"
        ].unique().tolist()

        splits[split] = sorted(subjects)

    validate_splits(splits)

    return splits


def validate_splits(splits: dict[str, list[str]]) -> None:
    """Check that participants are unique and do not overlap across splits."""
    seen = set()

    for split_name, subjects in splits.items():
        if len(subjects) != len(set(subjects)):
            raise ValueError(f"Duplicate participants in {split_name}")

        overlap = seen.intersection(subjects)

        if overlap:
            raise ValueError(
                f"Participant overlap in {split_name}: {sorted(overlap)}"
            )

        seen.update(subjects)
