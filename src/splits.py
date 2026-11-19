"""Utilities for creating and validating subject-level dataset splits.

FeTA:
    Create 60/20/20 splits stratified by gestational age.
    Check pathology and reconstruction balance.

BOBs:
    Create participant-level splits stratified by age at first scan.
    Keep all MRI sessions from the same participant in one split.
    Check sex and session-age balance.

Fixed seeds are used for reproducibility.
All splits are subject-independent to avoid data leakage.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def get_subject_id(path: Path) -> str:
    """Extract the subject ID from a BIDS-like file path."""
    for part in Path(path).parts:
        if part.startswith("sub-"):
            return part

    raise ValueError(f"No subject ID found in {path}")


def split_feta(
    image_paths: list[Path],
    metadata_path: Path,
    seed: int = 367,
) -> dict[str, list[str]]:
    """Create reproducible FeTA splits stratified by gestational age.

    Uses 60% training, 20% validation and 20% testing.

    Gestational age groups:
        20-24, 24-28, 28-32 and 32-36 weeks.

    Seed 367 was selected using metadata only, to balance
    age groups, pathology and reconstruction methods.

    Args:
        image_paths: Paths to the FeTA MRI files.
        metadata_path: Path to the FeTA participants.tsv file.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with train, val and test subject IDs.
    """
    # Load subject metadata.
    df = pd.read_csv(metadata_path, sep="\t")

    required = {"participant_id", "gestational_age", "pathology"}

    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing FeTA metadata values.")

    if df["participant_id"].duplicated().any():
        raise ValueError("Duplicate subjects in FeTA metadata.")

    # Check that metadata and MRI files contain the same subjects.
    image_subjects = [get_subject_id(path) for path in image_paths]

    if len(image_subjects) != len(set(image_subjects)):
        raise ValueError("Duplicate FeTA MRI subjects.")

    if set(image_subjects) != set(df["participant_id"]):
        raise ValueError("FeTA MRI subjects do not match metadata.")

    # Keep the original row order to reproduce the notebook split.
    df = df.copy()

    # Group subjects by gestational age.
    df["age_group"] = pd.cut(
        df["gestational_age"],
        bins=[20, 24, 28, 32, 36],
        labels=["20-24", "24-28", "28-32", "32-36"],
        right=False,
    )

    if df["age_group"].isna().any():
        raise ValueError("Gestational age outside expected range.")

    # First split: 60% train, 40% remaining.
    train_df, remaining_df = train_test_split(
        df,
        test_size=0.4,
        stratify=df["age_group"],
        random_state=seed,
    )

    # Second split: 20% validation, 20% test.
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=0.5,
        stratify=remaining_df["age_group"],
        random_state=seed,
    )

    subsets = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    if len(df) == 80:
        assert len(train_df) == 48
        assert len(val_df) == 16
        assert len(test_df) == 16

    # Check the known balance for our chosen FeTA split.
    if seed == 367 and len(df) == 80:
        expected_pathology = {"train": 29, "val": 10, "test": 10}
        expected_age_groups = {
            "train": [12, 19, 12, 5],
            "val": [4, 6, 4, 2],
            "test": [4, 6, 4, 2],
        }

        for name, subset in subsets.items():
            assert (
                (subset["pathology"] == "Pathological").sum()
                == expected_pathology[name]
            )
            assert (
                subset["age_group"].value_counts(sort=False).tolist()
                == expected_age_groups[name]
            )

        # Get reconstruction methods from the MRI filenames.
        reconstruction = {}

        for path in image_paths:
            subject = get_subject_id(path)
            filename = Path(path).name.lower()

            if "rec-mial" in filename:
                reconstruction[subject] = "MIAL"
            elif "rec-irtk" in filename:
                reconstruction[subject] = "IRTK"
            else:
                raise ValueError(f"Unknown reconstruction: {path}")

        expected_mial = {"train": 24, "val": 8, "test": 8}

        for name, subset in subsets.items():
            mial_count = sum(
                reconstruction[subject] == "MIAL"
                for subject in subset["participant_id"]
            )
            assert mial_count == expected_mial[name]

    # Save only subject IDs, as before.
    splits = {
        name: sorted(subset["participant_id"].tolist())
        for name, subset in subsets.items()
    }

    validate_splits(splits)

    return splits


def split_bobs(
    metadata_path: Path,
    seed: int = 559,
) -> dict[str, list[str]]:
    """Create reproducible BOBs train/val/test splits.

    Splits the 51 participants into 31 train, 10 val and 10 test.

    We stratify by age at the first MRI session.
    All sessions from the same participant stay in the same split.

    Seed 559 was selected using metadata only, to balance
    sex and age distributions across MRI sessions.

    Args:
        metadata_path: Path to BOBs participants.csv.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with train, val and test participant IDs.
    """
    df = pd.read_csv(metadata_path)

    required = {"participant_id", "session_id", "age", "sex"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing BOBs columns: {missing}")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing BOBs metadata values.")

    if df.duplicated(subset=["participant_id", "session_id"]).any():
        raise ValueError("Duplicate BOBs sessions.")

    # One row per participant, based on the earliest MRI session.
    # Keep the same ordering as in our notebook.
    participants = (
        df.sort_values("age")
        .drop_duplicates(subset="participant_id")
        .copy()
    )

    # Create age groups (months).
    participants["age_group"] = pd.cut(
        participants["age"],
        bins=[1, 4, 7, 10],
        labels=["1-4", "4-7", "7-10"],
        right=False,
    )

    if participants["age_group"].isna().any():
        raise ValueError("Age outside expected range.")

    if len(participants) != 51:
        raise ValueError(
            f"Expected 51 participants, found {len(participants)}."
        )

    # First split: 31 train, 20 remaining.
    train_df, remaining_df = train_test_split(
        participants,
        train_size=31,
        stratify=participants["age_group"],
        random_state=seed,
    )

    # Second split: 10 validation, 10 test.
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=10,
        stratify=remaining_df["age_group"],
        random_state=seed,
    )

    subsets = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    assert len(train_df) == 31
    assert len(val_df) == 10
    assert len(test_df) == 10
    
    # Check the balance found during our investigation.
    if seed == 559 and len(participants) == 51:
        expected_female = {"train": 19, "val": 6, "test": 6}
        expected_sessions = {"train": 42, "val": 15, "test": 14}

        for name, subset in subsets.items():
            female_count = (subset["sex"] == "Female").sum()
            assert female_count == expected_female[name]

            session_count = df["participant_id"].isin(
                subset["participant_id"]
            ).sum()
            assert session_count == expected_sessions[name]

    # Save participant IDs, not individual sessions.
    splits = {
        name: sorted(subset["participant_id"].tolist())
        for name, subset in subsets.items()
    }

    validate_splits(splits)

    # Make sure all 71 MRI sessions are assigned.
    all_ids = set().union(*(set(ids) for ids in splits.values()))

    assert all_ids == set(df["participant_id"])
    assert len(df) == 71

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
