"""Create and validate reproducible FeTA and BOBs dataset splits.

FeTA:
    Split 80 fetal MRI subjects into 60% train, 20% validation,
    and 20% test, stratified by gestational age.

BOBs:
    Split 51 infant participants into 31 train, 10 validation,
    and 10 test participants, stratified by age at first scan.
    All sessions from the same participant remain in one split.

The fixed seeds were selected using dataset metadata to balance
relevant characteristics without using model performance.
Participant-level separation prevents data leakage.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def get_subject_id(path: Path) -> str:
    """Extract a subject identifier from a BIDS-style file path.

    Args:
        path: Path containing a directory named 'sub-*'.

    Returns:
        Subject identifier, including the 'sub-' prefix.

    Raises:
        ValueError: If no subject identifier is found.
    """
    for part in Path(path).parts:
        if part.startswith("sub-"):
            return part

    raise ValueError(f"No subject ID found in {path}")


def split_feta(
    image_paths: list[Path],
    metadata_path: Path,
    seed: int = 367,
) -> dict[str, list[str]]:
    """Create stratified train/validation/test splits for FeTA.

    Uses a 60/20/20 subject-level split, stratified by gestational
    age groups: [20,24), [24,28), [28,32), and [32,36) weeks.

    Seed 367 was selected using metadata to balance gestational age,
    pathology, and MRI reconstruction methods.

    Args:
        image_paths: Paths to the FeTA MRI files.
        metadata_path: Path to FeTA participants.tsv.
        seed: Random seed for splitting (default: 367).

    Returns:
        Dictionary mapping 'train', 'val', and 'test' to sorted
        subject identifiers.

    Raises:
        ValueError: If metadata is missing, invalid, or inconsistent
            with the available MRI files.
    """
    df = pd.read_csv(metadata_path, sep="\t")

    required = {"participant_id", "gestational_age", "pathology"}

    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing FeTA metadata values.")

    if df["participant_id"].duplicated().any():
        raise ValueError("Duplicate subjects in FeTA metadata.")

    # Verify that MRI files and metadata refer to the same subjects.
    image_subjects = [get_subject_id(path) for path in image_paths]

    if len(image_subjects) != len(set(image_subjects)):
        raise ValueError("Duplicate FeTA MRI subjects.")

    if set(image_subjects) != set(df["participant_id"]):
        raise ValueError("FeTA MRI subjects do not match metadata.")

    # Preserve metadata row order for reproducibility.
    df = df.copy()

    df["age_group"] = pd.cut(
        df["gestational_age"],
        bins=[20, 24, 28, 32, 36],
        labels=["20-24", "24-28", "28-32", "32-36"],
        right=False,
    )

    if df["age_group"].isna().any():
        raise ValueError("Gestational age outside expected range.")

    # Split subjects into 60% train, then divide the remaining 40%.
    train_df, remaining_df = train_test_split(
        df,
        test_size=0.4,
        stratify=df["age_group"],
        random_state=seed,
    )

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

    # Verify the expected metadata balance for the selected split.
    if seed == 367 and len(df) == 80:
        expected_pathology = {"train": 29, "val": 10, "test": 10}
        expected_age_groups = {
            "train": [12, 19, 12, 5],
            "val": [4, 6, 4, 2],
            "test": [4, 6, 4, 2],
        }

        for name, subset in subsets.items():
            assert (subset["pathology"] == "Pathological").sum() == expected_pathology[
                name
            ]
            assert (
                subset["age_group"].value_counts(sort=False).tolist()
                == expected_age_groups[name]
            )

        # Reconstruction type is encoded in the MRI filename.
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
    """Create participant-level train/validation/test splits for BOBs.

    Stratifies the 51 participants by age at their earliest scan,
    using age groups [1,4), [4,7), and [7,10) months.

    Assigns 31 participants to training, 10 to validation, and
    10 to testing. Multiple scans from one participant always
    belong to the same split.

    Seed 559 was selected using metadata to balance sex and age
    distributions across MRI sessions.

    Args:
        metadata_path: Path to BOBs participants.csv.
        seed: Random seed for splitting (default: 559).

    Returns:
        Dictionary mapping 'train', 'val', and 'test' to sorted
        participant identifiers.

    Raises:
        ValueError: If metadata is incomplete, contains duplicate
            sessions, or has unexpected participant counts.
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

    # Use the earliest MRI session to stratify each participant.
    participants = df.sort_values("age").drop_duplicates(subset="participant_id").copy()

    participants["age_group"] = pd.cut(
        participants["age"],
        bins=[1, 4, 7, 10],
        labels=["1-4", "4-7", "7-10"],
        right=False,
    )

    if participants["age_group"].isna().any():
        raise ValueError("Age outside expected range.")

    if len(participants) != 51:
        raise ValueError(f"Expected 51 participants, found {len(participants)}.")

    # First allocate 31 training participants, then split the rest equally.
    train_df, remaining_df = train_test_split(
        participants,
        train_size=31,
        stratify=participants["age_group"],
        random_state=seed,
    )

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

    # Verify sex balance and scan counts for the selected split.
    if seed == 559 and len(participants) == 51:
        expected_female = {"train": 19, "val": 6, "test": 6}
        expected_sessions = {"train": 42, "val": 15, "test": 14}

        for name, subset in subsets.items():
            female_count = (subset["sex"] == "Female").sum()
            assert female_count == expected_female[name]

            session_count = df["participant_id"].isin(subset["participant_id"]).sum()
            assert session_count == expected_sessions[name]

    # Store participant IDs, not individual MRI sessions.
    splits = {
        name: sorted(subset["participant_id"].tolist())
        for name, subset in subsets.items()
    }

    validate_splits(splits)

    all_ids = set().union(*(set(ids) for ids in splits.values()))

    assert all_ids == set(df["participant_id"])
    assert len(df) == 71

    return splits


def validate_splits(splits: dict[str, list[str]]) -> None:
    """Validate that train, validation, and test participants do not overlap.

    Args:
        splits: Dictionary mapping split names to participant ID lists.

    Raises:
        ValueError: If participant IDs are duplicated within a split
            or shared across splits.
    """
    seen = set()

    for split_name, subjects in splits.items():
        if len(subjects) != len(set(subjects)):
            raise ValueError(f"Duplicate participants in {split_name}")

        overlap = seen.intersection(subjects)

        if overlap:
            raise ValueError(f"Participant overlap in {split_name}: {sorted(overlap)}")

        seen.update(subjects)
