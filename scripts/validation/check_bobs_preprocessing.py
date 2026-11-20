"""
Validate SSL3D/nnU-Net preprocessing outputs for the BOBs dataset.

This script performs structural and numerical quality-control checks
on the preprocessed Baby Open Brains (BOBs) MRI volumes and their
corresponding segmentation masks.

The validation compares preprocessed cases against the converted
nnU-Net raw dataset to ensure that every expected case is present.

Main validation steps:
    1. Locate the raw and preprocessed BOBs datasets.
    2. Verify the existence of dataset metadata and SSL3D plans.
    3. Check the expected number of cases in each split.
    4. Compare raw and preprocessed MRI/segmentation case identifiers.
    5. Ensure that no case identifier appears in multiple splits.
    6. Verify that every case contains the expected output files.
    7. Validate MRI and segmentation tensor dimensions.
    8. Check that MRI and segmentation values are finite.
    9. Verify segmentation labels and foreground presence.
   10. Detect constant MRI volumes.
   11. Check that preprocessing metadata is present and valid.
   12. Report validation results for each split.

Expected dataset sizes:
    Training:    42 MRI sessions
    Validation:  15 MRI sessions
    Test:        14 MRI sessions

Expected files per preprocessed case:
    <case_id>.b2nd        Preprocessed MRI volume
    <case_id>_seg.b2nd    Preprocessed segmentation mask
    <case_id>.pkl         nnU-Net preprocessing metadata

Expected array format:
    (1, D, H, W)

Segmentation labels:
    -1: Ignore label, potentially introduced by nnU-Net.
     0: Background.
   1-7: Foreground segmentation classes.

Requirements:
    - nnSSL/nnU-Net preprocessing environment.
    - Blosc2 and NumPy.
    - Access to raw and preprocessed BOBs datasets.

Notes:
    - This script only reads existing files.
    - No MRI or segmentation data is modified.
    - No preprocessing or model training is performed.
    - The checks do not establish voxel-wise image/mask alignment.
    - Case-level ID checks do not independently establish
      participant-level split independence.
    - Pickle metadata must come from trusted preprocessing outputs.

The script exits with an error if any required validation fails.
"""

import argparse
import json
import pickle
from pathlib import Path

import blosc2
import numpy as np


# ---------------------------------------------------------------------
# Dataset paths and validation configuration
# ---------------------------------------------------------------------

# Default nnU-Net directories mounted inside the Docker environment.
DEFAULT_RAW_ROOT = Path("/opt/nnunet_resources/nnUNet_raw")
DEFAULT_PREPROCESSED_ROOT = Path("/opt/nnunet_resources/nnUNet_preprocessed")

# Expected number of MRI sessions in each dataset split.
EXPECTED = {"train": 42, "val": 15, "test": 14}

# Allowed segmentation labels.
# -1 may be used internally by nnU-Net to identify ignored regions.
ALLOWED_LABELS = {-1, 0, 1, 2, 3, 4, 5, 6, 7}  # -1 may be nnU-Net ignore label

# Mapping between raw nnU-Net image/label directories and their
# corresponding SSL3D preprocessed output directories.
SPLITS = {
    "train": ("imagesTr", "labelsTr", "Spacing__1.00_1.00_1.00___Norm__Z"),
    "val": ("imagesVal", "labelsVal", "SSL3D_val"),
    "test": ("imagesTs", "labelsTs", "SSL3D_test"),
}


def case_ids(folder: Path, suffix: str):
    """
    Extract case identifiers from files in a dataset directory.

    Parameters
    ----------
    folder : Path
        Directory containing MRI, segmentation, or metadata files.

    suffix : str
        Filename suffix to remove when extracting case identifiers.

        Examples:
            "_0000.nii.gz" : Raw single-channel MRI.
            ".nii.gz"      : Raw segmentation mask.
            ".b2nd"        : Preprocessed MRI.
            "_seg.b2nd"    : Preprocessed segmentation mask.
            ".pkl"         : Preprocessing metadata.

    Returns
    -------
    set[str]
        Unique case identifiers extracted from matching filenames.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.

    Notes
    -----
    The function only inspects filenames.
    It does not load or validate file contents.
    """

    # Ensure the requested dataset directory exists.
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    # Find files matching the suffix and remove that suffix
    # to recover their case identifiers.
    return {p.name[: -len(suffix)] for p in folder.glob("*" + suffix)}


def check_case(folder: Path, case_id: str):
    """
    Validate one preprocessed BOBs MRI/segmentation case.

    Parameters
    ----------
    folder : Path
        Directory containing the preprocessed case files.

    case_id : str
        Identifier of the MRI session to validate.

    Returns
    -------
    list[str]
        List of validation errors detected for the case.

        An empty list indicates that all implemented checks passed.

    Validation
    ----------
    Files:
        - MRI .b2nd file exists.
        - Segmentation _seg.b2nd file exists.
        - Preprocessing metadata .pkl file exists.

    MRI:
        - Shape follows (1, D, H, W).
        - Contains only finite values.
        - Does not have zero intensity standard deviation.

    Segmentation:
        - Shape matches the MRI.
        - Contains only finite values.
        - Labels belong to {-1, 0, ..., 7}.
        - Contains at least one foreground voxel.

    Metadata:
        - Can be loaded successfully.
        - Is a non-empty dictionary.

    Notes
    -----
    Individual validation failures are collected rather than
    immediately terminating execution.

    Unexpected exceptions during inspection are also recorded
    as case-specific errors.

    This function does not verify anatomical alignment or the
    completeness of individual preprocessing metadata fields.
    """

    # -------------------------------------------------------------
    # 1. Define the expected preprocessing output files
    # -------------------------------------------------------------

    # Store validation errors detected for this case.
    errors = []

    # Expected image, segmentation, and metadata file paths.
    paths = (
        folder / f"{case_id}.b2nd",
        folder / f"{case_id}_seg.b2nd",
        folder / f"{case_id}.pkl",
    )

    # Stop inspecting this case if any required file is missing.
    if not all(p.is_file() for p in paths):
        return [f"Missing file(s): {[p.name for p in paths if not p.is_file()]}"]

    # -------------------------------------------------------------
    # 2. Load preprocessed MRI and segmentation arrays
    # -------------------------------------------------------------

    try:
        # Read the complete MRI volume from its Blosc2 file.
        image = np.asarray(blosc2.open(str(paths[0]), mode="r")[:])

        # Read the corresponding preprocessed segmentation mask.
        seg = np.asarray(blosc2.open(str(paths[1]), mode="r")[:])

        # ---------------------------------------------------------
        # 3. Validate MRI and segmentation dimensions
        # ---------------------------------------------------------

        # The expected MRI format is:
        # (channels=1, depth, height, width).
        if image.ndim != 4 or image.shape[0] != 1:
            errors.append(f"Unexpected MRI shape {image.shape}")

        # Segmentation and MRI arrays must have identical shapes
        # to represent corresponding voxel locations.
        if seg.shape != image.shape:
            errors.append(f"Image/seg shape mismatch: {image.shape} vs {seg.shape}")

        # ---------------------------------------------------------
        # 4. Validate numerical values
        # ---------------------------------------------------------

        # Reject MRI volumes containing NaN or infinity.
        if not np.isfinite(image).all():
            errors.append("MRI has NaN/Inf")

        # Reject segmentation masks containing NaN or infinity.
        if not np.isfinite(seg).all():
            errors.append("Mask has NaN/Inf")

        # If the mask is finite, verify that all observed labels
        # belong to the expected segmentation label set.
        elif not np.all(np.isin(np.unique(seg), list(ALLOWED_LABELS))):
            errors.append(f"Unexpected labels: {np.unique(seg).tolist()}")

        # ---------------------------------------------------------
        # 5. Check segmentation foreground and MRI variation
        # ---------------------------------------------------------

        # Ensure the segmentation contains at least one foreground
        # voxel, represented by a label strictly greater than zero.
        if not np.any(seg > 0):
            errors.append("No foreground segmentation labels")

        # Detect constant MRI volumes.
        # A zero standard deviation indicates no intensity variation.
        if image.size and float(np.std(image)) == 0.0:
            errors.append("Constant MRI")

        # ---------------------------------------------------------
        # 6. Validate preprocessing metadata
        # ---------------------------------------------------------

        # Only load metadata generated locally by our trusted preprocessing.
        #
        # Pickle is not safe for untrusted files because loading
        # arbitrary pickle data can execute Python code.
        with paths[2].open("rb") as f:
            props = pickle.load(f)

        # Metadata is expected to be a non-empty dictionary.
        if not isinstance(props, dict) or not props:
            errors.append("Missing/invalid preprocessing metadata")

    # -------------------------------------------------------------
    # 7. Record unexpected inspection errors
    # -------------------------------------------------------------

    # Catch errors arising from file reading, deserialization,
    # or numerical inspection and report them for this case.
    except Exception as exc:
        errors.append(f"Cannot inspect case: {type(exc).__name__}: {exc}")

    # Return all errors detected for the current case.
    return errors


def main():
    """
    Validate all preprocessed BOBs training, validation, and test cases.

    The function:
        1. Parses input directory configuration.
        2. Verifies required dataset metadata and SSL3D plans.
        3. Checks raw and preprocessed case identifiers.
        4. Runs numerical and structural validation for every case.
        5. Reports passed and failed cases.
        6. Exits with an error if any case fails.

    Notes
    -----
    The expected number of BOBs sessions is fixed by the project's
    predefined dataset splits.

    The script checks every case in the selected dataset.
    """

    # -------------------------------------------------------------
    # 1. Parse command-line arguments
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(description=__doc__)

    # Root directory of the converted nnU-Net raw datasets.
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)

    # Root directory of the preprocessed nnU-Net datasets.
    parser.add_argument(
        "--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED_ROOT
    )

    # Select the BOBs dataset directory.
    parser.add_argument("--dataset", default="Dataset102_BOBs")

    args = parser.parse_args()

    # -------------------------------------------------------------
    # 2. Locate the raw and preprocessed datasets
    # -------------------------------------------------------------

    raw = args.raw_root / args.dataset
    processed = args.preprocessed_root / args.dataset

    # Verify required raw dataset metadata.
    assert (raw / "dataset.json").is_file(), f"Raw dataset missing: {raw}"

    # Verify that the BOBs segmentation label mapping is available.
    assert (raw / "label_mapping.json").is_file(), "Missing BOBs label mapping"

    # -------------------------------------------------------------
    # 3. Verify the existing SSL3D preprocessing plans
    # -------------------------------------------------------------

    # Find plans files associated with the BOBs SSL3D configuration.
    plans_files = list(processed.glob("ptPlans__SSL3D_BOBs*.json"))

    # Require exactly one matching plans file.
    if len(plans_files) != 1:
        raise ValueError(f"Expected one SSL3D_BOBs plan, got {plans_files}")

    # Load the existing plans metadata.
    plans = json.loads(plans_files[0].read_text(encoding="utf-8"))

    # Check that the plans contain preprocessing configurations.
    if "configurations" not in plans:
        raise ValueError("SSL3D plans are missing configurations")

    # -------------------------------------------------------------
    # 4. Validate all dataset splits
    # -------------------------------------------------------------

    # Count cases that fail one or more validation checks.
    failures = 0

    # Track case identifiers across splits to detect duplicates.
    all_cases = set()

    for split, (image_folder, label_folder, processed_folder) in SPLITS.items():
        # ---------------------------------------------------------
        # 5. Verify raw MRI and segmentation identifiers
        # ---------------------------------------------------------

        # Extract raw MRI case identifiers.
        expected = case_ids(raw / image_folder, "_0000.nii.gz")

        # Extract raw segmentation identifiers.
        labels = case_ids(raw / label_folder, ".nii.gz")

        # Every MRI must have a corresponding raw segmentation mask.
        if expected != labels:
            raise ValueError(f"Raw images/masks mismatch for {split}")

        # Verify the expected number of MRI sessions in this split.
        if len(expected) != EXPECTED[split]:
            raise ValueError(
                f"Expected {EXPECTED[split]} {split} cases, got {len(expected)}"
            )

        # Check for case identifiers already observed in earlier splits.
        if all_cases.intersection(expected):
            raise ValueError(
                f"Duplicate case IDs across splits: {all_cases.intersection(expected)}"
            )

        # Register the current split's identifiers.
        all_cases.update(expected)

        # ---------------------------------------------------------
        # 6. Verify preprocessed output identifiers
        # ---------------------------------------------------------

        # Locate the split-specific preprocessed output directory.
        folder = processed / processed_folder

        # Collect preprocessed MRI identifiers.
        # Exclude segmentation files with the "_seg" suffix.
        actual_images = {c for c in case_ids(folder, ".b2nd") if not c.endswith("_seg")}

        # Collect preprocessed segmentation identifiers.
        actual_labels = case_ids(folder, "_seg.b2nd")

        # Collect preprocessing metadata identifiers.
        actual_properties = case_ids(folder, ".pkl")

        # Compare expected raw identifiers with each output category.
        for label, actual in (
            ("images", actual_images),
            ("masks", actual_labels),
            ("metadata", actual_properties),
        ):
            # Detect missing or unexpected preprocessed cases.
            if actual != expected:
                raise ValueError(
                    f"{split} {label} ID mismatch: missing={sorted(expected - actual)}, "
                    f"extra={sorted(actual - expected)}"
                )

        # ---------------------------------------------------------
        # 7. Validate individual preprocessed cases
        # ---------------------------------------------------------

        # Number of cases passing every implemented validation check.
        passed = 0

        # Inspect cases in a consistent sorted order.
        for case_id in sorted(expected):
            # Collect validation errors for the current case.
            errors = check_case(folder, case_id)

            if errors:
                # Count a case as failed if at least one error occurs.
                failures += 1

                # Display all errors detected for this case.
                print(f"FAIL {split}/{case_id}: {'; '.join(errors)}", flush=True)

            else:
                # The case passed all implemented checks.
                passed += 1

        # Display the split-level validation summary.
        print(f"{split.upper()}: {passed}/{len(expected)}", flush=True)

    # -------------------------------------------------------------
    # 8. Report overall preprocessing validation results
    # -------------------------------------------------------------

    # Report the number of cases that passed across all splits.
    print(
        f"TOTAL: {sum(EXPECTED.values()) - failures}/{sum(EXPECTED.values())}",
        flush=True,
    )

    # Exit with a failure status if any case did not pass validation.
    if failures:
        raise SystemExit(f"{failures} cases failed")

    # All expected cases passed the implemented checks.
    print("ALL BOBs CHECKS PASSED!", flush=True)


if __name__ == "__main__":
    main()
