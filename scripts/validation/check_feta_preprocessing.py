"""
Validate SSL3D/nnU-Net preprocessing outputs for the FeTA dataset.

This script performs structural, numerical, and metadata quality-control
checks on preprocessed FeTA fetal brain MRI volumes and their
corresponding segmentation masks.

The validation covers all predefined train, validation, and test cases.

Main validation steps:
    1. Load the predefined FeTA subject-level splits.
    2. Verify split sizes and subject independence.
    3. Load the existing SSL3D preprocessing plans.
    4. Verify the expected voxel spacing and normalization scheme.
    5. Check MRI, segmentation, and metadata file identifiers.
    6. Validate the dimensions of every preprocessed volume.
    7. Verify image numerical types and finite intensity values.
    8. Detect MRI volumes with negligible intensity variation.
    9. Validate segmentation label values and foreground presence.
   10. Verify required nnU-Net preprocessing metadata fields.
   11. Report validation results for every subject and split.

Expected dataset sizes:
    Training:    48 subjects
    Validation:  16 subjects
    Test:        16 subjects

Expected SSL3D configuration:
    Spacing:       (1.0, 1.0, 1.0) mm
    Normalization: ZScoreNormalization

Expected files per preprocessed subject:
    <subject>.b2nd        Preprocessed MRI volume
    <subject>_seg.b2nd    Preprocessed segmentation mask
    <subject>.pkl         nnU-Net preprocessing metadata

Expected array format:
    (1, D, H, W)

Segmentation labels:
    -1: Ignore label, potentially introduced during preprocessing.
     0: Background.
   1-7: Foreground anatomical tissue classes.

Requirements:
    - nnSSL/nnU-Net preprocessing environment.
    - Blosc2 and NumPy.
    - Existing FeTA SSL3D preprocessing outputs.
    - Predefined FeTA subject-level splits.

Notes:
    - The script only reads existing preprocessing outputs.
    - No image or segmentation data is modified.
    - All 80 expected FeTA cases are inspected.
    - Structural and numerical checks do not establish
      voxel-wise anatomical alignment.
    - Pickle metadata must originate from trusted preprocessing.

The script raises an error if any required validation fails.
"""

import json
import pickle
from pathlib import Path

import blosc2
import numpy as np


# ---------------------------------------------------------------------
# Dataset paths and preprocessing configuration
# ---------------------------------------------------------------------

# Root directory containing preprocessed FeTA SSL3D volumes.
ROOT = Path("/opt/nnunet_resources/nnUNet_preprocessed/Dataset101_FeTA")

# Predefined FeTA train/validation/test subject-level splits.
SPLITS_PATH = Path("/opt/project_splits/feta_splits.json")

# Expected adapted SSL3D plans file.
# The filename records 1 mm isotropic spacing and Z-score normalization.
PLANS_PATH = ROOT / ("ptPlans__SSL3D_FeTA____Spacing__1.00_1.00_1.00___Norm__Z.json")

# Map each dataset split to its preprocessed output directory.
#
# Training uses the directory identified by its preprocessing settings.
# Validation and test use dedicated SSL3D output directories.
SPLIT_DIRS = {
    "train": ROOT / "Spacing__1.00_1.00_1.00___Norm__Z",
    "val": ROOT / "SSL3D_val",
    "test": ROOT / "SSL3D_test",
}

# Expected numbers of FeTA subjects in each split.
EXPECTED_COUNTS = {"train": 48, "val": 16, "test": 16}

# -1 may be used internally for ignored regions.
# Background is 0 and foreground tissue classes are 1 through 7.
ALLOWED_LABELS = set(range(8)) | {-1}


def load_b2nd(path):
    """
    Load a preprocessed nnU-Net Blosc2 array.

    Parameters
    ----------
    path : Path
        Path to a preprocessed MRI or segmentation .b2nd file.

    Returns
    -------
    np.ndarray
        Complete array stored in the Blosc2 file.

    Notes
    -----
    The file is opened in read-only mode.
    The source data is not modified.
    """

    # Read the complete array from the Blosc2 container.
    return np.asarray(blosc2.open(urlpath=str(path), mode="r")[:])


def check_case(subject, folder):
    """
    Validate one preprocessed FeTA MRI/segmentation pair.

    Parameters
    ----------
    subject : str
        FeTA subject identifier, such as "sub-006".

    folder : Path
        Directory containing the subject's preprocessed files.

    Returns
    -------
    dict
        Summary of the validated case containing:
            - Subject identifier.
            - MRI array shape.
            - Minimum MRI intensity.
            - Maximum MRI intensity.
            - Mean MRI intensity.
            - MRI intensity standard deviation.
            - Observed segmentation labels.

    Validation
    ----------
    Files:
        - MRI .b2nd exists.
        - Segmentation _seg.b2nd exists.
        - Metadata .pkl exists.

    MRI:
        - Shape follows (1, D, H, W).
        - Data type is floating-point.
        - Intensities are finite.
        - Standard deviation is at least 1e-8.

    Segmentation:
        - Shape follows (1, D, H, W).
        - Spatial dimensions match the MRI.
        - Contains finite integer-valued labels.
        - Labels belong to {-1, 0, ..., 7}.
        - Contains at least one anatomical foreground class.

    Metadata:
        - Is stored as a dictionary.
        - Contains required spacing and cropping information.

    Raises
    ------
    FileNotFoundError
        If any required preprocessing output is missing.

    ValueError
        If image, segmentation, or metadata validation fails.

    Notes
    -----
    The function checks the preprocessed arrays but does not
    independently verify anatomical alignment against raw MRI data.
    """

    # -------------------------------------------------------------
    # 1. Locate the expected preprocessing output files
    # -------------------------------------------------------------

    # Construct image, segmentation, and metadata paths.
    image_path = folder / f"{subject}.b2nd"
    mask_path = folder / f"{subject}_seg.b2nd"
    metadata_path = folder / f"{subject}.pkl"

    # Verify that all three required files exist.
    for path in (image_path, mask_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing: {path}")

    # -------------------------------------------------------------
    # 2. Load MRI and segmentation arrays
    # -------------------------------------------------------------

    # Read the actual preprocessed arrays.
    image = load_b2nd(image_path)
    mask = load_b2nd(mask_path)

    # -------------------------------------------------------------
    # 3. Validate MRI and segmentation dimensions
    # -------------------------------------------------------------

    # nnU-Net format: (channels, depth, height, width).
    # Only one MRI channel is expected.
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"{subject}: invalid MRI shape {image.shape}")

    # The segmentation must also contain one channel dimension.
    if mask.ndim != 4 or mask.shape[0] != 1:
        raise ValueError(f"{subject}: invalid mask shape {mask.shape}")

    # MRI and mask must have the same spatial dimensions.
    if image.shape[1:] != mask.shape[1:]:
        raise ValueError(
            f"{subject}: MRI/mask shape mismatch {image.shape} vs {mask.shape}"
        )

    # -------------------------------------------------------------
    # 4. Validate MRI numerical properties
    # -------------------------------------------------------------

    # MRI intensities must use a floating-point numerical type.
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"{subject}: unexpected MRI dtype {image.dtype}")

    # Reject MRI volumes containing NaN or infinity values.
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{subject}: MRI contains NaN or Inf")

    # Reject nearly constant MRI volumes.
    #
    # Very low intensity variation may indicate invalid input
    # data or unexpected preprocessing behavior.
    if float(np.std(image)) < 1e-8:
        raise ValueError(f"{subject}: MRI has almost no intensity variation")

    # -------------------------------------------------------------
    # 5. Validate segmentation numerical values
    # -------------------------------------------------------------

    # Segmentation masks must not contain NaN or infinity.
    if not np.all(np.isfinite(mask)):
        raise ValueError(f"{subject}: mask contains NaN or Inf")

    # Extract all distinct labels observed in this segmentation.
    unique_labels = np.unique(mask)

    # Verify that labels are integer-valued.
    #
    # A small numerical tolerance allows insignificant
    # floating-point representation differences.
    if not np.allclose(
        unique_labels,
        np.round(unique_labels),
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError(f"{subject}: non-integer segmentation labels")

    # Convert the observed label values to integer identifiers.
    observed_labels = set(np.rint(unique_labels).astype(int).tolist())

    # Check that every observed label belongs to the allowed set.
    if not observed_labels.issubset(ALLOWED_LABELS):
        raise ValueError(f"{subject}: unexpected labels {sorted(observed_labels)}")

    # -------------------------------------------------------------
    # 6. Verify anatomical foreground presence
    # -------------------------------------------------------------

    # At least one anatomical foreground class must exist.
    # Foreground labels are represented by integers 1 through 7.
    if not np.any((mask >= 1) & (mask <= 7)):
        raise ValueError(f"{subject}: segmentation has no foreground")

    # -------------------------------------------------------------
    # 7. Validate nnU-Net preprocessing metadata
    # -------------------------------------------------------------

    # Metadata were generated by our own nnU-Net preprocessing.
    #
    # Only trusted local pickle files should be loaded because
    # unpickling arbitrary files can execute Python code.
    with open(metadata_path, "rb") as f:
        properties = pickle.load(f)

    # Metadata must be stored as a Python dictionary.
    if not isinstance(properties, dict):
        raise ValueError(f"{subject}: invalid metadata format")

    # Verify the presence of required preprocessing properties.
    #
    # These fields describe original spacing and geometry changes
    # applied during cropping and resampling.
    for key in (
        "spacing",
        "shape_before_cropping",
        "bbox_used_for_cropping",
        "shape_after_cropping_and_before_resampling",
    ):
        if key not in properties:
            raise ValueError(f"{subject}: missing metadata key {key}")

    # -------------------------------------------------------------
    # 8. Return case-level validation statistics
    # -------------------------------------------------------------

    # Store image statistics and observed labels for reporting.
    return {
        "subject": subject,
        "shape": list(image.shape),
        "image_min": float(image.min()),
        "image_max": float(image.max()),
        "image_mean": float(image.mean()),
        "image_std": float(image.std()),
        "labels": sorted(observed_labels),
    }


def main():
    """
    Validate all FeTA SSL3D preprocessing outputs.

    The function performs:
        1. Subject-level split validation.
        2. SSL3D plans validation.
        3. Preprocessed file identifier checks.
        4. Individual case validation.
        5. Split-level and overall result reporting.

    Each subject is validated using check_case().

    Case-specific errors are collected to provide a complete
    validation report rather than stopping at the first failed case.

    The script raises RuntimeError if any validation error remains.
    """

    # -------------------------------------------------------------
    # 1. Load and validate the predefined FeTA splits
    # -------------------------------------------------------------

    print("===== CHECK FETA SSL3D PREPROCESSING =====")

    # Load predefined subject identifiers for train, validation, and test.
    with open(SPLITS_PATH, encoding="utf-8") as f:
        splits = json.load(f)

    # Combine all identifiers to verify the complete dataset.
    all_ids = splits["train"] + splits["val"] + splits["test"]

    # Require exactly 80 unique subject identifiers.
    # This prevents duplicated subjects across splits.
    if len(all_ids) != 80 or len(set(all_ids)) != 80:
        raise ValueError("Invalid or overlapping FeTA splits")

    # Verify the expected number of subjects in each split.
    for split_name, count in EXPECTED_COUNTS.items():
        if len(splits[split_name]) != count:
            raise ValueError(f"Unexpected {split_name} split size")

    # -------------------------------------------------------------
    # 2. Validate the adapted SSL3D preprocessing plans
    # -------------------------------------------------------------

    # Verify the adapted SSL3D configuration.
    with open(PLANS_PATH, encoding="utf-8") as f:
        plans = json.load(f)

    # Retrieve the full-resolution 3D preprocessing configuration.
    config = plans["configurations"]["3d_fullres"]

    # Expected voxel spacing: 1 mm isotropic resolution.
    if not np.allclose(config["spacing"], [1, 1, 1]):
        raise ValueError(f"Unexpected spacing: {config['spacing']}")

    # Verify Z-score intensity normalization.
    if config["normalization_schemes"] != ["ZScoreNormalization"]:
        raise ValueError("Unexpected normalization scheme")

    print("SSL3D plans: OK")
    print("Spacing: 1 x 1 x 1 mm")
    print("Normalization: Z-score")

    # -------------------------------------------------------------
    # 3. Initialize validation result containers
    # -------------------------------------------------------------

    # Collect errors detected across all dataset splits.
    errors = []

    # Store successful case summaries grouped by split.
    results = {}

    # -------------------------------------------------------------
    # 4. Validate each dataset split
    # -------------------------------------------------------------

    for split_name, folder in SPLIT_DIRS.items():
        print(f"\n===== {split_name.upper()} =====")

        # Ensure the expected preprocessed split directory exists.
        if not folder.is_dir():
            raise FileNotFoundError(folder)

        # Expected subject identifiers defined by the split JSON.
        expected_ids = set(splits[split_name])

        # ---------------------------------------------------------
        # 5. Compare image, mask, and metadata identifiers
        # ---------------------------------------------------------

        # Collect preprocessed MRI case identifiers.
        # Exclude files ending in "_seg.b2nd".
        image_ids = {
            p.name.removesuffix(".b2nd")
            for p in folder.glob("*.b2nd")
            if not p.name.endswith("_seg.b2nd")
        }

        # Collect segmentation case identifiers.
        mask_ids = {p.name.removesuffix("_seg.b2nd") for p in folder.glob("*_seg.b2nd")}

        # Collect preprocessing metadata case identifiers.
        metadata_ids = {p.name.removesuffix(".pkl") for p in folder.glob("*.pkl")}

        # Compare each output type against the expected subject set.
        for kind, ids in (
            ("MRI", image_ids),
            ("mask", mask_ids),
            ("metadata", metadata_ids),
        ):
            # Record missing or unexpected case identifiers.
            if ids != expected_ids:
                errors.append(
                    f"{split_name} {kind}: "
                    f"missing={sorted(expected_ids - ids)}, "
                    f"extra={sorted(ids - expected_ids)}"
                )

        # Initialize successful validation results for this split.
        results[split_name] = []

        # ---------------------------------------------------------
        # 6. Validate each preprocessed subject
        # ---------------------------------------------------------

        for subject in sorted(expected_ids):
            try:
                # Validate the MRI, segmentation, and metadata.
                result = check_case(subject, folder)

                # Store statistics for successfully validated subjects.
                results[split_name].append(result)

                # Display the successful case and MRI dimensions.
                print(f"[OK] {subject} | {result['shape']}")

            except Exception as exc:
                # Record case-specific errors without interrupting
                # validation of the remaining subjects.
                errors.append(f"{split_name}/{subject}: {exc}")

                print(f"[FAIL] {subject}: {exc}")

        # Display the number of successful cases in this split.
        print(
            f"{split_name}: "
            f"{len(results[split_name])}/"
            f"{len(expected_ids)} subjects passed"
        )

    # -------------------------------------------------------------
    # 7. Generate the final validation report
    # -------------------------------------------------------------

    # Final summary.
    print("\n===== FINAL SUMMARY =====")

    # Count successfully validated cases across all splits.
    total_passed = sum(len(v) for v in results.values())

    # Display the results separately for train, validation, and test.
    for split_name in ("train", "val", "test"):
        print(
            f"{split_name.upper()}: "
            f"{len(results[split_name])}/"
            f"{EXPECTED_COUNTS[split_name]}"
        )

    # Display the overall number of successful FeTA cases.
    print(f"TOTAL: {total_passed}/80")

    # -------------------------------------------------------------
    # 8. Report any validation failures
    # -------------------------------------------------------------

    if errors:
        print("\nERRORS:")

        # Display all collected errors to facilitate debugging.
        for error in errors:
            print(f"- {error}")

        # Fail the validation if any structural, numerical,
        # or metadata inconsistency was detected.
        raise RuntimeError(f"Preprocessing validation failed: {len(errors)} error(s)")

    # -------------------------------------------------------------
    # 9. Confirm successful preprocessing validation
    # -------------------------------------------------------------

    print("\nALL CHECKS PASSED!")
    print("All 80 FeTA cases passed the structural")
    print("and numerical preprocessing checks.")


if __name__ == "__main__":
    main()
