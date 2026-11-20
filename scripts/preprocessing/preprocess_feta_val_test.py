"""
Preprocess FeTA validation and test volumes using existing SSL3D plans.

This script applies the nnU-Net preprocessing configuration already
generated for the FeTA training dataset to validation and test cases.

The goal is to ensure that all FeTA splits are processed using the
same SSL3D spatial and intensity normalization configuration.

Main steps:
    1. Load predefined FeTA train/validation/test splits.
    2. Verify split sizes and subject independence.
    3. Load the existing SSL3D nnU-Net plans.
    4. Initialize the official DefaultPreprocessor.
    5. Preprocess validation MRI/segmentation pairs.
    6. Preprocess test MRI/segmentation pairs.
    7. Verify all expected output files.

Expected split sizes:
    Training:    48 subjects (already preprocessed)
    Validation:  16 subjects
    Test:        16 subjects

Output directories:
    Dataset101_FeTA/SSL3D_val/
    Dataset101_FeTA/SSL3D_test/

Expected output files per subject:
    <subject>.b2nd
    <subject>_seg.b2nd
    <subject>.pkl

Important:
    - No fingerprint extraction or experiment planning is performed.
    - Training data is not reprocessed.
    - Existing SSL3D training-derived plans are reused.
    - Complete existing outputs are skipped.
    - Incomplete output sets trigger an error.
    - Original raw MRI and segmentation files are not modified.
"""

import json
from pathlib import Path

from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
    DefaultPreprocessor,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


# ============================================================
# PATHS
# ============================================================

# Identifier of the FeTA nnU-Net dataset.
DATASET_NAME = "Dataset101_FeTA"

# Read nnU-Net data roots from the configured nnU-Net environment.
RAW_DIR = Path(nnUNet_raw) / DATASET_NAME
PREPROCESSED_DIR = Path(nnUNet_preprocessed) / DATASET_NAME

# Location of the predefined subject-level split file
# inside the preprocessing environment.
SPLITS_PATH = Path("/opt/project_splits/feta_splits.json")

# Name of the previously generated FeTA SSL3D plans file.
# The filename records the target voxel spacing and normalization.
PLANS_NAME = "ptPlans__SSL3D_FeTA____Spacing__1.00_1.00_1.00___Norm__Z.json"

# Resolve the existing plans and dataset metadata files.
PLANS_PATH = PREPROCESSED_DIR / PLANS_NAME
DATASET_JSON_PATH = PREPROCESSED_DIR / "dataset.json"


def preprocess_split(
    split_name,
    subject_ids,
    image_folder,
    label_folder,
    output_folder,
    preprocessor,
    plans_manager,
    configuration_manager,
    dataset_json,
):
    """
    Preprocess one split using existing SSL3D plans.

    Uses the official nnU-Net DefaultPreprocessor.
    Does not recompute fingerprint or planning.

    Parameters
    ----------
    split_name : str
        Descriptive name of the split being processed.

    subject_ids : list
        Identifiers of the subjects belonging to the split.

    image_folder : str
        Name of the raw MRI image directory.

    label_folder : str
        Name of the corresponding raw segmentation directory.

    output_folder : Path
        Destination for preprocessed images, masks, and metadata.

    preprocessor : DefaultPreprocessor
        Initialized nnU-Net preprocessing object.

    plans_manager : PlansManager
        Manager containing the previously generated SSL3D plans.

    configuration_manager
        nnU-Net configuration containing the preprocessing settings.

    dataset_json : dict
        Dataset metadata required by the preprocessing procedure.

    Notes
    -----
    Each subject produces an image, segmentation, and metadata file.

    Cases with all three output files already present are skipped.
    Partially processed cases raise an error to prevent inconsistent
    or incomplete preprocessing results.

    Raises
    ------
    FileNotFoundError
        If an input MRI or segmentation file is missing.

    RuntimeError
        If preprocessing outputs are incomplete or missing.
    """

    # -------------------------------------------------------------
    # 1. Prepare the output directory
    # -------------------------------------------------------------

    # Create the destination directory if it does not exist.
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n===== PREPROCESSING {split_name.upper()} =====")
    print(f"Subjects: {len(subject_ids)}")

    # -------------------------------------------------------------
    # 2. Process each subject in the selected split
    # -------------------------------------------------------------

    # Sorting gives a consistent processing order.
    for subject in sorted(subject_ids):
        # Construct nnU-Net raw MRI and segmentation paths.
        # _0000 indicates the first image channel.
        image_path = RAW_DIR / image_folder / f"{subject}_0000.nii.gz"
        label_path = RAW_DIR / label_folder / f"{subject}.nii.gz"

        # Verify that the required source files exist.
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        if not label_path.is_file():
            raise FileNotFoundError(label_path)

        # Base filename used for the preprocessed outputs.
        output_base = output_folder / subject

        # Expected preprocessed image, segmentation, and metadata files.
        expected_files = [
            output_folder / f"{subject}.b2nd",
            output_folder / f"{subject}_seg.b2nd",
            output_folder / f"{subject}.pkl",
        ]

        # Identify any outputs that are already present.
        existing = [p for p in expected_files if p.exists()]

        # Do not overwrite correctly processed cases.
        if len(existing) == 3:
            print(f"Already processed: {subject}")
            continue

        # Detect partial/incomplete output.
        # The subject must not be silently reused if only some
        # preprocessing outputs are present.
        if existing:
            raise RuntimeError(
                f"Incomplete output for {subject}: {existing}. "
                "Inspect and remove only these incomplete files "
                "before rerunning."
            )

        # Official nnU-Net preprocessing and saving.
        # The existing SSL3D plans control the preprocessing settings.
        preprocessor.run_case_save(
            output_filename_truncated=str(output_base),
            image_files=[str(image_path)],
            seg_file=str(label_path),
            plans_manager=plans_manager,
            configuration_manager=configuration_manager,
            dataset_json=dataset_json,
            overwrite_existing=True,
        )

        # Verify that preprocessing created all expected files.
        for path in expected_files:
            if not path.is_file():
                raise RuntimeError(f"Expected output was not created: {path}")

        print(f"Processed: {subject}")

    # -------------------------------------------------------------
    # 3. Final split-level verification
    # -------------------------------------------------------------

    # Confirm that every requested subject has complete outputs.
    # This includes subjects skipped because their files already existed.
    for subject in subject_ids:
        for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
            path = output_folder / f"{subject}{suffix}"
            if not path.is_file():
                raise RuntimeError(f"Missing output: {path}")

    print(f"\n{split_name.upper()} COMPLETE: {len(subject_ids)}/{len(subject_ids)}")


def main():
    """
    Preprocess FeTA validation and test data with existing SSL3D plans.

    Training data is assumed to have been preprocessed previously.
    The same configuration is reused for validation and test cases.
    """

    # --------------------------------------------------------
    # 1. Load the original splits
    # --------------------------------------------------------

    # Read the predefined FeTA subject-level split definitions.
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]

    # Verify the expected training, validation, and test sizes.
    assert len(train_ids) == 48
    assert len(val_ids) == 16
    assert len(test_ids) == 16

    # Ensure all 80 subject identifiers are unique across splits.
    assert len(set(train_ids + val_ids + test_ids)) == 80

    print("===== FETA SPLITS =====")
    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")

    # --------------------------------------------------------
    # 2. Load existing SSL3D plans
    # --------------------------------------------------------

    # Verify that the existing training-derived plans are available.
    if not PLANS_PATH.is_file():
        raise FileNotFoundError(PLANS_PATH)

    # Verify that the nnU-Net dataset metadata is available.
    if not DATASET_JSON_PATH.is_file():
        raise FileNotFoundError(DATASET_JSON_PATH)

    # Load the predefined SSL3D preprocessing plans.
    plans_manager = PlansManager(str(PLANS_PATH))

    # Select the 3D full-resolution configuration.
    configuration_manager = plans_manager.get_configuration("3d_fullres")

    # Load dataset-specific metadata.
    with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    # Report the configuration reused across validation and test.
    print("\n===== SSL3D CONFIGURATION =====")
    print(f"Plans: {PLANS_NAME}")
    print(f"Spacing: {configuration_manager.spacing}")
    print(f"Normalization: {configuration_manager.normalization_schemes}")

    # --------------------------------------------------------
    # 3. Initialize the official nnU-Net preprocessor
    # --------------------------------------------------------

    # Use the official nnU-Net preprocessing implementation.
    preprocessor = DefaultPreprocessor(verbose=False)

    # --------------------------------------------------------
    # 4. Validation
    # --------------------------------------------------------

    # Preprocess validation subjects using the existing SSL3D plans.
    preprocess_split(
        split_name="validation",
        subject_ids=val_ids,
        image_folder="imagesVal",
        label_folder="labelsVal",
        output_folder=PREPROCESSED_DIR / "SSL3D_val",
        preprocessor=preprocessor,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        dataset_json=dataset_json,
    )

    # --------------------------------------------------------
    # 5. Test
    # --------------------------------------------------------

    # Preprocess test subjects using the same training-derived plans.
    preprocess_split(
        split_name="test",
        subject_ids=test_ids,
        image_folder="imagesTs",
        label_folder="labelsTs",
        output_folder=PREPROCESSED_DIR / "SSL3D_test",
        preprocessor=preprocessor,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        dataset_json=dataset_json,
    )

    # --------------------------------------------------------
    # 6. Display final preprocessing summary
    # --------------------------------------------------------

    print("\n===== ALL DONE =====")
    print("Train: 48 (previously processed)")
    print("Validation: 16")
    print("Test: 16")
    print("\nAll splits use the same SSL3D plans.")


if __name__ == "__main__":
    main()
