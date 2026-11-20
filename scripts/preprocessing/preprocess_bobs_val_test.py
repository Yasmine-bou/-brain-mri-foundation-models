"""
Preprocess BOBs validation and test volumes using existing SSL3D plans.

This script applies the nnU-Net preprocessing configuration previously
created for the BOBs training dataset to validation and test cases.

The procedure:
    1. Locate the raw and preprocessed BOBs dataset directories.
    2. Load the existing SSL3D_BOBs plans file.
    3. Initialize the configured nnU-Net preprocessor.
    4. Identify and validate MRI/segmentation pairs.
    5. Verify the expected number of validation and test cases.
    6. Preprocess each case using the existing training-derived plans.
    7. Save the preprocessed image, segmentation, and metadata files.

Expected split sizes:
    Validation: 15 MRI sessions
    Test:       14 MRI sessions

Output directories:
    Dataset102_BOBs/SSL3D_val/
    Dataset102_BOBs/SSL3D_test/

Expected output files per case:
    <case_id>.b2nd
    <case_id>_seg.b2nd
    <case_id>.pkl

Run inside the nnSSL Docker environment, with nnUNet_raw and
nnUNet_preprocessed mounted:

    python /tmp/preprocess_bobs_val_test.py

Important:
    - Existing training-derived plans are reused.
    - No fingerprint extraction or experiment planning is performed.
    - Training cases are not reprocessed.
    - Complete outputs are skipped unless --overwrite is supplied.
    - Incomplete outputs trigger an error unless overwriting is enabled.
"""

import argparse
import json
from pathlib import Path

from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
    DefaultPreprocessor,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


# ---------------------------------------------------------------------
# Default paths and dataset configuration
# ---------------------------------------------------------------------

# Mounted nnU-Net directories inside the Docker environment.
DEFAULT_RAW_ROOT = Path("/opt/nnunet_resources/nnUNet_raw")
DEFAULT_PREPROCESSED_ROOT = Path("/opt/nnunet_resources/nnUNet_preprocessed")

# Expected numbers of BOBs MRI sessions in validation and test splits.
EXPECTED_COUNTS = {"val": 15, "test": 14}


def cases_from_raw(raw: Path, image_folder: str, label_folder: str):
    """
    Discover and validate MRI/segmentation pairs in a raw nnU-Net split.

    Parameters
    ----------
    raw : Path
        Root directory of the raw BOBs nnU-Net dataset.

    image_folder : str
        Name of the image directory, such as imagesVal or imagesTs.

    label_folder : str
        Name of the corresponding segmentation directory.

    Returns
    -------
    list
        List of tuples containing:
            - Case identifier.
            - List containing the MRI image path.
            - Corresponding segmentation path.

    Raises
    ------
    FileNotFoundError
        If the requested image or label directory does not exist.

    ValueError
        If image and segmentation identifiers do not match.

    Notes
    -----
    The MRI filename suffix _0000 indicates the first image channel
    in the nnU-Net naming convention.
    """

    # Locate the image and segmentation directories.
    image_dir, label_dir = raw / image_folder, raw / label_folder

    # Both directories must be present.
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing {image_dir} or {label_dir}")

    cases = []

    # Discover single-channel MRI files.
    images = sorted(image_dir.glob("*_0000.nii.gz"))

    # Extract case IDs from MRI filenames.
    image_ids = {p.name[: -len("_0000.nii.gz")] for p in images}

    # Extract case IDs from segmentation filenames.
    label_ids = {p.name[: -len(".nii.gz")] for p in label_dir.glob("*.nii.gz")}

    # Ensure that each MRI has a matching segmentation and vice versa.
    if image_ids != label_ids:
        raise ValueError(
            f"{image_folder}/{label_folder} mismatch: "
            f"images without labels={sorted(image_ids - label_ids)}, "
            f"labels without images={sorted(label_ids - image_ids)}"
        )

    # Build the input specifications expected by nnU-Net preprocessing.
    for image in images:
        case_id = image.name[: -len("_0000.nii.gz")]
        cases.append((case_id, [str(image)], str(label_dir / (case_id + ".nii.gz"))))

    return cases


def main():
    """
    Preprocess BOBs validation and test cases using existing SSL3D plans.

    The script reuses the training-derived nnU-Net configuration and
    processes validation and test cases independently.

    Command-line options allow overriding dataset directories,
    choosing a configuration, and enabling output replacement.
    """

    # -------------------------------------------------------------
    # 1. Parse command-line arguments
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(description=__doc__)

    # Root directory containing raw nnU-Net datasets.
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)

    # Root directory containing preprocessed nnU-Net datasets.
    parser.add_argument(
        "--preprocessed-root", type=Path, default=DEFAULT_PREPROCESSED_ROOT
    )

    # Select the BOBs dataset directory.
    parser.add_argument("--dataset", default="Dataset102_BOBs")

    # Select the nnU-Net planning configuration.
    parser.add_argument("--configuration", default="3d_fullres")

    # Allow existing validation/test cases to be reprocessed.
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing val/test cases"
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # 2. Locate the raw and preprocessed dataset directories
    # -------------------------------------------------------------

    raw = args.raw_root / args.dataset
    preprocessed = args.preprocessed_root / args.dataset

    # Both the source and existing preprocessing directories are required.
    if not raw.is_dir() or not preprocessed.is_dir():
        raise FileNotFoundError(
            f"Raw or preprocessed dataset missing: {raw}, {preprocessed}"
        )

    # -------------------------------------------------------------
    # 3. Load existing training-derived SSL3D plans
    # -------------------------------------------------------------

    # Find the BOBs SSL3D plans configuration.
    plan_files = sorted(preprocessed.glob("ptPlans__SSL3D_BOBs*.json"))

    # Require exactly one matching plans file to avoid ambiguity.
    if len(plan_files) != 1:
        raise RuntimeError(
            f"Expected exactly one SSL3D_BOBs plans file; found: {plan_files}"
        )

    # Load the plans and selected preprocessing configuration.
    plans_manager = PlansManager(str(plan_files[0]))
    configuration_manager = plans_manager.get_configuration(args.configuration)

    # Read the raw dataset metadata required by nnU-Net.
    dataset_json = json.loads((raw / "dataset.json").read_text(encoding="utf-8"))

    # Instantiate the preprocessor specified by the configuration.
    preprocessor = configuration_manager.preprocessor_class(verbose=False)

    # Ensure the configured preprocessor is compatible with the
    # expected nnU-Net DefaultPreprocessor implementation.
    if not isinstance(preprocessor, DefaultPreprocessor):
        raise TypeError(f"Unexpected preprocessor: {type(preprocessor)}")

    # -------------------------------------------------------------
    # 4. Define validation and test preprocessing jobs
    # -------------------------------------------------------------

    # Each entry specifies input image/label folders and output directory.
    splits = {
        "val": ("imagesVal", "labelsVal", "SSL3D_val"),
        "test": ("imagesTs", "labelsTs", "SSL3D_test"),
    }

    for split, (images_name, labels_name, destination) in splits.items():
        # Discover matching MRI/segmentation pairs.
        cases = cases_from_raw(raw, images_name, labels_name)

        # Check the expected number of cases for the selected split.
        if len(cases) != EXPECTED_COUNTS[split]:
            raise ValueError(
                f"{split}: expected {EXPECTED_COUNTS[split]} cases, got {len(cases)}"
            )

        # Create the destination directory if necessary.
        out_dir = preprocessed / destination
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"{split.upper()}: {len(cases)} cases -> {out_dir}", flush=True)

        # ---------------------------------------------------------
        # 5. Process each case using the existing SSL3D configuration
        # ---------------------------------------------------------

        for i, (case_id, images, seg_path) in enumerate(cases, start=1):
            # Base path used to construct output filenames.
            target = out_dir / case_id

            # Expected image, segmentation, and metadata outputs.
            outputs = [
                target.with_suffix(".b2nd"),
                out_dir / f"{case_id}_seg.b2nd",
                target.with_suffix(".pkl"),
            ]

            # Check which output files already exist.
            already = [p.exists() for p in outputs]

            # Skip fully processed cases unless overwriting is enabled.
            if all(already) and not args.overwrite:
                print(
                    f"  [{i}/{len(cases)}] {case_id}: already exists, skipping",
                    flush=True,
                )
                continue

            # Prevent silent reuse of partially written results.
            if any(already) and not args.overwrite:
                raise FileExistsError(
                    f"Incomplete output for {case_id}: {outputs}. Use --overwrite to redo."
                )

            print(f"  [{i}/{len(cases)}] {case_id}", flush=True)

            # Apply the official nnU-Net preprocessing procedure.
            # The existing plans determine spatial and intensity processing.
            preprocessor.run_case_save(
                str(target),
                images,
                seg_path,
                plans_manager,
                configuration_manager,
                dataset_json,
            )

        # Display completion information for the current split.
        print(f"{split.upper()} COMPLETE: {len(cases)}/{len(cases)}", flush=True)

    print("BOBs VALIDATION + TEST PREPROCESSING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
