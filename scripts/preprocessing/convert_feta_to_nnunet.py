"""
Convert original FeTA fetal brain MRI data into nnU-Net raw format.

This script prepares the FeTA dataset for downstream SSL3D/nnU-Net
preprocessing while preserving the predefined train/validation/test splits.

Main steps:
    1. Load and validate the predefined FeTA splits.
    2. Verify that all 80 subjects are present exactly once.
    3. Validate MRI/segmentation filenames, shapes, and affines.
    4. Check segmentation labels and numerical validity.
    5. Create nnU-Net raw image and label directories.
    6. Copy original T2-weighted MRI files unchanged.
    7. Convert segmentation mask storage from float32 to uint8.
    8. Generate dataset.json and a split reference file.
    9. Verify the number of generated files.

Expected dataset split:
    Training:    48 subjects
    Validation:  16 subjects
    Test:        16 subjects

Notes:
    - Original MRI files are copied without preprocessing.
    - Segmentation label values remain in the range 0..7.
    - Validation and test cases are stored outside imagesTr.
    - splits_final.json is saved for traceability, not as a
      directly usable nnU-Net trainer split.
    - Existing non-empty output directories are not overwritten.
"""

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================
# PATHS
# ============================================================

# Resolve the repository root from the script location.
# Note: this expression assumes the script's original location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory containing the original FeTA 2.4 subject folders.
SOURCE_DIR = PROJECT_ROOT / "data" / "FeTA" / "feta_2.4"

# Predefined and reproducible subject-level dataset splits.
SPLITS_PATH = PROJECT_ROOT / "data" / "splits" / "feta_splits.json"

# Destination directory for the nnU-Net raw FeTA dataset.
OUTPUT_DIR = PROJECT_ROOT / "data" / "nnUNet_raw" / "Dataset101_FeTA"


# ============================================================
# FETA LABELS
# ============================================================

# Anatomical segmentation classes used in FeTA.
# Background is 0; the seven foreground tissue classes are 1..7.
FETA_LABELS = {
    "background": 0,
    "external_csf": 1,
    "grey_matter": 2,
    "white_matter": 3,
    "ventricles": 4,
    "cerebellum": 5,
    "deep_grey_matter": 6,
    "brainstem": 7,
}


def convert_feta_to_nnunet():
    """
    Convert original FeTA data to nnU-Net format.

    - Keep existing train/val/test splits (48/16/16).
    - Store train, validation, and test separately.
    - Copy original MRI images without preprocessing.
    - Convert segmentation masks from float32 to uint8.
    - Generate dataset.json and a split reference file.

    The conversion validates all source MRI/mask pairs before
    creating the output dataset.

    Raises
    ------
    FileNotFoundError
        If the FeTA source directory or split file is missing.

    ValueError
        If splits, MRI/mask pairs, spatial metadata, or labels
        do not satisfy the expected constraints.

    FileExistsError
        If the destination directory is already non-empty.
    """

    # --------------------------------------------------------
    # 1. Load and validate existing splits
    # --------------------------------------------------------

    # Ensure the source dataset and split metadata are available.
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"FeTA dataset not found: {SOURCE_DIR}")

    if not SPLITS_PATH.is_file():
        raise FileNotFoundError(f"Splits not found: {SPLITS_PATH}")

    # Read the predefined training, validation, and test subject IDs.
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Require exactly the three expected split names.
    if set(splits.keys()) != {"train", "val", "test"}:
        raise ValueError("Expected split keys: train, val, test.")

    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]

    # Combine all subject IDs for cross-split validation.
    all_ids = train_ids + val_ids + test_ids

    # Verify the predefined split sizes.
    if (len(train_ids), len(val_ids), len(test_ids)) != (48, 16, 16):
        raise ValueError("Unexpected FeTA split sizes.")

    # Ensure that no subject appears more than once.
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate subjects or overlap between splits.")

    # Discover all available FeTA subject directories.
    subjects_on_disk = {p.name for p in SOURCE_DIR.glob("sub-*") if p.is_dir()}

    # Ensure that split IDs and actual subject directories match exactly.
    if set(all_ids) != subjects_on_disk:
        raise ValueError(
            "Split IDs do not match FeTA subjects.\n"
            f"Missing from splits: "
            f"{sorted(subjects_on_disk - set(all_ids))}\n"
            f"Missing from disk: "
            f"{sorted(set(all_ids) - subjects_on_disk)}"
        )

    print("===== FETA SPLITS =====")
    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")

    # --------------------------------------------------------
    # 2. Validate original MRI and segmentation files
    # --------------------------------------------------------

    # Store validated MRI/mask paths for later conversion.
    source_files = {}

    for subject in sorted(all_ids):
        # FeTA stores anatomical images inside each subject's anat folder.
        anat_dir = SOURCE_DIR / subject / "anat"

        # Find the T2-weighted MRI and corresponding segmentation mask.
        images = sorted(anat_dir.glob("*_T2w.nii.gz"))
        masks = sorted(anat_dir.glob("*_dseg.nii.gz"))

        # Require exactly one image and one segmentation per subject.
        if len(images) != 1 or len(masks) != 1:
            raise ValueError(
                f"{subject}: expected exactly one MRI and one mask; "
                f"found {len(images)} MRI(s), "
                f"{len(masks)} mask(s)."
            )

        image_path = images[0]
        mask_path = masks[0]

        # Ensure MRI and mask use the same reconstruction.
        # The filename prefix identifies the underlying reconstruction.
        image_prefix = image_path.name.removesuffix("_T2w.nii.gz")
        mask_prefix = mask_path.name.removesuffix("_dseg.nii.gz")

        if image_prefix != mask_prefix:
            raise ValueError(f"{subject}: MRI and mask reconstruction mismatch.")

        # Load NIfTI headers and image objects.
        image = nib.load(str(image_path))
        mask = nib.load(str(mask_path))

        # Check dimensions.
        # Voxel-wise segmentation requires matching spatial shapes.
        if image.shape != mask.shape:
            raise ValueError(f"{subject}: MRI/mask shape mismatch.")

        # Allow tiny floating-point differences.
        # In particular, sub-011 has a ~1e-8 difference.
        # The tolerance avoids rejecting practically identical affines.
        if not np.allclose(
            image.affine,
            mask.affine,
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError(f"{subject}: MRI/mask affine mismatch.")

        # FeTA masks are stored as float32,
        # although their labels are integers 0 through 7.
        labels = np.asarray(mask.dataobj)

        # Reject non-finite segmentation values.
        if not np.all(np.isfinite(labels)):
            raise ValueError(f"{subject}: mask contains NaN or Inf.")

        # Confirm that floating-point mask values represent integers.
        # A small tolerance accommodates numerical storage differences.
        if not np.all(
            np.isclose(
                labels,
                np.round(labels),
                rtol=0,
                atol=1e-6,
            )
        ):
            raise ValueError(f"{subject}: mask contains non-integer values.")

        # Identify the distinct segmentation labels in the volume.
        unique_labels = np.unique(labels)

        # Require all labels to belong to the FeTA label convention.
        if not np.all(np.isin(unique_labels, range(8))):
            raise ValueError(f"{subject}: unexpected labels: {unique_labels}")

        # Save validated file paths for the conversion step.
        source_files[subject] = (
            image_path,
            mask_path,
        )

    print("\nAll 80 MRI/mask pairs validated.")

    # --------------------------------------------------------
    # 3. Create nnU-Net directories
    # --------------------------------------------------------

    # Avoid overwriting an existing conversion.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {OUTPUT_DIR}\n"
            "Rename or move the existing directory before rerunning."
        )

    # Training folders (standard nnU-Net)
    images_tr = OUTPUT_DIR / "imagesTr"
    labels_tr = OUTPUT_DIR / "labelsTr"

    # Validation folders (custom folders)
    images_val = OUTPUT_DIR / "imagesVal"
    labels_val = OUTPUT_DIR / "labelsVal"

    # Test folders
    images_ts = OUTPUT_DIR / "imagesTs"
    labels_ts = OUTPUT_DIR / "labelsTs"

    # Create all image and label directories.
    for folder in (
        images_tr,
        labels_tr,
        images_val,
        labels_val,
        images_ts,
        labels_ts,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 4. Copy MRIs and convert segmentation masks
    # --------------------------------------------------------

    def copy_subjects(subject_ids, image_folder, mask_folder):
        """
        Copy MRI images and convert segmentation masks for one split.

        Parameters
        ----------
        subject_ids
            Subject identifiers belonging to the selected split.

        image_folder
            Destination directory for unchanged T2w MRI volumes.

        mask_folder
            Destination directory for uint8 segmentation masks.

        Notes
        -----
        MRI data is copied directly without resampling or
        intensity normalization.

        Segmentation values are rounded and stored as uint8.
        The original NIfTI affine and header are preserved.
        """

        for subject in sorted(subject_ids):
            src_image, src_mask = source_files[subject]

            # One input channel (T2w): _0000
            # nnU-Net requires this suffix for the first imaging channel.
            dst_image = image_folder / f"{subject}_0000.nii.gz"
            dst_mask = mask_folder / f"{subject}.nii.gz"

            # Copy MRI without preprocessing.
            shutil.copy2(src_image, dst_image)

            # Convert segmentation float32 -> uint8.
            seg = nib.load(str(src_mask))

            # Round validated integer-valued labels before conversion.
            seg_data = np.rint(np.asarray(seg.dataobj)).astype(np.uint8)

            # Create a new NIfTI segmentation while preserving metadata.
            seg_uint8 = nib.Nifti1Image(
                seg_data,
                seg.affine,
                header=seg.header.copy(),
            )

            seg_uint8.set_data_dtype(np.uint8)

            # Write the converted segmentation mask.
            nib.save(
                seg_uint8,
                str(dst_mask),
            )

            print(f"Converted {subject}")

    # Convert training cases.
    print("\n===== CONVERTING TRAIN =====")
    copy_subjects(
        train_ids,
        images_tr,
        labels_tr,
    )

    # Convert validation cases.
    print("\n===== CONVERTING VALIDATION =====")
    copy_subjects(
        val_ids,
        images_val,
        labels_val,
    )

    # Convert test cases.
    print("\n===== CONVERTING TEST =====")
    copy_subjects(
        test_ids,
        images_ts,
        labels_ts,
    )

    # --------------------------------------------------------
    # 5. Create dataset.json
    # --------------------------------------------------------

    # Define the nnU-Net input channel and segmentation classes.
    dataset_json = {
        "channel_names": {"0": "T2w"},
        "labels": FETA_LABELS,
        "numTraining": len(train_ids),
        "file_ending": ".nii.gz",
    }

    # Save the nnU-Net dataset metadata.
    with open(
        OUTPUT_DIR / "dataset.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(dataset_json, f, indent=2)

    # --------------------------------------------------------
    # 6. Save split reference
    # --------------------------------------------------------

    # For traceability only:
    # validation cases are NOT in nnU-Net imagesTr,
    # so this is not directly usable as a trainer split.
    split_reference = [
        {
            "train": sorted(train_ids),
            "val": sorted(val_ids),
        }
    ]

    # Preserve the selected subject split for later reference.
    with open(
        OUTPUT_DIR / "splits_final.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(split_reference, f, indent=2)

    # --------------------------------------------------------
    # 7. Verify output
    # --------------------------------------------------------

    # Verify that the expected number of files was generated.
    assert len(list(images_tr.glob("*.nii.gz"))) == 48
    assert len(list(labels_tr.glob("*.nii.gz"))) == 48

    assert len(list(images_val.glob("*.nii.gz"))) == 16
    assert len(list(labels_val.glob("*.nii.gz"))) == 16

    assert len(list(images_ts.glob("*.nii.gz"))) == 16
    assert len(list(labels_ts.glob("*.nii.gz"))) == 16

    print("\n===== CONVERSION COMPLETE =====")
    print(f"Output: {OUTPUT_DIR}")

    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")

    print("\ndataset.json:      created")
    print("splits_final.json: created (reference only)")

    print("\nMRI images were NOT preprocessed.")
    print("Segmentation masks were converted to uint8.")
    print("Only train cases are in imagesTr.")


if __name__ == "__main__":
    convert_feta_to_nnunet()
