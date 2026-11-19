
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "data" / "FeTA" / "feta_2.4"
SPLITS_PATH = PROJECT_ROOT / "data" / "splits" / "feta_splits.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "nnUNet_raw" / "Dataset101_FeTA"


# ============================================================
# FETA LABELS
# ============================================================

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
    """

    # --------------------------------------------------------
    # 1. Load and validate existing splits
    # --------------------------------------------------------

    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"FeTA dataset not found: {SOURCE_DIR}"
        )

    if not SPLITS_PATH.is_file():
        raise FileNotFoundError(
            f"Splits not found: {SPLITS_PATH}"
        )

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    if set(splits.keys()) != {"train", "val", "test"}:
        raise ValueError(
            "Expected split keys: train, val, test."
        )

    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]

    all_ids = train_ids + val_ids + test_ids

    if (len(train_ids), len(val_ids), len(test_ids)) != (48, 16, 16):
        raise ValueError("Unexpected FeTA split sizes.")

    if len(all_ids) != len(set(all_ids)):
        raise ValueError(
            "Duplicate subjects or overlap between splits."
        )

    subjects_on_disk = {
        p.name
        for p in SOURCE_DIR.glob("sub-*")
        if p.is_dir()
    }

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

    source_files = {}

    for subject in sorted(all_ids):

        anat_dir = SOURCE_DIR / subject / "anat"

        images = sorted(
            anat_dir.glob("*_T2w.nii.gz")
        )

        masks = sorted(
            anat_dir.glob("*_dseg.nii.gz")
        )

        if len(images) != 1 or len(masks) != 1:
            raise ValueError(
                f"{subject}: expected exactly one MRI and one mask; "
                f"found {len(images)} MRI(s), "
                f"{len(masks)} mask(s)."
            )

        image_path = images[0]
        mask_path = masks[0]

        # Ensure MRI and mask use the same reconstruction.
        image_prefix = image_path.name.removesuffix(
            "_T2w.nii.gz"
        )

        mask_prefix = mask_path.name.removesuffix(
            "_dseg.nii.gz"
        )

        if image_prefix != mask_prefix:
            raise ValueError(
                f"{subject}: MRI and mask reconstruction mismatch."
            )

        image = nib.load(str(image_path))
        mask = nib.load(str(mask_path))

        # Check dimensions.
        if image.shape != mask.shape:
            raise ValueError(
                f"{subject}: MRI/mask shape mismatch."
            )

        # Allow tiny floating-point differences.
        # In particular, sub-011 has a ~1e-8 difference.
        if not np.allclose(
            image.affine,
            mask.affine,
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError(
                f"{subject}: MRI/mask affine mismatch."
            )

        # FeTA masks are stored as float32,
        # although their labels are integers 0 through 7.
        labels = np.asarray(mask.dataobj)

        if not np.all(np.isfinite(labels)):
            raise ValueError(
                f"{subject}: mask contains NaN or Inf."
            )

        if not np.all(
            np.isclose(
                labels,
                np.round(labels),
                rtol=0,
                atol=1e-6,
            )
        ):
            raise ValueError(
                f"{subject}: mask contains non-integer values."
            )

        unique_labels = np.unique(labels)

        if not np.all(
            np.isin(unique_labels, range(8))
        ):
            raise ValueError(
                f"{subject}: unexpected labels: {unique_labels}"
            )

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

        for subject in sorted(subject_ids):
            src_image, src_mask = source_files[subject]

            # One input channel (T2w): _0000
            dst_image = image_folder / f"{subject}_0000.nii.gz"
            dst_mask = mask_folder / f"{subject}.nii.gz"

            # Copy MRI without preprocessing.
            shutil.copy2(src_image, dst_image)

            # Convert segmentation float32 -> uint8.
            seg = nib.load(str(src_mask))

            seg_data = np.rint(
                np.asarray(seg.dataobj)
            ).astype(np.uint8)

            seg_uint8 = nib.Nifti1Image(
                seg_data,
                seg.affine,
                header=seg.header.copy(),
            )

            seg_uint8.set_data_dtype(np.uint8)

            nib.save(
                seg_uint8,
                str(dst_mask),
            )

            print(f"Converted {subject}")

    print("\n===== CONVERTING TRAIN =====")
    copy_subjects(
        train_ids,
        images_tr,
        labels_tr,
    )

    print("\n===== CONVERTING VALIDATION =====")
    copy_subjects(
        val_ids,
        images_val,
        labels_val,
    )

    print("\n===== CONVERTING TEST =====")
    copy_subjects(
        test_ids,
        images_ts,
        labels_ts,
    )

    # --------------------------------------------------------
    # 5. Create dataset.json
    # --------------------------------------------------------

    dataset_json = {
        "channel_names": {
            "0": "T2w"
        },
        "labels": FETA_LABELS,
        "numTraining": len(train_ids),
        "file_ending": ".nii.gz",
    }

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

    with open(
        OUTPUT_DIR / "splits_final.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(split_reference, f, indent=2)

    # --------------------------------------------------------
    # 7. Verify output
    # --------------------------------------------------------

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
