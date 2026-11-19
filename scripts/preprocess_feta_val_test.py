
import json
from pathlib import Path

from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


# ============================================================
# PATHS
# ============================================================

DATASET_NAME = "Dataset101_FeTA"

RAW_DIR = Path(nnUNet_raw) / DATASET_NAME
PREPROCESSED_DIR = Path(nnUNet_preprocessed) / DATASET_NAME

SPLITS_PATH = Path("/opt/project_splits/feta_splits.json")

PLANS_NAME = (
    "ptPlans__SSL3D_FeTA____"
    "Spacing__1.00_1.00_1.00___Norm__Z.json"
)

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
    """

    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n===== PREPROCESSING {split_name.upper()} =====")
    print(f"Subjects: {len(subject_ids)}")

    for subject in sorted(subject_ids):

        image_path = RAW_DIR / image_folder / f"{subject}_0000.nii.gz"
        label_path = RAW_DIR / label_folder / f"{subject}.nii.gz"

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        if not label_path.is_file():
            raise FileNotFoundError(label_path)

        output_base = output_folder / subject

        expected_files = [
            output_folder / f"{subject}.b2nd",
            output_folder / f"{subject}_seg.b2nd",
            output_folder / f"{subject}.pkl",
        ]

        existing = [p for p in expected_files if p.exists()]

        # Do not overwrite correctly processed cases.
        if len(existing) == 3:
            print(f"Already processed: {subject}")
            continue

        # Detect partial/incomplete output.
        if existing:
            raise RuntimeError(
                f"Incomplete output for {subject}: {existing}. "
                "Inspect and remove only these incomplete files "
                "before rerunning."
            )

        # Official nnU-Net preprocessing and saving.
        preprocessor.run_case_save(
            output_filename_truncated=str(output_base),
            image_files=[str(image_path)],
            seg_file=str(label_path),
            plans_manager=plans_manager,
            configuration_manager=configuration_manager,
            dataset_json=dataset_json,
            overwrite_existing=True,
        )

        for path in expected_files:
            if not path.is_file():
                raise RuntimeError(
                    f"Expected output was not created: {path}"
                )

        print(f"Processed: {subject}")

    # Final verification.
    for subject in subject_ids:
        for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
            path = output_folder / f"{subject}{suffix}"
            if not path.is_file():
                raise RuntimeError(f"Missing output: {path}")

    print(
        f"\n{split_name.upper()} COMPLETE: "
        f"{len(subject_ids)}/{len(subject_ids)}"
    )


def main():

    # --------------------------------------------------------
    # 1. Load the original splits
    # --------------------------------------------------------

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]

    assert len(train_ids) == 48
    assert len(val_ids) == 16
    assert len(test_ids) == 16

    assert len(set(train_ids + val_ids + test_ids)) == 80

    print("===== FETA SPLITS =====")
    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")

    # --------------------------------------------------------
    # 2. Load existing SSL3D plans
    # --------------------------------------------------------

    if not PLANS_PATH.is_file():
        raise FileNotFoundError(PLANS_PATH)

    if not DATASET_JSON_PATH.is_file():
        raise FileNotFoundError(DATASET_JSON_PATH)

    plans_manager = PlansManager(str(PLANS_PATH))

    configuration_manager = plans_manager.get_configuration(
        "3d_fullres"
    )

    with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    print("\n===== SSL3D CONFIGURATION =====")
    print(f"Plans: {PLANS_NAME}")
    print(f"Spacing: {configuration_manager.spacing}")
    print(
        f"Normalization: "
        f"{configuration_manager.normalization_schemes}"
    )

    # --------------------------------------------------------
    # 3. Initialize the official nnU-Net preprocessor
    # --------------------------------------------------------

    preprocessor = DefaultPreprocessor(verbose=False)

    # --------------------------------------------------------
    # 4. Validation
    # --------------------------------------------------------

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

    print("\n===== ALL DONE =====")
    print("Train: 48 (previously processed)")
    print("Validation: 16")
    print("Test: 16")
    print("\nAll splits use the same SSL3D plans.")


if __name__ == "__main__":
    main()
