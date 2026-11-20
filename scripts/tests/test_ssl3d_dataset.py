"""
Smoke test for the SSL3DDataset on preprocessed FeTA and BOBs MRI data.

This script validates the custom PyTorch dataset used to load
preprocessed 3D MRI volumes and their corresponding segmentation masks.

The test covers both developmental brain MRI datasets:
    - FeTA (Dataset101_FeTA)
    - BOBs (Dataset102_BOBs)

For each dataset, the following splits are checked:
    - Training
    - Validation
    - Test

Main validation steps:
    1. Load each dataset split using SSL3DDataset.
    2. Verify the expected number of MRI cases.
    3. Create a PyTorch DataLoader.
    4. Retrieve one batch of MRI images and segmentation masks.
    5. Validate tensor shapes and numerical data types.
    6. Check that image and mask values are finite.
    7. Verify that segmentation labels belong to {-1, 0, ..., 7}.
    8. Verify deterministic center-patch extraction for validation
       and test datasets.

Expected dataset sizes:
    FeTA:
        Train: 48 cases
        Val:   16 cases
        Test:  16 cases

    BOBs:
        Train: 42 MRI sessions
        Val:   15 MRI sessions
        Test:  14 MRI sessions

Expected batch shapes:
    Images: (B, 1, P, P, P)
    Masks:  (B, P, P, P)

Where:
    B = Batch size
    P = Spatial patch size

Usage (from the repository root inside the nnSSL container):
    python -m scripts.tests.test_ssl3d_dataset --patch-size 160

Optional arguments:
    --preprocessed-root   Path to preprocessed nnU-Net datasets.
    --patch-size          Cubic patch size (default: 160).
    --batch-size          Number of samples per batch (default: 1).

Important:
    - This script only reads existing preprocessed files.
    - No MRI volumes or segmentation masks are modified.
    - No model is loaded or trained.
    - Only the first batch of each split is inspected.
    - Random training patches are not checked for reproducibility.
    - The test can run without a GPU.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.ssl3d_dataset import SSL3DDataset


# ---------------------------------------------------------------------
# Expected dataset configuration
# ---------------------------------------------------------------------

# Number of MRI cases or sessions expected in each dataset split.
#
# FeTA contains one MRI case per subject in these splits.
# BOBs may contain multiple sessions from the same participant.
EXPECTED_COUNTS = {
    "feta": {"train": 48, "val": 16, "test": 16},
    "bobs": {"train": 42, "val": 15, "test": 14},
}


def check_batch(batch, dataset_name, split, patch_size, batch_size):
    """
    Validate the structure and numerical contents of one data batch.

    Parameters
    ----------
    batch : dict
        Dictionary returned by the PyTorch DataLoader.

        Expected keys:
            "image": MRI image tensor.
            "mask": Segmentation mask tensor.
            "case_id": Identifiers of the loaded MRI cases.

    dataset_name : str
        Name of the dataset being tested ("feta" or "bobs").

    split : str
        Dataset split ("train", "val", or "test").

    patch_size : int
        Expected spatial size of the cubic 3D patches.

    batch_size : int
        Actual number of samples in the retrieved batch.

    Validations
    -----------
    Image tensor:
        - Shape: (B, 1, P, P, P)
        - Data type: torch.float32
        - All intensity values must be finite.

    Segmentation tensor:
        - Shape: (B, P, P, P)
        - Data type: torch.int64
        - All values must be finite.
        - Labels must belong to {-1, 0, ..., 7}.

    Metadata:
        - The number of case identifiers must match the batch size.

    Notes
    -----
    The mask does not contain an additional channel dimension,
    making it compatible with PyTorch segmentation losses such
    as CrossEntropyLoss.

    Label -1 represents ignored voxels and must be excluded
    from the segmentation loss during training.

    Assertions raise AssertionError when an expected condition fails.
    """

    # -------------------------------------------------------------
    # 1. Retrieve tensors and case identifiers
    # -------------------------------------------------------------

    # MRI batch with shape (B, 1, D, H, W).
    images = batch["image"]

    # Segmentation batch with shape (B, D, H, W).
    masks = batch["mask"]

    # Identifiers of the MRI cases loaded in this batch.
    case_ids = batch["case_id"]

    # -------------------------------------------------------------
    # 2. Validate tensor dimensions
    # -------------------------------------------------------------

    # Verify the image batch shape.
    #
    # The first dimension is the batch size.
    # The second dimension is the single MRI channel.
    # The remaining dimensions correspond to the cubic 3D patch.
    assert images.shape == (batch_size, 1, patch_size, patch_size, patch_size), (
        images.shape
    )

    # Verify the segmentation batch shape.
    # Segmentation masks contain class indices without a channel axis.
    assert masks.shape == (batch_size, patch_size, patch_size, patch_size), masks.shape

    # -------------------------------------------------------------
    # 3. Validate tensor data types
    # -------------------------------------------------------------

    # MRI intensities must be stored as float32.
    assert images.dtype == torch.float32, images.dtype

    # Segmentation class indices must be stored as int64.
    assert masks.dtype == torch.int64, masks.dtype

    # -------------------------------------------------------------
    # 4. Validate numerical values
    # -------------------------------------------------------------

    # Ensure that MRI intensities do not contain NaN or infinity.
    assert bool(torch.isfinite(images).all()), "MRI contains NaN/Inf"

    # Ensure that segmentation masks contain only finite values.
    assert bool(torch.isfinite(masks).all()), "Mask contains NaN/Inf"

    # -------------------------------------------------------------
    # 5. Validate segmentation labels and metadata
    # -------------------------------------------------------------

    # Extract all distinct segmentation labels in the current batch.
    labels = torch.unique(masks).tolist()

    # Allowed labels:
    # -1: Ignore label
    #  0: Background
    # 1..7: Foreground segmentation classes
    assert set(labels).issubset(set(range(-1, 8))), f"Unexpected labels: {labels}"

    # Every batch sample must have a corresponding case identifier.
    assert len(case_ids) == batch_size

    # -------------------------------------------------------------
    # 6. Display batch validation results
    # -------------------------------------------------------------

    print(f"  {dataset_name.upper()} {split}: cases={list(case_ids)}", flush=True)

    # Display image and mask tensor dimensions and data types.
    print(
        f"    image={tuple(images.shape)} {images.dtype} | mask={tuple(masks.shape)} {masks.dtype}",
        flush=True,
    )

    # Display the segmentation labels present in the batch.
    print(f"    labels={labels} | finite=True", flush=True)


def main():
    """
    Run dataset smoke tests for FeTA and BOBs.

    This function:
        1. Parses command-line arguments.
        2. Initializes every dataset split.
        3. Verifies split sizes.
        4. Loads and validates one batch per split.
        5. Checks center-patch reproducibility for validation
           and test datasets.

    The test uses DataLoader with:
        - shuffle=False
        - num_workers=0

    This provides deterministic case ordering and performs
    data loading in the main process.

    No training or model inference is performed.
    """

    # -------------------------------------------------------------
    # 1. Parse command-line arguments
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(description=__doc__)

    # Root directory containing Dataset101_FeTA and Dataset102_BOBs.
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path.home() / "brain-mri-foundation-models/data/nnUNet_preprocessed",
    )

    # Spatial size of the cubic 3D patches.
    parser.add_argument("--patch-size", type=int, default=160)

    # Number of samples loaded together by the DataLoader.
    parser.add_argument("--batch-size", type=int, default=1)

    args = parser.parse_args()

    # -------------------------------------------------------------
    # 2. Validate test configuration
    # -------------------------------------------------------------

    # Patch size and batch size must be strictly positive.
    if args.patch_size <= 0 or args.batch_size <= 0:
        parser.error("--patch-size and --batch-size must be positive")

    # Display the selected dataset directory and patch configuration.
    print(f"Preprocessed root: {args.preprocessed_root}", flush=True)

    print(
        f"Patch size: {args.patch_size}^3 | batch size: {args.batch_size}", flush=True
    )

    # -------------------------------------------------------------
    # 3. Test every dataset and split
    # -------------------------------------------------------------

    # Iterate over FeTA and BOBs.
    for dataset_name, splits in EXPECTED_COUNTS.items():
        print(f"\n=== {dataset_name.upper()} ===", flush=True)

        # Test training, validation, and test splits independently.
        for split, expected_count in splits.items():
            # Initialize the dataset with the requested patch size.
            #
            # Default sampling:
            #     Training: random patches.
            #     Validation/Test: center patches.
            dataset = SSL3DDataset(
                preprocessed_root=args.preprocessed_root,
                dataset=dataset_name,
                split=split,
                patch_size=args.patch_size,
            )

            # Verify that the split contains the expected number of cases.
            assert len(dataset) == expected_count, (
                f"{dataset_name}/{split}: expected {expected_count} cases, got {len(dataset)}"
            )

            # ---------------------------------------------------------
            # 4. Create a DataLoader and retrieve one batch
            # ---------------------------------------------------------

            # Create a DataLoader without shuffling or worker processes.
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )

            # Retrieve only the first batch.
            # The script does not iterate over the complete split.
            batch = next(iter(loader))

            # The final batch may contain fewer samples than requested.
            # For this first batch, account for splits smaller than
            # the configured batch size.
            actual_batch_size = min(args.batch_size, expected_count)

            # Validate image shape, mask shape, data types,
            # numerical values, labels, and case identifiers.
            check_batch(batch, dataset_name, split, args.patch_size, actual_batch_size)

            # ---------------------------------------------------------
            # 5. Verify deterministic validation/test sampling
            # ---------------------------------------------------------

            # Center sampling should return exactly the same patch twice.
            if split != "train":
                # Load the first case again directly from the dataset.
                repeat = dataset[0]

                # Confirm that repeated loading produces the same MRI patch.
                assert torch.equal(repeat["image"], batch["image"][0])

                # Confirm that the segmentation patch is also identical.
                assert torch.equal(repeat["mask"], batch["mask"][0])

                print("    center patch reproducible=True", flush=True)

    # All dataset and batch assertions were satisfied.
    print("\nALL SSL3D DATASET TESTS PASSED!", flush=True)


if __name__ == "__main__":
    main()
