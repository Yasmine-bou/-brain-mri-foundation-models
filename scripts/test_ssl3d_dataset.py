"""Smoke-test SSL3DDataset on preprocessed FeTA and BOBs MRI/masks.

Run from the repository root inside the nnSSL Apptainer container:
    python -m scripts.test_ssl3d_dataset --patch-size 160

This test only reads files. It does not modify the datasets or train a model.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.ssl3d_dataset import SSL3DDataset


EXPECTED_COUNTS = {
    "feta": {"train": 48, "val": 16, "test": 16},
    "bobs": {"train": 42, "val": 15, "test": 14},
}


def check_batch(batch, dataset_name, split, patch_size, batch_size):
    images = batch["image"]
    masks = batch["mask"]
    case_ids = batch["case_id"]

    assert images.shape == (batch_size, 1, patch_size, patch_size, patch_size), images.shape
    assert masks.shape == (batch_size, patch_size, patch_size, patch_size), masks.shape
    assert images.dtype == torch.float32, images.dtype
    assert masks.dtype == torch.int64, masks.dtype
    assert bool(torch.isfinite(images).all()), "MRI contains NaN/Inf"
    assert bool(torch.isfinite(masks).all()), "Mask contains NaN/Inf"

    labels = torch.unique(masks).tolist()
    assert set(labels).issubset(set(range(-1, 8))), f"Unexpected labels: {labels}"
    assert len(case_ids) == batch_size

    print(f"  {dataset_name.upper()} {split}: cases={list(case_ids)}", flush=True)
    print(f"    image={tuple(images.shape)} {images.dtype} | mask={tuple(masks.shape)} {masks.dtype}", flush=True)
    print(f"    labels={labels} | finite=True", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path.home() / "brain-mri-foundation-models/data/nnUNet_preprocessed",
    )
    parser.add_argument("--patch-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.patch_size <= 0 or args.batch_size <= 0:
        parser.error("--patch-size and --batch-size must be positive")

    print(f"Preprocessed root: {args.preprocessed_root}", flush=True)
    print(f"Patch size: {args.patch_size}^3 | batch size: {args.batch_size}", flush=True)

    for dataset_name, splits in EXPECTED_COUNTS.items():
        print(f"\n=== {dataset_name.upper()} ===", flush=True)
        for split, expected_count in splits.items():
            dataset = SSL3DDataset(
                preprocessed_root=args.preprocessed_root,
                dataset=dataset_name,
                split=split,
                patch_size=args.patch_size,
            )
            assert len(dataset) == expected_count, (
                f"{dataset_name}/{split}: expected {expected_count} cases, got {len(dataset)}"
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )
            batch = next(iter(loader))
            actual_batch_size = min(args.batch_size, expected_count)
            check_batch(batch, dataset_name, split, args.patch_size, actual_batch_size)

            # Center sampling should return exactly the same patch twice.
            if split != "train":
                repeat = dataset[0]
                assert torch.equal(repeat["image"], batch["image"][0])
                assert torch.equal(repeat["mask"], batch["mask"][0])
                print("    center patch reproducible=True", flush=True)

    print("\nALL SSL3D DATASET TESTS PASSED!", flush=True)


if __name__ == "__main__":
    main()
