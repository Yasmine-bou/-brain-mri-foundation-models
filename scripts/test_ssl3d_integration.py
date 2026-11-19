"""Integration smoke test: real FeTA/BOBs SSL3D patches -> frozen ResEncL.

Run on an Izar GPU compute node from the repository root:
    apptainer exec --nv --bind "$HOME:$HOME" ~/containers/ssl3d_nnunet.sif \
        python -m scripts.test_ssl3d_integration --patch-size 160

Reads one validation MRI/mask patch per dataset and runs both encoder modes.
Does not train, update weights, save features, or modify dataset files.
"""

import argparse
import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.frozen_encoder import make_frozen_encoder
from src.ssl3d_dataset import SSL3DDataset


CHANNELS = (32, 64, 128, 256, 320, 320)


def test_encoder(mode: str, image: torch.Tensor, checkpoint: Path | None) -> None:
    """Run a single frozen encoder and validate multiscale outputs."""
    print(f"  --- {mode.upper()} ---", flush=True)
    encoder = make_frozen_encoder(
        mode=mode,
        checkpoint_path=checkpoint if mode == "pretrained" else None,
    ).to("cuda")
    assert not encoder.training, "Encoder must be in eval mode"
    assert not any(p.requires_grad for p in encoder.parameters()), "Encoder is not frozen"

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        features = encoder(image)

    if isinstance(features, torch.Tensor):
        features = [features]
    assert len(features) == 6, f"Expected 6 feature maps, got {len(features)}"

    batch_size, _, depth, height, width = image.shape
    assert all(s % 32 == 0 for s in (depth, height, width)), (
        "This test expects a patch size divisible by 32"
    )
    for index, (feature, channels) in enumerate(zip(features, CHANNELS)):
        factor = 2 ** index
        expected = (batch_size, channels, depth // factor, height // factor, width // factor)
        assert tuple(feature.shape) == expected, (
            f"Feature {index}: expected {expected}, got {tuple(feature.shape)}"
        )
        assert bool(torch.isfinite(feature).all()), f"Feature {index} contains NaN/Inf"
        print(f"    Feature {index}: {tuple(feature.shape)} | finite=True", flush=True)

    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  {mode.upper()} PASSED! | peak GPU tensor memory: {peak_gib:.2f} GiB", flush=True)

    del features, encoder
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessed-root", type=Path,
        default=Path.home() / "brain-mri-foundation-models/data/nnUNet_preprocessed",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path.home() / "ssl3d_checkpoints/checkpoint_best.pth",
    )
    parser.add_argument("--patch-size", type=int, default=160)
    args = parser.parse_args()

    if args.patch_size <= 0 or args.patch_size % 32 != 0:
        parser.error("--patch-size must be a positive multiple of 32 (e.g. 160)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run inside Apptainer --nv on an allocated GPU node")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Patch size: {args.patch_size}^3 | batch size: 1", flush=True)

    for name in ("feta", "bobs"):
        print(f"\n=== {name.upper()} VALIDATION ===", flush=True)
        dataset = SSL3DDataset(
            preprocessed_root=args.preprocessed_root,
            dataset=name,
            split="val",
            patch_size=args.patch_size,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        image, mask = batch["image"], batch["mask"]
        assert image.shape == (1, 1, args.patch_size, args.patch_size, args.patch_size)
        assert mask.shape == (1, args.patch_size, args.patch_size, args.patch_size)
        assert image.dtype == torch.float32 and mask.dtype == torch.int64
        assert bool(torch.isfinite(image).all()) and bool(torch.isfinite(mask).all())
        assert set(torch.unique(mask).tolist()).issubset(set(range(-1, 8)))
        print(f"  Case: {batch['case_id'][0]}", flush=True)
        print(f"  Image: {tuple(image.shape)} | mask: {tuple(mask.shape)}", flush=True)

        image = image.to("cuda")
        for mode in ("pretrained", "random"):
            test_encoder(mode, image, args.checkpoint)
        del image, mask, batch, loader, dataset
        gc.collect()
        torch.cuda.empty_cache()

    print("\nALL SSL3D INTEGRATION TESTS PASSED!", flush=True)
    print("Note: forward-only validation; segmentation head/training not tested.", flush=True)


if __name__ == "__main__":
    main()
