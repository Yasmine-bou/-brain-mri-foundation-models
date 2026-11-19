"""Smoke test of frozen pretrained/random SSL3D ResEncL encoders on FeTA.

Run inside the nnSSL Apptainer image on an Izar GPU compute node.
This is NOT training and does not save feature maps.
"""

import argparse
from pathlib import Path

import blosc2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from nnunetv2.utilities.get_network_via_name import get_network_from_name


class FeTAPatchDataset(Dataset):
    """Load matching preprocessed MRI/mask pairs and center crop/pad to a patch."""

    def __init__(self, split_dir: Path, patch_size: int = 96, max_cases: int = 3):
        self.patch_size = patch_size
        images = sorted(p for p in split_dir.glob("sub-*.b2nd") if not p.name.endswith("_seg.b2nd"))
        self.images = images[:max_cases]
        if not self.images:
            raise FileNotFoundError(f"No FeTA MRI .b2nd files in {split_dir}")

    def __len__(self):
        return len(self.images)

    @staticmethod
    def center_crop_or_pad(array: np.ndarray, size: int, fill: float) -> np.ndarray:
        if array.ndim != 4 or array.shape[0] != 1:
            raise ValueError(f"Expected (1,D,H,W), got {array.shape}")
        result = np.full((1, size, size, size), fill, dtype=array.dtype)
        src, dst = [], []
        for dim in array.shape[1:]:
            n = min(dim, size)
            src_start = max((dim - size) // 2, 0)
            dst_start = max((size - dim) // 2, 0)
            src.append(slice(src_start, src_start + n))
            dst.append(slice(dst_start, dst_start + n))
        result[(slice(None), *dst)] = array[(slice(None), *src)]
        return result

    def __getitem__(self, index):
        path = self.images[index]
        seg_path = path.with_name(path.name.replace(".b2nd", "_seg.b2nd"))
        image = np.asarray(blosc2.open(str(path), mode="r")[:])
        mask = np.asarray(blosc2.open(str(seg_path), mode="r")[:])
        if image.shape != mask.shape:
            raise ValueError(f"MRI/mask mismatch: {path.name}: {image.shape} vs {mask.shape}")
        image = self.center_crop_or_pad(image, self.patch_size, 0).astype(np.float32)
        mask = self.center_crop_or_pad(mask, self.patch_size, -1).astype(np.int64)
        return torch.from_numpy(image), torch.from_numpy(mask), path.stem


def make_frozen_encoder(mode: str, checkpoint_path: Path, seed: int = 42):
    """Return official ResEncL encoder with pretrained or reproducible random weights."""
    if mode not in {"pretrained", "random"}:
        raise ValueError(mode)
    if mode == "random":
        torch.manual_seed(seed)

    network = get_network_from_name(
        arch_class_name="ResEncL",
        input_channels=1,
        output_channels=8,  # Full decoder is not used; encoder does not depend on this.
        allow_init=(mode == "random"),
        deep_supervision=False,
    )
    encoder = network.encoder
    if mode == "pretrained":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        source = checkpoint["network_weights"]
        weights = {
            key[len("encoder."):]: value
            for key, value in source.items()
            if key.startswith("encoder.")
        }
        encoder.load_state_dict(weights, strict=True)
        print(f"  Loaded {len(weights)} pretrained encoder state entries", flush=True)
        del checkpoint, source, weights

    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Preprocessed FeTA split directory, e.g. SSL3D_val")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="SSL3D checkpoint_best.pth")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--num-cases", type=int, default=3)