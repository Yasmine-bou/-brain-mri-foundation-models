"""PyTorch Dataset for SSL3D/nnU-Net preprocessed FeTA and BOBs volumes.

Reads paired .b2nd image and _seg.b2nd label volumes from the existing
SSL3D preprocessing directories. Does not modify or preprocess source files.

The training split uses random 3D patches; validation/test use a reproducible
center patch by default. Use ``patch_size=None`` to return whole volumes (which
may have different spatial shapes, requiring batch_size=1 or custom collate).

Labels: 0=background, 1..7=foreground, -1=ignore. No label remapping here.
For training, the segmentation loss must explicitly ignore label -1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import blosc2
import numpy as np
import torch
from torch.utils.data import Dataset


DATASET_IDS = {"feta": "Dataset101_FeTA", "bobs": "Dataset102_BOBs"}
SPLIT_FOLDERS = {
    "train": "Spacing__1.00_1.00_1.00___Norm__Z",
    "val": "SSL3D_val",
    "test": "SSL3D_test",
}
ALLOWED_LABELS = frozenset(range(-1, 8))


class SSL3DDataset(Dataset):
    """Load paired MRI/segmentation volumes and extract aligned 3D patches.

    Args:
        preprocessed_root: Parent folder containing ``Dataset101_FeTA`` and/or
            ``Dataset102_BOBs`` (or, alternatively, the specific dataset folder).
        dataset: ``"feta"`` or ``"bobs"``.
        split: ``"train"``, ``"val"`` or ``"test"``.
        patch_size: Integer for cubic patches, a (D,H,W) tuple, or ``None``
            for complete variable-sized volumes.
        sampling: ``"random"``, ``"foreground"``, or ``"center"``.
            Defaults to random for train, center for val/test. Foreground mode
            selects a random foreground voxel with probability
            ``foreground_probability`` (else a random spatial crop).
        foreground_probability: Probability in [0,1] for foreground sampling.
            Used only when sampling="foreground".
        validate_labels: Check mask labels against {-1,0,...,7} on read.

    Returned dictionary:
        ``image``: float32 tensor (1,D,H,W)
        ``mask``: int64 tensor (D,H,W), suitable for CrossEntropyLoss
        ``case_id``: case name (including BOBs session identifiers)
        ``dataset`` / ``split``: descriptive strings

    Note: A patch is one training sample per case per epoch. A production
    trainer may want more than one random patch per volume per epoch.
    """

    def __init__(
        self,
        preprocessed_root: str | Path,
        dataset: Literal["feta", "bobs"],
        split: Literal["train", "val", "test"],
        patch_size: int | tuple[int, int, int] | None = 96,
        sampling: Literal["random", "foreground", "center"] | None = None,
        foreground_probability: float = 0.5,
        validate_labels: bool = True,
    ) -> None:
        if dataset not in DATASET_IDS:
            raise ValueError(f"Unknown dataset {dataset!r}; choose 'feta' or 'bobs'")
        if split not in SPLIT_FOLDERS:
            raise ValueError(f"Unknown split {split!r}; choose train, val or test")
        if sampling is None:
            sampling = "random" if split == "train" else "center"
        if sampling not in {"random", "foreground", "center"}:
            raise ValueError(f"Unknown sampling strategy: {sampling!r}")
        if not 0.0 <= foreground_probability <= 1.0:
            raise ValueError("foreground_probability must be between 0 and 1")
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3
        if patch_size is not None:
            if len(patch_size) != 3 or any(not isinstance(s, int) or s <= 0 for s in patch_size):
                raise ValueError("patch_size must be a positive int, 3-tuple, or None")
        if patch_size is None and sampling != "center":
            raise ValueError("When patch_size=None, use sampling='center'")

        root = Path(preprocessed_root)
        dataset_root = root if root.name == DATASET_IDS[dataset] else root / DATASET_IDS[dataset]
        self.folder = dataset_root / SPLIT_FOLDERS[split]
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Preprocessed split folder missing: {self.folder}")

        self.images = sorted(
            p for p in self.folder.glob("*.b2nd")
            if not p.name.endswith("_seg.b2nd")
        )
        if not self.images:
            raise FileNotFoundError(f"No MRI .b2nd files found in {self.folder}")

        masks = {p.name[:-len('_seg.b2nd')] for p in self.folder.glob('*_seg.b2nd')}
        images = {p.name[:-len('.b2nd')] for p in self.images}
        if images != masks:
            raise ValueError(
                f"MRI/mask case IDs differ: missing masks={sorted(images - masks)}, "
                f"orphan masks={sorted(masks - images)}"
            )

        self.dataset_name = dataset
        self.split = split
        self.patch_size = patch_size
        self.sampling = sampling
        self.foreground_probability = foreground_probability
        self.validate_labels = validate_labels

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        return np.asarray(blosc2.open(str(path), mode="r")[:])

    def _extract_patch(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.patch_size is None:
            return image, mask

        spatial_shape = image.shape[1:]
        patch_shape = self.patch_size
        rng = np.random.default_rng()  # independent draws; worker-safe

        if self.sampling == "foreground" and rng.random() < self.foreground_probability:
            locations = np.argwhere(mask[0] > 0)
            point = locations[rng.integers(len(locations))] if len(locations) else None
        else:
            point = None

        # Compute centered/random start coordinates in each spatial dimension.
        src_slices = []
        dst_slices = []
        for axis, (dim, target) in enumerate(zip(spatial_shape, patch_shape)):
            if point is not None:
                start = int(point[axis]) - target // 2
                start = max(0, min(start, dim - target))
            elif self.sampling == "center":
                start = max(0, (dim - target) // 2)
            else:
                start = int(rng.integers(0, dim - target + 1)) if dim > target else 0
            length = min(dim, target)
            dst_start = (target - dim) // 2 if dim < target else 0
            src_slices.append(slice(start, start + length))
            dst_slices.append(slice(dst_start, dst_start + length))

        img_patch = np.zeros((1, *patch_shape), dtype=np.float32)
        seg_patch = np.full((1, *patch_shape), -1, dtype=np.int64)
        src = (slice(None), *src_slices)
        dst = (slice(None), *dst_slices)
        img_patch[dst] = image[src]
        seg_patch[dst] = mask[src]
        return img_patch, seg_patch

    def __getitem__(self, index: int) -> dict:
        path = self.images[index]
        case_id = path.name[:-len('.b2nd')]
        mask_path = self.folder / f"{case_id}_seg.b2nd"

        image = self._load(path)
        mask = self._load(mask_path)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"{case_id}: unexpected image shape {image.shape}")
        if mask.shape != image.shape:
            raise ValueError(f"{case_id}: image/mask shapes differ: {image.shape} vs {mask.shape}")
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError(f"{case_id}: image contains non-numeric or nonfinite values")
        if not np.isfinite(mask).all() or not np.all(mask == np.rint(mask)):
            raise ValueError(f"{case_id}: segmentation contains invalid/non-integer values")
        if self.validate_labels and not set(np.unique(mask).tolist()).issubset(ALLOWED_LABELS):
            raise ValueError(f"{case_id}: unexpected mask labels {np.unique(mask)}")

        image = image.astype(np.float32, copy=False)
        mask = mask.astype(np.int64, copy=False)
        image, mask = self._extract_patch(image, mask)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask[0])),
            "case_id": case_id,
            "dataset": self.dataset_name,
            "split": self.split,
        }
