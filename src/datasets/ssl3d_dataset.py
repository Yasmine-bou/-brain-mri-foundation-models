"""
PyTorch Dataset for 3D fetal brain MRI segmentation (FeTA and BOBs).

This module provides a custom PyTorch Dataset for loading MRI volumes and
their corresponding segmentation masks from the existing SSL3D/nnU-Net
preprocessing directories.

The dataset supports:
    - FeTA (Dataset101_FeTA) and BOBs (Dataset102_BOBs).
    - Training, validation, and test splits.
    - Random, foreground-based, and center patch sampling.
    - Configurable 3D patch sizes or complete MRI volumes.
    - Validation of image shapes, numerical values, and segmentation labels.
    - Direct integration with PyTorch DataLoader.

Data format:
    MRI volumes and segmentation masks are stored as paired .b2nd files.

    Example:
        case001.b2nd       -> MRI volume
        case001_seg.b2nd   -> Segmentation mask

    Both arrays are expected to have shape (1, D, H, W), where:
        1 : Single MRI channel
        D : Depth
        H : Height
        W : Width

Segmentation labels:
    -1     : Ignore label (excluded from the segmentation loss)
     0     : Background
     1..7  : Foreground anatomical classes

Important:
    - This module only loads and samples existing preprocessed volumes.
    - No additional preprocessing or label remapping is performed.
    - Original dataset files are never modified.
    - The training loss must ignore voxels labeled -1
      (e.g., CrossEntropyLoss(ignore_index=-1)).

Sampling strategy:
    Training:
        Random 3D patches by default.

    Validation / Test:
        Deterministic center patches by default.

    Setting patch_size=None returns the complete volume. Since volumes
    may have different spatial dimensions, this generally requires
    batch_size=1 or a custom DataLoader collate function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import blosc2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------

# Map user-friendly dataset names to their preprocessed folder names.
DATASET_IDS = {
    "feta": "Dataset101_FeTA",
    "bobs": "Dataset102_BOBs",
}

# Map each data split to its corresponding preprocessing directory.
#
# Training data is stored in the normalized and resampled directory,
# while validation and test data use their dedicated SSL3D folders.
SPLIT_FOLDERS = {
    "train": "Spacing__1.00_1.00_1.00___Norm__Z",
    "val": "SSL3D_val",
    "test": "SSL3D_test",
}

# Allowed segmentation labels:
# -1 = ignored voxel, 0 = background, 1..7 = foreground classes.
ALLOWED_LABELS = frozenset(range(-1, 8))


# ---------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------


class SSL3DDataset(Dataset):
    """
    PyTorch Dataset for paired 3D MRI volumes and segmentation masks.

    Each sample consists of an MRI volume and its corresponding
    voxel-wise segmentation mask.

    During training, the dataset can extract random or foreground-based
    3D patches. For validation and testing, center patches are used
    by default to ensure reproducible evaluation.

    Parameters
    ----------
    preprocessed_root : str or Path
        Root directory containing the preprocessed datasets.

        This can be either:
            - The parent directory containing Dataset101_FeTA
              and Dataset102_BOBs.
            - The specific dataset directory itself.

    dataset : {"feta", "bobs"}
        Dataset to load.

    split : {"train", "val", "test"}
        Data split to use.

    patch_size : int, tuple[int, int, int], or None, default=160
        Spatial size of the extracted 3D patches.

        - int: cubic patches of shape (size, size, size).
        - tuple: patches of shape (D, H, W).
        - None: return complete volumes without cropping.

        If a volume is smaller than the requested patch, it is padded.

    sampling : {"random", "foreground", "center"} or None
        Strategy used to select the spatial patch.

        - "random":
            Select a random crop from the volume.

        - "foreground":
            With probability foreground_probability, select a
            random voxel belonging to a foreground class (label > 0)
            and extract a patch around it. Otherwise, use a random crop.

        - "center":
            Extract a patch centered in the volume.

        If None:
            - Training uses "random".
            - Validation and test use "center".

    foreground_probability : float, default=0.5
        Probability of choosing foreground-based sampling when
        sampling="foreground".

        Must be between 0 and 1.

    validate_labels : bool, default=True
        Whether to verify that segmentation masks contain only
        the allowed labels {-1, 0, 1, ..., 7}.

    Returns
    -------
    dict
        Dictionary containing:

        "image" : torch.FloatTensor
            MRI image with shape (1, D, H, W).

        "mask" : torch.LongTensor
            Segmentation mask with shape (D, H, W).

            The channel dimension is removed so that the mask
            can be passed directly to a segmentation loss such
            as torch.nn.CrossEntropyLoss.

        "case_id" : str
            Unique case identifier, including session information
            when present in BOBs filenames.

        "dataset" : str
            Dataset name ("feta" or "bobs").

        "split" : str
            Data split ("train", "val", or "test").

    Notes
    -----
    One dataset index corresponds to one MRI case.

    During training, each call to __getitem__ can return a different
    random patch from the same volume.

    However, one pass through the dataset produces only one patch
    per case. Training pipelines requiring multiple patches per
    volume per epoch may need a dedicated sampling strategy.
    """

    def __init__(
        self,
        preprocessed_root: str | Path,
        dataset: Literal["feta", "bobs"],
        split: Literal["train", "val", "test"],
        patch_size: int | tuple[int, int, int] | None = 160,
        sampling: Literal["random", "foreground", "center"] | None = None,
        foreground_probability: float = 0.5,
        validate_labels: bool = True,
    ) -> None:
        # -------------------------------------------------------------
        # 1. Validate dataset and split arguments
        # -------------------------------------------------------------

        # Ensure that the requested dataset is supported.
        if dataset not in DATASET_IDS:
            raise ValueError(f"Unknown dataset {dataset!r}; choose 'feta' or 'bobs'")

        # Ensure that the requested split exists in the configuration.
        if split not in SPLIT_FOLDERS:
            raise ValueError(f"Unknown split {split!r}; choose train, val or test")

        # Select the default sampling strategy according to the split.
        # Random sampling is used for training, while center sampling
        # ensures deterministic validation and test patches.
        if sampling is None:
            sampling = "random" if split == "train" else "center"

        # Verify that the selected sampling strategy is supported.
        if sampling not in {"random", "foreground", "center"}:
            raise ValueError(f"Unknown sampling strategy: {sampling!r}")

        # Foreground sampling probability must lie in [0, 1].
        if not 0.0 <= foreground_probability <= 1.0:
            raise ValueError("foreground_probability must be between 0 and 1")

        # -------------------------------------------------------------
        # 2. Validate and standardize the patch size
        # -------------------------------------------------------------

        # Convert a single integer into a cubic (D, H, W) patch size.
        # Example: 160 -> (160, 160, 160).
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3

        # When patch extraction is enabled, ensure that all spatial
        # dimensions are strictly positive integers.
        if patch_size is not None:
            if len(patch_size) != 3 or any(
                not isinstance(s, int) or s <= 0 for s in patch_size
            ):
                raise ValueError("patch_size must be a positive int, 3-tuple, or None")

        # Whole-volume loading does not require spatial sampling.
        # Therefore, only the center sampling mode is accepted.
        if patch_size is None and sampling != "center":
            raise ValueError("When patch_size=None, use sampling='center'")

        # -------------------------------------------------------------
        # 3. Locate the preprocessed dataset directory
        # -------------------------------------------------------------

        root = Path(preprocessed_root)

        # Accept either the parent preprocessing directory or the
        # dataset-specific directory directly.
        dataset_root = (
            root if root.name == DATASET_IDS[dataset] else root / DATASET_IDS[dataset]
        )

        # Select the directory corresponding to the requested split.
        self.folder = dataset_root / SPLIT_FOLDERS[split]

        # Fail immediately if the expected directory does not exist.
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Preprocessed split folder missing: {self.folder}")

        # -------------------------------------------------------------
        # 4. Discover MRI volumes and verify image-mask pairing
        # -------------------------------------------------------------

        # Collect all MRI .b2nd files.
        # Segmentation files are excluded using the "_seg.b2nd" suffix.
        #
        # Sorting ensures a consistent dataset index order.
        self.images = sorted(
            p for p in self.folder.glob("*.b2nd") if not p.name.endswith("_seg.b2nd")
        )

        # The dataset must contain at least one MRI volume.
        if not self.images:
            raise FileNotFoundError(f"No MRI .b2nd files found in {self.folder}")

        # Extract case identifiers from segmentation filenames.
        # Example: "case001_seg.b2nd" -> "case001".
        masks = {p.name[: -len("_seg.b2nd")] for p in self.folder.glob("*_seg.b2nd")}

        # Extract case identifiers from MRI filenames.
        # Example: "case001.b2nd" -> "case001".
        images = {p.name[: -len(".b2nd")] for p in self.images}

        # Every MRI must have exactly one corresponding mask.
        # This also detects segmentation masks without MRI volumes.
        if images != masks:
            raise ValueError(
                f"MRI/mask case IDs differ: "
                f"missing masks={sorted(images - masks)}, "
                f"orphan masks={sorted(masks - images)}"
            )

        # -------------------------------------------------------------
        # 5. Store dataset configuration
        # -------------------------------------------------------------

        self.dataset_name = dataset
        self.split = split
        self.patch_size = patch_size
        self.sampling = sampling
        self.foreground_probability = foreground_probability
        self.validate_labels = validate_labels

    def __len__(self) -> int:
        """
        Return the number of MRI cases in the selected split.

        Each case is counted once, independently of the number
        of possible patches that can be extracted from its volume.
        """

        return len(self.images)

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        """
        Load a preprocessed .b2nd volume into a NumPy array.

        Parameters
        ----------
        path : Path
            Path to an MRI image or segmentation mask.

        Returns
        -------
        np.ndarray
            Array containing the complete stored volume.

        Notes
        -----
        The file is opened in read-only mode.

        This method does not modify the original preprocessed data.
        """

        # Read the complete Blosc2 array into memory.
        return np.asarray(blosc2.open(str(path), mode="r")[:])

    def _extract_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract a spatially aligned 3D patch from an MRI and its mask.

        The same crop coordinates are applied to both arrays so
        that each MRI voxel remains aligned with its segmentation label.

        Parameters
        ----------
        image : np.ndarray
            MRI volume with shape (1, D, H, W).

        mask : np.ndarray
            Segmentation volume with shape (1, D, H, W).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Extracted image patch and segmentation patch.

            Image shape: (1, patch_D, patch_H, patch_W)
            Mask shape:  (1, patch_D, patch_H, patch_W)

        Sampling
        --------
        random:
            Select independently randomized crop positions along
            the three spatial axes.

        foreground:
            With the configured probability, choose a random
            foreground voxel and position the patch around it.

        center:
            Select a centered crop along each spatial dimension.

        Padding
        -------
        If the input volume is smaller than the requested patch:

            - Image padding uses 0.
            - Segmentation padding uses -1 (ignore label).

        The -1 padding prevents artificial background voxels
        from contributing to the segmentation loss.
        """

        # -------------------------------------------------------------
        # 1. Return the entire volume if patch extraction is disabled
        # -------------------------------------------------------------

        if self.patch_size is None:
            return image, mask

        # Image shape is (C, D, H, W).
        # Only the spatial dimensions are used for cropping.
        spatial_shape = image.shape[1:]

        # Requested output patch dimensions (D, H, W).
        patch_shape = self.patch_size

        # -------------------------------------------------------------
        # 2. Optionally select a foreground voxel
        # -------------------------------------------------------------

        # PyTorch random operations follow the DataLoader worker seed,
        # allowing reproducible sampling when seeds are controlled.
        point = None

        if (
            self.sampling == "foreground"
            and torch.rand(1).item() < self.foreground_probability
        ):
            # Find all spatial coordinates belonging to foreground
            # segmentation classes (labels strictly greater than 0).
            #
            # The channel dimension is removed before searching.
            locations = np.argwhere(mask[0] > 0)

            # If foreground voxels exist, select one randomly.
            # Otherwise, the method falls back to random cropping.
            if len(locations) > 0:
                idx = torch.randint(len(locations), (1,)).item()

                point = locations[idx]

        # -------------------------------------------------------------
        # 3. Compute crop and padding coordinates
        # -------------------------------------------------------------

        # Source slices identify the region extracted from the volume.
        # Destination slices identify where this region is placed
        # in the fixed-size output patch.
        src_slices = []
        dst_slices = []

        # Process depth, height, and width independently.
        for axis, (dim, target) in enumerate(zip(spatial_shape, patch_shape)):
            if point is not None:
                # Foreground sampling:
                # Center the patch around the selected foreground voxel.
                start = int(point[axis]) - target // 2

                # Restrict the crop to valid positions inside the volume.
                start = max(0, min(start, dim - target))

            elif self.sampling == "center":
                # Center sampling:
                # Extract a spatially centered patch.
                start = max(0, (dim - target) // 2)

            else:
                # Random sampling:
                # Select a random valid starting coordinate.
                #
                # If the dimension is smaller than the patch,
                # cropping begins at zero and padding is added later.
                start = (
                    torch.randint(0, dim - target + 1, (1,)).item()
                    if dim > target
                    else 0
                )

            # Number of original voxels copied along this axis.
            # It cannot exceed either the input dimension
            # or the requested patch dimension.
            length = min(dim, target)

            # When the volume is smaller than the patch, center
            # the copied voxels inside the output padded patch.
            dst_start = (target - dim) // 2 if dim < target else 0

            # Store the spatial slice in the original volume.
            src_slices.append(slice(start, start + length))

            # Store the corresponding slice in the output patch.
            dst_slices.append(slice(dst_start, dst_start + length))

        # -------------------------------------------------------------
        # 4. Allocate output patches with appropriate padding
        # -------------------------------------------------------------

        # Initialize the MRI patch with zeros.
        # Missing spatial regions remain zero-padded.
        img_patch = np.zeros(
            (1, *patch_shape),
            dtype=np.float32,
        )

        # Initialize the segmentation patch with ignore labels.
        # Padded voxels must not contribute to the segmentation loss.
        seg_patch = np.full(
            (1, *patch_shape),
            -1,
            dtype=np.int64,
        )

        # Include the channel dimension in both slice tuples.
        # slice(None) keeps the complete channel dimension.
        src = (slice(None), *src_slices)
        dst = (slice(None), *dst_slices)

        # -------------------------------------------------------------
        # 5. Copy aligned image and segmentation regions
        # -------------------------------------------------------------

        # Apply identical spatial cropping coordinates to the MRI
        # and segmentation mask to preserve voxel-wise alignment.
        img_patch[dst] = image[src]
        seg_patch[dst] = mask[src]

        return img_patch, seg_patch

    def __getitem__(self, index: int) -> dict:
        """
        Load, validate, and return one MRI segmentation sample.

        This method is called by PyTorch DataLoader whenever
        a dataset sample is requested.

        Parameters
        ----------
        index : int
            Index of the MRI case to load.

        Returns
        -------
        dict
            Dictionary containing:

            image:
                Float32 MRI tensor with shape (1, D, H, W).

            mask:
                Int64 segmentation tensor with shape (D, H, W).

            case_id:
                Identifier of the MRI case.

            dataset:
                Name of the source dataset.

            split:
                Name of the selected data split.

        Processing steps
        ----------------
        1. Identify the MRI and segmentation file paths.
        2. Load both volumes from .b2nd files.
        3. Validate shapes, values, and segmentation labels.
        4. Convert arrays to the required numerical types.
        5. Extract aligned 3D patches if requested.
        6. Convert NumPy arrays to PyTorch tensors.
        7. Return the sample and its metadata.
        """

        # -------------------------------------------------------------
        # 1. Identify the MRI and segmentation files
        # -------------------------------------------------------------

        # Retrieve the MRI file corresponding to this dataset index.
        path = self.images[index]

        # Extract the case identifier by removing the file extension.
        case_id = path.name[: -len(".b2nd")]

        # Construct the corresponding segmentation mask filename.
        mask_path = self.folder / f"{case_id}_seg.b2nd"

        # -------------------------------------------------------------
        # 2. Load MRI and segmentation volumes
        # -------------------------------------------------------------

        image = self._load(path)
        mask = self._load(mask_path)

        # -------------------------------------------------------------
        # 3. Validate image and segmentation shapes
        # -------------------------------------------------------------

        # The MRI must have four dimensions:
        # (Channels, Depth, Height, Width).
        #
        # This implementation expects exactly one MRI channel.
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"{case_id}: unexpected image shape {image.shape}")

        # The segmentation mask must match the MRI shape exactly.
        # This guarantees spatial alignment before patch extraction.
        if mask.shape != image.shape:
            raise ValueError(
                f"{case_id}: image/mask shapes differ: {image.shape} vs {mask.shape}"
            )

        # -------------------------------------------------------------
        # 4. Validate numerical values and segmentation labels
        # -------------------------------------------------------------

        # Check that MRI intensities are numeric and finite.
        # NaN and infinity values would cause problems during training.
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError(
                f"{case_id}: image contains non-numeric or nonfinite values"
            )

        # Segmentation labels must be finite integer-valued numbers.
        # Fractional labels are invalid for semantic segmentation.
        if not np.isfinite(mask).all() or not np.all(mask == np.rint(mask)):
            raise ValueError(
                f"{case_id}: segmentation contains invalid/non-integer values"
            )

        # Optionally verify that all labels belong to the expected set:
        # {-1, 0, 1, 2, 3, 4, 5, 6, 7}.
        #
        # No label remapping is performed.
        if self.validate_labels and not set(np.unique(mask).tolist()).issubset(
            ALLOWED_LABELS
        ):
            raise ValueError(f"{case_id}: unexpected mask labels {np.unique(mask)}")

        # -------------------------------------------------------------
        # 5. Convert arrays to the required numerical types
        # -------------------------------------------------------------

        # MRI intensities are represented using float32.
        # This is the standard numerical type for PyTorch model inputs.
        image = image.astype(np.float32, copy=False)

        # Segmentation labels use int64, as required by
        # PyTorch CrossEntropyLoss.
        mask = mask.astype(np.int64, copy=False)

        # -------------------------------------------------------------
        # 6. Extract aligned 3D patches
        # -------------------------------------------------------------

        # Apply the configured sampling strategy.
        # If patch_size=None, the entire volume is returned.
        image, mask = self._extract_patch(image, mask)

        # -------------------------------------------------------------
        # 7. Convert arrays to PyTorch tensors and return sample
        # -------------------------------------------------------------

        return {
            # MRI tensor: (1, D, H, W), dtype=torch.float32.
            #
            # ascontiguousarray ensures a compatible memory layout
            # before conversion from NumPy to PyTorch.
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            # Segmentation tensor: (D, H, W), dtype=torch.int64.
            #
            # Remove the singleton channel dimension because
            # CrossEntropyLoss expects class indices without
            # an additional channel dimension.
            "mask": torch.from_numpy(np.ascontiguousarray(mask[0])),
            # Metadata used to identify and track each sample.
            "case_id": case_id,
            "dataset": self.dataset_name,
            "split": self.split,
        }
