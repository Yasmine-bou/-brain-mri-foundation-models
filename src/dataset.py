
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import json
from torch.utils.data import Dataset

from src.data_utils import load_nifti

BOBS_LABEL_MAPPING = {
    0: 0,
    2: 1,
    3: 2,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
}


class BOBsDataset(Dataset):
    """
    PyTorch Dataset for BOBs resampled75.

    Loads an MRI (T1w or T2w) and its BOBs9 segmentation.
    No preprocessing is applied.
    """

    def __init__(
        self,
        data_dir: Path,
        modality: str = "T2w",
        split: str | None = None,
    ):
        self.data_dir = Path(data_dir)

        if modality not in ["T1w", "T2w"]:
            raise ValueError("modality must be 'T1w' or 'T2w'")

        self.modality = modality
        self.samples = []

        # Find all imaging sessions
        sessions = sorted(self.data_dir.glob("sub-*/ses-*"))

        # Filter sessions using predefined splits
        if split is not None:
            valid_splits = {"train", "val", "test_i", "test_ho"}

            if split not in valid_splits:
                raise ValueError(f"Invalid split: {split}")

            splits_df = pd.read_csv(
                self.data_dir / "splits_40_10_15_35.csv",
                dtype={"participant_id": str, "session_id": str},
            )

            selected = set(
                zip(
                    splits_df.loc[
                        splits_df["splits"] == split, "participant_id"
                    ],
                    splits_df.loc[
                        splits_df["splits"] == split, "session_id"
                    ],
                )
            )

            sessions = [
                session for session in sessions
                if (session.parent.name, session.name) in selected
            ]

        # Store MRI–segmentation file pairs
        for session in sessions:
            anat_dir = session / "anat"

            mri_files = list(anat_dir.glob(f"*_{modality}.nii.gz"))
            seg_files = list(
                anat_dir.glob("*_desc-bobs9_dseg.nii.gz")
            )

            if len(mri_files) != 1 or len(seg_files) != 1:
                raise ValueError(
                    f"Missing or duplicate files: {session}"
                )

            self.samples.append({
                "subject": session.parent.name,
                "session": session.name,
                "mri_path": mri_files[0],
                "seg_path": seg_files[0],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Load MRI and segmentation
        _, mri = load_nifti(sample["mri_path"])
        _, seg = load_nifti(
            sample["seg_path"], is_segmentation=True
        )

        if mri.shape != seg.shape:
            raise ValueError(
                f"Shape mismatch: {sample['subject']}"
            )

        # Convert to PyTorch tensors
        mri = torch.from_numpy(
            np.ascontiguousarray(mri, dtype=np.float32)
        ).unsqueeze(0)

        seg = torch.from_numpy(
            np.ascontiguousarray(seg, dtype=np.int64)
        )

        return {
            "image": mri,
            "mask": seg,
            "subject": sample["subject"],
            "session": sample["session"],
        }

class FeTADataset(Dataset):
    """
    PyTorch Dataset for FeTA.

    Loads one T2w MRI and its segmentation per subject.
    No preprocessing or train/val/test split is applied.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.samples = []

        subjects = sorted(self.data_dir.glob("sub-*"))

        for subject in subjects:
            if not subject.is_dir():
                continue

            anat_dir = subject / "anat"

            mri_files = list(anat_dir.glob("*_T2w.nii.gz"))
            seg_files = list(anat_dir.glob("*_dseg.nii.gz"))

            if len(mri_files) != 1 or len(seg_files) != 1:
                raise ValueError(
                    f"Missing or duplicate MRI/segmentation: {subject.name}"
                )

            self.samples.append({
                "subject": subject.name,
                "mri_path": mri_files[0],
                "seg_path": seg_files[0],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Load MRI and segmentation
        _, mri = load_nifti(sample["mri_path"])
        _, seg = load_nifti(
            sample["seg_path"], is_segmentation=True
        )

        if mri.shape != seg.shape:
            raise ValueError(
                f"Shape mismatch: {sample['subject']}"
            )

        # Convert to PyTorch tensors
        mri = torch.from_numpy(
            np.ascontiguousarray(mri, dtype=np.float32)
        ).unsqueeze(0)

        seg = torch.from_numpy(
            np.ascontiguousarray(seg, dtype=np.int64)
        )

        return {
            "image": mri,
            "mask": seg,
            "subject": sample["subject"],
        }


class BrainMRIPatchDataset(Dataset):
    """PyTorch Dataset for preprocessed 3D MRI segmentation patches.

    Loads MRI/mask pairs from .npy files and extracts aligned 3D patches.
    Supports FeTA and BOBs, using their subject-level splits.

    Training patches are sampled randomly. Validation and test patches
    are centered for reproducible pipeline checks.

    Args:
        data_dir: Directory containing preprocessed MRI/mask files.
        splits_path: JSON file containing the dataset splits.
        dataset: Either "feta" or "bobs".
        split: Split to load (e.g. "train" or "val").
        patch_size: Size of the extracted 3D patches.
    """

    def __init__(
        self,
        data_dir: Path,
        splits_path: Path,
        dataset: str = "feta",
        split: str = "train",
        patch_size: tuple[int, int, int] = (96, 96, 96),
    ):
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.split = split
        self.patch_size = tuple(patch_size)

        if dataset not in ("feta", "bobs"):
            raise ValueError("dataset must be 'feta' or 'bobs'")

        if len(self.patch_size) != 3 or any(s <= 0 for s in self.patch_size):
            raise ValueError("patch_size must contain three positive integers")

        # Load the subject IDs assigned to this split.
        with open(splits_path, encoding="utf-8") as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(f"Unknown split: {split}")

        subject_ids = set(splits[split])

        # BOBs segmentation files have a different naming convention.
        seg_suffix = (
            "_dseg.npy" if dataset == "feta"
            else "_desc-bobs9_dseg.npy"
        )

        self.samples = []

        for img_path in sorted(
            self.data_dir.glob("sub-*/**/anat/*_T2w.npy")
        ):
            subject = next(
                part for part in img_path.parts
                if part.startswith("sub-")
            )

            if subject not in subject_ids:
                continue

            seg_path = img_path.with_name(
                img_path.name.replace("_T2w.npy", seg_suffix)
            )

            if not seg_path.exists():
                raise FileNotFoundError(f"Missing mask: {seg_path}")

            self.samples.append({
                "subject": subject,
                "session": next(
                    (part for part in img_path.parts
                     if part.startswith("ses-")),
                    None,
                ),
                "mri_path": img_path,
                "seg_path": seg_path,
            })

        # Make sure every requested subject was found.
        found_ids = {s["subject"] for s in self.samples}

        if found_ids != subject_ids:
            raise ValueError(
                f"Missing subjects: {sorted(subject_ids - found_ids)}"
            )

        if not self.samples:
            raise ValueError(f"No MRI samples found for {dataset}/{split}")

    def __len__(self):
        """Return the number of MRI sessions in the split."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return one aligned MRI/mask patch as PyTorch tensors."""
        sample = self.samples[index]

        image = np.load(sample["mri_path"]).astype(np.float32)
        mask = np.load(sample["seg_path"]).astype(np.int64)

        # Convert BOBs labels to consecutive class IDs (0-7).
        if self.dataset == "bobs":
            original_labels = set(np.unique(mask).tolist())

            if not original_labels.issubset(BOBS_LABEL_MAPPING):
                raise ValueError(f"Unexpected BOBs labels: {original_labels}")

            remapped_mask = np.zeros_like(mask)

            for original, new in BOBS_LABEL_MAPPING.items():
                remapped_mask[mask == original] = new

            mask = remapped_mask


        if image.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch: {sample['mri_path'].name}"
            )

        # Add local padding only when a dimension is too small.
        padding = []
        for dim, size in zip(image.shape, self.patch_size):
            total = max(0, size - dim)
            before = total // 2
            after = total - before
            padding.append((before, after))

        image = np.pad(image, padding, mode="constant")
        mask = np.pad(mask, padding, mode="constant")

        # Random patch for training; center patch for validation/test.
        if self.split == "train":
            starts = [
                np.random.randint(0, dim - size + 1)
                for dim, size in zip(image.shape, self.patch_size)
            ]
        else:
            starts = [
                (dim - size) // 2
                for dim, size in zip(image.shape, self.patch_size)
            ]

        slices = tuple(
            slice(start, start + size)
            for start, size in zip(starts, self.patch_size)
        )

        image_patch = np.ascontiguousarray(image[slices])
        mask_patch = np.ascontiguousarray(mask[slices])

        return {
            "image": torch.from_numpy(image_patch).unsqueeze(0),
            "mask": torch.from_numpy(mask_patch),
            "subject": sample["subject"],
            "session": sample["session"] or "",
        }

