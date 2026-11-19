
"""
PyTorch datasets for loading original FeTA and BOBs MRI data.

These datasets load raw NIfTI images and segmentation masks.
They do not perform preprocessing.

For SSL3D training, use the preprocessed nnU-Net .b2nd files
and the corresponding nnU-Net data loading pipeline.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_utils import load_nifti


class FeTADataset(Dataset):
    """
    Dataset for original FeTA T2w MRI and segmentation masks.

    Supports predefined train/val/test subject splits.
    No preprocessing is applied.
    """

    def __init__(
        self,
        data_dir: Path,
        splits_path: Path | None = None,
        split: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.samples = []

        if split is not None:
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid split: {split}")

            if splits_path is None:
                raise ValueError(
                    "splits_path is required when split is specified."
                )

            with open(splits_path, encoding="utf-8") as f:
                splits = json.load(f)

            selected_subjects = set(splits[split])
        else:
            selected_subjects = None

        for subject_dir in sorted(self.data_dir.glob("sub-*")):
            if not subject_dir.is_dir():
                continue

            subject = subject_dir.name

            if (
                selected_subjects is not None
                and subject not in selected_subjects
            ):
                continue

            anat_dir = subject_dir / "anat"

            mri_files = list(anat_dir.glob("*_T2w.nii.gz"))
            seg_files = list(anat_dir.glob("*_dseg.nii.gz"))

            if len(mri_files) != 1 or len(seg_files) != 1:
                raise ValueError(
                    f"Missing or duplicate MRI/segmentation: {subject}"
                )

            self.samples.append({
                "subject": subject,
                "mri_path": mri_files[0],
                "seg_path": seg_files[0],
            })

        if selected_subjects is not None:
            found = {sample["subject"] for sample in self.samples}

            if found != selected_subjects:
                raise ValueError(
                    f"Missing FeTA subjects: "
                    f"{sorted(selected_subjects - found)}"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        _, mri = load_nifti(sample["mri_path"])
        _, seg = load_nifti(
            sample["seg_path"],
            is_segmentation=True,
        )

        if mri.shape != seg.shape:
            raise ValueError(
                f"Shape mismatch: {sample['subject']}"
            )

        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(mri, dtype=np.float32)
            ).unsqueeze(0),

            "mask": torch.from_numpy(
                np.ascontiguousarray(seg, dtype=np.int64)
            ),

            "subject": sample["subject"],
        }


class BOBsDataset(Dataset):
    """
    Dataset for original BOBs MRI sessions and segmentations.

    Supports predefined participant-level train/val/test splits.
    All sessions from the same participant stay in one split.

    No preprocessing is applied.
    """

    def __init__(
        self,
        data_dir: Path,
        splits_path: Path | None = None,
        split: str | None = None,
        modality: str = "T2w",
    ):
        self.data_dir = Path(data_dir)
        self.samples = []

        if modality not in {"T1w", "T2w"}:
            raise ValueError(
                "modality must be 'T1w' or 'T2w'"
            )

        if split is not None:
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid split: {split}")

            if splits_path is None:
                raise ValueError(
                    "splits_path is required when split is specified."
                )

            with open(splits_path, encoding="utf-8") as f:
                splits = json.load(f)

            selected_participants = set(splits[split])
        else:
            selected_participants = None

        # Find all participant/session directories.
        for session_dir in sorted(
            self.data_dir.glob("sub-*/ses-*")
        ):
            if not session_dir.is_dir():
                continue

            participant = session_dir.parent.name
            session = session_dir.name

            if (
                selected_participants is not None
                and participant not in selected_participants
            ):
                continue

            anat_dir = session_dir / "anat"

            mri_files = list(
                anat_dir.glob(f"*_{modality}.nii.gz")
            )

            seg_files = list(
                anat_dir.glob("*_desc-bobs9_dseg.nii.gz")
            )

            if len(mri_files) != 1 or len(seg_files) != 1:
                raise ValueError(
                    f"Missing or duplicate MRI/segmentation: "
                    f"{participant}/{session}"
                )

            self.samples.append({
                "subject": participant,
                "session": session,
                "mri_path": mri_files[0],
                "seg_path": seg_files[0],
            })

        if selected_participants is not None:
            found = {sample["subject"] for sample in self.samples}

            if found != selected_participants:
                raise ValueError(
                    f"Missing BOBs participants: "
                    f"{sorted(selected_participants - found)}"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        _, mri = load_nifti(sample["mri_path"])
        _, seg = load_nifti(
            sample["seg_path"],
            is_segmentation=True,
        )

        if mri.shape != seg.shape:
            raise ValueError(
                f"Shape mismatch: "
                f"{sample['subject']}/{sample['session']}"
            )

        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(mri, dtype=np.float32)
            ).unsqueeze(0),

            "mask": torch.from_numpy(
                np.ascontiguousarray(seg, dtype=np.int64)
            ),

            "subject": sample["subject"],
            "session": sample["session"],
        }
