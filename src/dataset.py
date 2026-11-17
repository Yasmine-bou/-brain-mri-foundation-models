
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_utils import load_nifti


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
