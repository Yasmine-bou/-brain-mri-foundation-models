"""Helpers for converting BOBs MRI and BOBs9 segmentations to nnU-Net.

A participant may have several sessions. Never split at scan level.
"""

import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np

# Original BOBs9 voxel codes -> consecutive nnU-Net class indices.
# These are numeric IDs, NOT verified anatomical names.
LABEL_MAP = {0: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7}
EXPECTED_COUNTS = {"train": 42, "val": 15, "test": 14}


def subject_id(entry):
    """Extract a normalized BOBs participant ID from a split entry.

    Accepts a string or dictionary and ensures the ID starts with 'sub-'.

    Args:
        entry: Participant identifier or dictionary containing one.

    Returns:
        Normalized participant ID, e.g. 'sub-116056'.
    """
    if isinstance(entry, str):
        value = entry
    elif isinstance(entry, dict):
        value = next(
            (
                entry[k]
                for k in (
                    "participant_id",
                    "subject_id",
                    "participant",
                    "subject",
                    "id",
                )
                if k in entry
            ),
            None,
        )
    else:
        value = None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Unsupported split entry: {entry!r}")
    value = value.strip().split("/")[0]
    return value if value.startswith("sub-") else f"sub-{value}"


def load_splits(path: Path):
    """Load and validate participant-level train/val/test splits.

    Ensures participants are unique within each split and that no
    participant appears in multiple splits.

    Args:
        path: Path to the JSON split file.

    Returns:
        Dictionary mapping split names to sets of participant IDs.
    """
    splits = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(splits) != {"train", "val", "test"}:
        raise ValueError(f"Expected train/val/test split keys, got {list(splits)}")
    groups = {key: {subject_id(x) for x in splits[key]} for key in splits}
    if any(len(groups[k]) != len(splits[k]) for k in groups):
        raise ValueError("Duplicate subject IDs in splits")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if groups[a] & groups[b]:
            raise ValueError(
                f"Subject leakage between {a} and {b}: {groups[a] & groups[b]}"
            )
    return groups


def matching_segmentation(mri: Path):
    """Find the matching BOBs9 mask for one T2w scan, requiring a unique match.

    BIDS-like naming is assumed: same scan prefix, replacing _T2w with
    _desc-bobs9_dseg. A descriptive error is raised if naming differs.
    """
    name = mri.name
    if not name.endswith("_T2w.nii.gz"):
        raise ValueError(f"Not a T2w MRI: {mri}")
    prefix = name[: -len("_T2w.nii.gz")]
    # First prefer the exact corresponding BIDS-style filename.
    exact = mri.with_name(prefix + "_desc-bobs9_dseg.nii.gz")
    if exact.exists():
        return exact
    # May live elsewhere within the same session/anat directory.
    candidates = list(mri.parent.glob(prefix + "*_desc-bobs9_dseg.nii.gz"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Expected exactly one BOBs9 mask for {mri}; found {len(candidates)}. "
        "Check the BOBs naming convention and adjust matching_segmentation()."
    )


def find_pairs(bobs_root: Path, groups: dict):
    """Find MRI/mask pairs, retaining all sessions for each participant."""
    bobs_root = Path(bobs_root)
    if not bobs_root.exists():
        raise FileNotFoundError(bobs_root)
    split_by_subject = {
        subject: split for split, ids in groups.items() for subject in ids
    }
    found = {key: [] for key in groups}
    unassigned = []
    for mri in sorted(bobs_root.rglob("*_T2w.nii.gz")):
        # Take subject from a directory called sub-*, otherwise from filename.
        subject = next((part for part in mri.parts if part.startswith("sub-")), None)
        if subject is None:
            match = re.search(r"(?:^|_)sub-([A-Za-z0-9]+)", mri.name)
            subject = "sub-" + match.group(1) if match else None
        if subject not in split_by_subject:
            unassigned.append(str(mri))
            continue
        mask = matching_segmentation(mri)
        found[split_by_subject[subject]].append((subject, mri, mask))
    if unassigned:
        raise ValueError(
            f"{len(unassigned)} T2w files have no subject in splits. Examples: {unassigned[:3]}"
        )
    missing = {
        split: sorted(ids - {subject for subject, _, _ in found[split]})
        for split, ids in groups.items()
    }
    if any(missing.values()):
        raise ValueError(f"Some split participants have no MRI: {missing}")
    return found


def make_case_id(subject: str, mri_path: Path) -> str:
    """Unique nnU-Net case name, retaining any session/scan descriptors."""
    filename = mri_path.name[: -len("_T2w.nii.gz")]
    # Filename usually contains sub- and ses- entities. If session occurs only
    # in folders, include it to avoid collisions.
    session = next((x for x in mri_path.parts if x.startswith("ses-")), None)
    if session and session not in filename:
        filename = f"{subject}_{session}_{filename}"
    # nnU-Net case names should be filename-safe and stable.
    return re.sub(r"[^A-Za-z0-9_-]", "_", filename)


def validated_arrays(mri_path: Path, seg_path: Path):
    """Validate a paired BOBs MRI and segmentation.

    Checks matching 3D shapes, spatial alignment through affine
    matrices, finite integer labels, and allowed BOBs9 class IDs.

    Args:
        mri_path: Path to the T2-weighted NIfTI image.
        seg_path: Path to its BOBs9 segmentation.

    Returns:
        Segmentation NIfTI object and original integer label array.

    Raises:
        ValueError: If image/mask geometry or labels are invalid.
    """
    image = nib.load(str(mri_path))
    seg = nib.load(str(seg_path))
    if len(image.shape) != 3 or image.shape != seg.shape:
        raise ValueError(f"MRI/mask shape mismatch: {mri_path} vs {seg_path}")
    if not np.allclose(image.affine, seg.affine, atol=1e-5):
        raise ValueError(f"MRI/mask affine mismatch: {mri_path} vs {seg_path}")
    raw = np.asarray(seg.dataobj)
    if not np.isfinite(raw).all() or not np.allclose(raw, np.rint(raw), atol=1e-5):
        raise ValueError(
            f"Segmentation has non-integer or non-finite values: {seg_path}"
        )
    labels = {int(v) for v in np.unique(raw)}
    if not labels.issubset(LABEL_MAP):
        raise ValueError(
            f"Unexpected labels in {seg_path}: {sorted(labels - LABEL_MAP.keys())}"
        )
    return seg, raw


def remap_mask(raw: np.ndarray) -> np.ndarray:
    """Convert BOBs9 labels to consecutive indices using LABEL_MAP.

    Args:
        raw: Segmentation containing original BOBs9 class IDs.

    Returns:
        uint8 segmentation with labels 0-7.

    Raises:
        ValueError: If unexpected or non-integer labels are present.
    """
    if not np.isfinite(raw).all():
        raise ValueError("Segmentation contains non-finite values.")

    if not np.all(raw == np.rint(raw)):
        raise ValueError("Segmentation contains non-integer labels.")

    unknown = set(np.unique(raw)) - set(LABEL_MAP)

    if unknown:
        raise ValueError(f"Unexpected BOBs9 labels: {sorted(unknown)}")

    output = np.zeros(raw.shape, dtype=np.uint8)

    for old, new in LABEL_MAP.items():
        output[raw == old] = new

    return output
