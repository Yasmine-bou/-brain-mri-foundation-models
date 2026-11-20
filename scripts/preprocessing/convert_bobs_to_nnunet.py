"""
Convert BOBs T2-weighted MRI volumes and BOBs9 segmentation masks
to the nnU-Net raw dataset format (Dataset102_BOBs).

The conversion pipeline:
    1. Load the predefined participant-level train/val/test splits.
    2. Find corresponding MRI and segmentation files.
    3. Validate expected scan counts and MRI/mask pairs.
    4. Verify that case identifiers are unique.
    5. Create the standard and custom nnU-Net split directories.
    6. Copy original T2w MRI volumes without modifying their contents.
    7. Remap segmentation labels using the BOBs label mapping.
    8. Generate dataset metadata and a conversion manifest.

The script does not perform spatial resampling, intensity normalization,
or any other MRI preprocessing.

Output structure:
    Dataset102_BOBs/
        imagesTr/                Training MRI volumes
        labelsTr/                Training segmentation masks
        imagesVal/               Validation MRI volumes
        labelsVal/               Validation segmentation masks
        imagesTs/                Test MRI volumes
        labelsTs/                Test segmentation masks
        dataset.json             nnU-Net dataset description
        label_mapping.json       Original-to-target label mapping
        conversion_manifest.json Source file and split information

Run from the project root:
    python scripts/preprocessing/convert_bobs_to_nnunet.py --bobs-dir data/BOBs

Set --bobs-dir to the directory containing the actual sub-* folders.

Use --dry-run to validate the dataset without writing output files.

Important:
    - Existing non-empty output directories are not overwritten.
    - Input files are never modified.
    - Segmentation label mappings are defined in src.datasets.bobs_utils.
    - Anatomical class names are not inferred from numeric labels.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib

# Locate the project root and make project modules importable.
# Note: this path expression assumes the script's original location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import BOBs-specific validation, splitting, and label-mapping utilities.
from src.datasets.bobs_utils import (  # noqa: E402
    EXPECTED_COUNTS,
    LABEL_MAP,
    find_pairs,
    load_splits,
    make_case_id,
    remap_mask,
    validated_arrays,
)


def convert(bobs_dir: Path, splits_file: Path, output: Path, dry_run: bool = False):
    """
    Convert validated BOBs MRI/segmentation pairs to nnU-Net raw format.

    Parameters
    ----------
    bobs_dir : Path
        Directory containing the original BOBs participant folders.

    splits_file : Path
        JSON file defining the participant-level train/val/test splits.

    output : Path
        Destination directory for Dataset102_BOBs.

    dry_run : bool, default=False
        If True, validate all MRI/mask pairs without creating output files.

    Notes
    -----
    MRI volumes are copied without preprocessing.

    Segmentation masks are validated and remapped to the target
    label convention defined in src.datasets.bobs_utils.

    The conversion aborts if scan counts are unexpected, case
    identifiers are duplicated, or the destination is non-empty.
    """

    # -------------------------------------------------------------
    # 1. Load splits and locate MRI/mask pairs
    # -------------------------------------------------------------

    # Read the predefined participant groups.
    groups = load_splits(splits_file)

    print("Participants:", {k: len(v) for k, v in groups.items()}, flush=True)

    # Find MRI and segmentation pairs belonging to each split.
    pairs = find_pairs(bobs_dir, groups)

    # Count MRI sessions in each split.
    counts = {k: len(v) for k, v in pairs.items()}
    print("Scans:", counts, flush=True)

    # Ensure the discovered dataset matches the expected scan counts.
    # Stop before creating any output if the counts are inconsistent.
    if counts != EXPECTED_COUNTS:
        raise ValueError(
            f"Expected {EXPECTED_COUNTS} scans; found {counts}. No output written."
        )

    # Define output directories for each data split.
    folders = {
        "train": ("imagesTr", "labelsTr"),
        "val": ("imagesVal", "labelsVal"),
        "test": ("imagesTs", "labelsTs"),
    }

    cases = []
    seen = set()

    # -------------------------------------------------------------
    # 2. Validate all source cases before writing files
    # -------------------------------------------------------------

    # Validate the WHOLE input before copying any files.
    for split, items in pairs.items():
        for participant, mri, mask in items:
            # Generate a case identifier using participant and MRI metadata.
            case_id = make_case_id(participant, mri)

            # Prevent duplicate identifiers, including across sessions.
            if case_id in seen:
                raise ValueError(
                    f"Duplicate case ID: {case_id}; check sessions and filenames"
                )

            seen.add(case_id)

            # Validate MRI and segmentation compatibility.
            validated_arrays(mri, mask)

            # Store validated case information for conversion.
            cases.append((split, participant, case_id, mri, mask))

    # Allow dataset validation without writing files.
    if dry_run:
        print(f"DRY RUN OK: {len(cases)} valid scan/mask pairs; nothing written.")
        return

    # -------------------------------------------------------------
    # 3. Prepare the nnU-Net output directory
    # -------------------------------------------------------------

    # Protect existing converted data from accidental overwriting.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {output}. Refusing to overwrite."
        )

    # Create the image and label directories for all splits.
    for names in folders.values():
        for folder in names:
            (output / folder).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 4. Copy MRI volumes and convert segmentation masks
    # -------------------------------------------------------------

    # Track the origin, participant, and split of every converted case.
    manifest = []

    for split, participant, case_id, mri_path, seg_path in cases:
        # Select output folders associated with the current split.
        img_folder, seg_folder = folders[split]

        # nnU-Net uses the _0000 suffix for the first imaging channel.
        out_mri = output / img_folder / f"{case_id}_0000.nii.gz"
        out_seg = output / seg_folder / f"{case_id}.nii.gz"

        # Copy the MRI without modifying voxel intensities or geometry.
        shutil.copy2(mri_path, out_mri)  # unchanged MRI, no preprocessing here

        # Reload the validated segmentation data.
        seg, raw = validated_arrays(mri_path, seg_path)

        # Apply the predefined BOBs label mapping while preserving
        # the original segmentation affine and header information.
        new_seg = nib.Nifti1Image(remap_mask(raw), seg.affine, seg.header.copy())

        # Store segmentation labels using unsigned 8-bit integers.
        new_seg.set_data_dtype("uint8")

        # Save the converted segmentation mask.
        nib.save(new_seg, str(out_seg))

        # Record the original file locations for traceability.
        manifest.append(
            {
                "case_id": case_id,
                "participant": participant,
                "split": split,
                "original_mri": str(mri_path),
                "original_seg": str(seg_path),
            }
        )

    # -------------------------------------------------------------
    # 5. Generate nnU-Net dataset metadata
    # -------------------------------------------------------------

    # Numeric placeholders are intentional: anatomical label names were not provided.
    # The target segmentation labels are documented numerically.
    dataset = {
        "channel_names": {"0": "T2w"},
        "labels": {"background": 0, **{f"class_{i}": i for i in range(1, 8)}},
        "numTraining": counts["train"],
        "file_ending": ".nii.gz",
        "description": "BOBs9; numeric labels remapped to 0..7; see label_mapping.json",
    }

    # Save the dataset description required by nnU-Net.
    (output / "dataset.json").write_text(
        json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
    )

    # Save the original-to-target segmentation label mapping.
    (output / "label_mapping.json").write_text(
        json.dumps({str(k): v for k, v in LABEL_MAP.items()}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Save the conversion manifest for reproducibility and provenance.
    (output / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"CONVERSION COMPLETE: {counts} -> {output}")


def main():
    """
    Parse command-line arguments and run the BOBs conversion.

    The optional --dry-run flag performs the complete validation
    procedure without creating files.
    """

    parser = argparse.ArgumentParser(description=__doc__)

    # Source directory containing the original BOBs dataset.
    parser.add_argument("--bobs-dir", type=Path, default=PROJECT_ROOT / "data" / "BOBs")

    # JSON file containing the predefined participant-level splits.
    parser.add_argument(
        "--splits",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "bobs_splits.json",
    )

    # Destination nnU-Net raw dataset directory.
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "nnUNet_raw" / "Dataset102_BOBs",
    )

    # Validate inputs without creating output directories or files.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pairs, splits and labels without writing",
    )

    args = parser.parse_args()

    # Execute the conversion using the selected paths and options.
    convert(args.bobs_dir, args.splits, args.output, args.dry_run)


if __name__ == "__main__":
    main()
