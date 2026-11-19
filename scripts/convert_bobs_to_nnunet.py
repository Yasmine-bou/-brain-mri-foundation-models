"""Convert BOBs T2w MRI + BOBs9 masks to nnU-Net raw Dataset102_BOBs.

Run from project root: python scripts/convert_bobs_to_nnunet.py --bobs-dir data/BOBs
Set --bobs-dir to the directory containing the actual sub-* folders.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.bobs_utils import (  # noqa: E402
    EXPECTED_COUNTS,
    LABEL_MAP,
    find_pairs,
    load_splits,
    make_case_id,
    remap_mask,
    validated_arrays,
)


def convert(bobs_dir: Path, splits_file: Path, output: Path, dry_run: bool = False):
    groups = load_splits(splits_file)
    print("Participants:", {k: len(v) for k, v in groups.items()}, flush=True)
    pairs = find_pairs(bobs_dir, groups)
    counts = {k: len(v) for k, v in pairs.items()}
    print("Scans:", counts, flush=True)
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Expected {EXPECTED_COUNTS} scans; found {counts}. No output written.")

    folders = {
        "train": ("imagesTr", "labelsTr"),
        "val": ("imagesVal", "labelsVal"),
        "test": ("imagesTs", "labelsTs"),
    }
    cases = []
    seen = set()
    # Validate the WHOLE input before copying any files.
    for split, items in pairs.items():
        for participant, mri, mask in items:
            case_id = make_case_id(participant, mri)
            if case_id in seen:
                raise ValueError(f"Duplicate case ID: {case_id}; check sessions and filenames")
            seen.add(case_id)
            validated_arrays(mri, mask)
            cases.append((split, participant, case_id, mri, mask))

    if dry_run:
        print(f"DRY RUN OK: {len(cases)} valid scan/mask pairs; nothing written.")
        return

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output}. Refusing to overwrite.")
    for names in folders.values():
        for folder in names:
            (output / folder).mkdir(parents=True, exist_ok=True)

    manifest = []
    for split, participant, case_id, mri_path, seg_path in cases:
        img_folder, seg_folder = folders[split]
        out_mri = output / img_folder / f"{case_id}_0000.nii.gz"
        out_seg = output / seg_folder / f"{case_id}.nii.gz"
        shutil.copy2(mri_path, out_mri)  # unchanged MRI, no preprocessing here
        seg, raw = validated_arrays(mri_path, seg_path)
        new_seg = nib.Nifti1Image(remap_mask(raw), seg.affine, seg.header.copy())
        new_seg.set_data_dtype("uint8")
        nib.save(new_seg, str(out_seg))
        manifest.append({"case_id": case_id, "participant": participant, "split": split,
                         "original_mri": str(mri_path), "original_seg": str(seg_path)})

    # Numeric placeholders are intentional: anatomical label names were not provided.
    dataset = {
        "channel_names": {"0": "T2w"},
        "labels": {"background": 0, **{f"class_{i}": i for i in range(1, 8)}},
        "numTraining": counts["train"],
        "file_ending": ".nii.gz",
        "description": "BOBs9; numeric labels remapped to 0..7; see label_mapping.json",
    }
    (output / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    (output / "label_mapping.json").write_text(
        json.dumps({str(k): v for k, v in LABEL_MAP.items()}, indent=2) + "\n", encoding="utf-8"
    )
    (output / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"CONVERSION COMPLETE: {counts} -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bobs-dir", type=Path, default=PROJECT_ROOT / "data" / "BOBs")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "data" / "splits" / "bobs_splits.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "nnUNet_raw" / "Dataset102_BOBs")
    parser.add_argument("--dry-run", action="store_true", help="Validate pairs, splits and labels without writing")
    args = parser.parse_args()
    convert(args.bobs_dir, args.splits, args.output, args.dry_run)


if __name__ == "__main__":
    main()
