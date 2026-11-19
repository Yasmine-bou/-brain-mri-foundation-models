"""Check BOBs SSL3D train/val/test outputs against converted nnU-Net raw data.

Works inside the nnSSL Docker image (Blosc2 available).
The checks verify expected case IDs, file triplets, shapes, finite values,
segmentation labels, and metadata, but do not establish voxel-wise alignment.
"""

import argparse
import json
import pickle
from pathlib import Path

import blosc2
import numpy as np

DEFAULT_RAW_ROOT = Path('/opt/nnunet_resources/nnUNet_raw')
DEFAULT_PREPROCESSED_ROOT = Path('/opt/nnunet_resources/nnUNet_preprocessed')
EXPECTED = {'train': 42, 'val': 15, 'test': 14}
ALLOWED_LABELS = {-1, 0, 1, 2, 3, 4, 5, 6, 7}  # -1 may be nnU-Net ignore label
SPLITS = {
    'train': ('imagesTr', 'labelsTr', 'Spacing__1.00_1.00_1.00___Norm__Z'),
    'val': ('imagesVal', 'labelsVal', 'SSL3D_val'),
    'test': ('imagesTs', 'labelsTs', 'SSL3D_test'),
}


def case_ids(folder: Path, suffix: str):
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return {p.name[:-len(suffix)] for p in folder.glob('*' + suffix)}


def check_case(folder: Path, case_id: str):
    errors = []
    paths = (
        folder / f'{case_id}.b2nd',
        folder / f'{case_id}_seg.b2nd',
        folder / f'{case_id}.pkl',
    )
    if not all(p.is_file() for p in paths):
        return [f'Missing file(s): {[p.name for p in paths if not p.is_file()]}']
    try:
        image = np.asarray(blosc2.open(str(paths[0]), mode='r')[:])
        seg = np.asarray(blosc2.open(str(paths[1]), mode='r')[:])
        if image.ndim != 4 or image.shape[0] != 1:
            errors.append(f'Unexpected MRI shape {image.shape}')
        if seg.shape != image.shape:
            errors.append(f'Image/seg shape mismatch: {image.shape} vs {seg.shape}')
        if not np.isfinite(image).all():
            errors.append('MRI has NaN/Inf')
        if not np.isfinite(seg).all():
            errors.append('Mask has NaN/Inf')
        elif not np.all(np.isin(np.unique(seg), list(ALLOWED_LABELS))):
            errors.append(f'Unexpected labels: {np.unique(seg).tolist()}')
        if not np.any(seg > 0):
            errors.append('No foreground segmentation labels')
        if image.size and float(np.std(image)) == 0.0:
            errors.append('Constant MRI')
        # Only load metadata generated locally by our trusted preprocessing.
        with paths[2].open('rb') as f:
            props = pickle.load(f)
        if not isinstance(props, dict) or not props:
            errors.append('Missing/invalid preprocessing metadata')
    except Exception as exc:
        errors.append(f'Cannot inspect case: {type(exc).__name__}: {exc}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument('--preprocessed-root', type=Path, default=DEFAULT_PREPROCESSED_ROOT)
    parser.add_argument('--dataset', default='Dataset102_BOBs')
    args = parser.parse_args()

    raw = args.raw_root / args.dataset
    processed = args.preprocessed_root / args.dataset
    assert (raw / 'dataset.json').is_file(), f'Raw dataset missing: {raw}'
    assert (raw / 'label_mapping.json').is_file(), 'Missing BOBs label mapping'
    plans_files = list(processed.glob('ptPlans__SSL3D_BOBs*.json'))
    if len(plans_files) != 1:
        raise ValueError(f'Expected one SSL3D_BOBs plan, got {plans_files}')
    plans = json.loads(plans_files[0].read_text(encoding='utf-8'))
    if 'configurations' not in plans:
        raise ValueError('SSL3D plans are missing configurations')

    failures = 0
    all_cases = set()
    for split, (image_folder, label_folder, processed_folder) in SPLITS.items():
        expected = case_ids(raw / image_folder, '_0000.nii.gz')
        labels = case_ids(raw / label_folder, '.nii.gz')
        if expected != labels:
            raise ValueError(f'Raw images/masks mismatch for {split}')
        if len(expected) != EXPECTED[split]:
            raise ValueError(f'Expected {EXPECTED[split]} {split} cases, got {len(expected)}')
        if all_cases.intersection(expected):
            raise ValueError(f'Duplicate case IDs across splits: {all_cases.intersection(expected)}')
        all_cases.update(expected)

        folder = processed / processed_folder
        actual_images = {c for c in case_ids(folder, '.b2nd') if not c.endswith('_seg')}
        actual_labels = case_ids(folder, '_seg.b2nd')
        actual_properties = case_ids(folder, '.pkl')
        for label, actual in (('images', actual_images), ('masks', actual_labels), ('metadata', actual_properties)):
            if actual != expected:
                raise ValueError(
                    f'{split} {label} ID mismatch: missing={sorted(expected - actual)}, '
                    f'extra={sorted(actual - expected)}'
                )

        passed = 0
        for case_id in sorted(expected):
            errors = check_case(folder, case_id)
            if errors:
                failures += 1
                print(f'FAIL {split}/{case_id}: {"; ".join(errors)}', flush=True)
            else:
                passed += 1
        print(f'{split.upper()}: {passed}/{len(expected)}', flush=True)

    print(f'TOTAL: {sum(EXPECTED.values()) - failures}/{sum(EXPECTED.values())}', flush=True)
    if failures:
        raise SystemExit(f'{failures} cases failed')
    print('ALL BOBs CHECKS PASSED!', flush=True)


if __name__ == '__main__':
    main()
