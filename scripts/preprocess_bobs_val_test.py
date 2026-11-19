"""Preprocess BOBs validation/test with the *existing* training-derived SSL3D plans.

Run inside the nnSSL Docker image, with nnUNet_raw and nnUNet_preprocessed mounted:
    python /tmp/preprocess_bobs_val_test.py

No fingerprint extraction, planning, or train-case preprocessing is performed.
"""

import argparse
import json
from pathlib import Path

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

DEFAULT_RAW_ROOT = Path('/opt/nnunet_resources/nnUNet_raw')
DEFAULT_PREPROCESSED_ROOT = Path('/opt/nnunet_resources/nnUNet_preprocessed')
EXPECTED_COUNTS = {'val': 15, 'test': 14}


def cases_from_raw(raw: Path, image_folder: str, label_folder: str):
    image_dir, label_dir = raw / image_folder, raw / label_folder
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f'Missing {image_dir} or {label_dir}')

    cases = []
    images = sorted(image_dir.glob('*_0000.nii.gz'))
    image_ids = {p.name[:-len('_0000.nii.gz')] for p in images}
    label_ids = {p.name[:-len('.nii.gz')] for p in label_dir.glob('*.nii.gz')}
    if image_ids != label_ids:
        raise ValueError(
            f'{image_folder}/{label_folder} mismatch: '
            f'images without labels={sorted(image_ids - label_ids)}, '
            f'labels without images={sorted(label_ids - image_ids)}'
        )
    for image in images:
        case_id = image.name[:-len('_0000.nii.gz')]
        cases.append((case_id, [str(image)], str(label_dir / (case_id + '.nii.gz'))))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument('--preprocessed-root', type=Path, default=DEFAULT_PREPROCESSED_ROOT)
    parser.add_argument('--dataset', default='Dataset102_BOBs')
    parser.add_argument('--configuration', default='3d_fullres')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing val/test cases')
    args = parser.parse_args()

    raw = args.raw_root / args.dataset
    preprocessed = args.preprocessed_root / args.dataset
    if not raw.is_dir() or not preprocessed.is_dir():
        raise FileNotFoundError(f'Raw or preprocessed dataset missing: {raw}, {preprocessed}')

    plan_files = sorted(preprocessed.glob('ptPlans__SSL3D_BOBs*.json'))
    if len(plan_files) != 1:
        raise RuntimeError(f'Expected exactly one SSL3D_BOBs plans file; found: {plan_files}')

    plans_manager = PlansManager(str(plan_files[0]))
    configuration_manager = plans_manager.get_configuration(args.configuration)
    dataset_json = json.loads((raw / 'dataset.json').read_text(encoding='utf-8'))
    preprocessor = configuration_manager.preprocessor_class(verbose=False)
    if not isinstance(preprocessor, DefaultPreprocessor):
        raise TypeError(f'Unexpected preprocessor: {type(preprocessor)}')

    splits = {
        'val': ('imagesVal', 'labelsVal', 'SSL3D_val'),
        'test': ('imagesTs', 'labelsTs', 'SSL3D_test'),
    }
    for split, (images_name, labels_name, destination) in splits.items():
        cases = cases_from_raw(raw, images_name, labels_name)
        if len(cases) != EXPECTED_COUNTS[split]:
            raise ValueError(f'{split}: expected {EXPECTED_COUNTS[split]} cases, got {len(cases)}')
        out_dir = preprocessed / destination
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'{split.upper()}: {len(cases)} cases -> {out_dir}', flush=True)
        for i, (case_id, images, seg_path) in enumerate(cases, start=1):
            target = out_dir / case_id
            outputs = [target.with_suffix('.b2nd'),
                       out_dir / f'{case_id}_seg.b2nd',
                       target.with_suffix('.pkl')]
            already = [p.exists() for p in outputs]
            if all(already) and not args.overwrite:
                print(f'  [{i}/{len(cases)}] {case_id}: already exists, skipping', flush=True)
                continue
            if any(already) and not args.overwrite:
                raise FileExistsError(f'Incomplete output for {case_id}: {outputs}. Use --overwrite to redo.')
            print(f'  [{i}/{len(cases)}] {case_id}', flush=True)
            preprocessor.run_case_save(
                str(target), images, seg_path,
                plans_manager, configuration_manager, dataset_json,
            )
        print(f'{split.upper()} COMPLETE: {len(cases)}/{len(cases)}', flush=True)

    print('BOBs VALIDATION + TEST PREPROCESSING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
