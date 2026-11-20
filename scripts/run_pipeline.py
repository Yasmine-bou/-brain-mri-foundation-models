"""Single entry point for FeTA/BOBs SSL3D data preparation and quality checks.

Run from the repository root, e.g.:
    python -m scripts.run_pipeline prepare --dataset feta
    python -m scripts.run_pipeline prepare --dataset bobs --bobs-dir data/bobs/resampled75
    python -m scripts.run_pipeline validate --dataset all
    python -m scripts.run_pipeline test

Requirements: Docker Desktop/Engine, the nnSSL Docker image, and the project's
Python requirements for raw-data conversion. No GPU is required by these tasks.
GPU-dependent encoder integration tests run separately on Izar.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
IMAGE = "petermcgor/nnunetv2:1.1.0-nnssl"
CHECKPOINT_IN_IMAGE = (
    "/opt/nnunet_resources/models/"
    "BaseMAETrainerExtendedHealth__nnsslPlans__onemmiso/"
    "fold_all/checkpoint_best.pth"
)
DATASET_CONFIG = {
    "feta": {"id": 101, "folder": "Dataset101_FeTA", "name": "SSL3D_FeTA", "train": 48},
    "bobs": {"id": 102, "folder": "Dataset102_BOBs", "name": "SSL3D_BOBs", "train": 42},
}


def run(command: list[str], dry_run: bool) -> None:
    """Print a command and execute it, failing immediately on nonzero exit."""
    import shlex

    print("\n$ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def mount(host: Path, target: str, read_only: bool = False) -> str:
    """Return a Docker bind-mount argument using an absolute host path."""
    specification = f"type=bind,source={host.resolve()},target={target}"
    return specification + (",readonly" if read_only else "")


def docker(
    command: list[str],
    *,
    dry_run: bool,
    raw: bool = False,
    preprocessed: bool = True,
    results: bool = False,
    splits: bool = False,
    scripts: list[tuple[Path, str]] | None = None,
    workdir: str | None = None,
) -> None:
    """Execute a command in the pinned nnSSL image with the required mounts."""
    cmd = ["docker", "run", "--rm"]
    volumes = []
    if raw:
        volumes.append(
            (ROOT / "data/nnUNet_raw", "/opt/nnunet_resources/nnUNet_raw", False)
        )
    if preprocessed:
        volumes.append(
            (
                ROOT / "data/nnUNet_preprocessed",
                "/opt/nnunet_resources/nnUNet_preprocessed",
                False,
            )
        )
    if results:
        volumes.append(
            (
                ROOT / "data/nnUNet_results",
                "/opt/nnunet_resources/nnUNet_results",
                False,
            )
        )
    if splits:
        volumes.append((ROOT / "data/splits", "/opt/project_splits", True))
    for host, destination in scripts or []:
        volumes.append((host, destination, True))
    for host, destination, readonly in volumes:
        if not dry_run and not host.exists():
            raise FileNotFoundError(f"Required mount source not found: {host}")
        cmd += ["--mount", mount(host, destination, readonly)]
    if workdir is not None:
        cmd += ["--workdir", workdir]
    run(cmd + [IMAGE, *command], dry_run)


def script_in_container(relative: str) -> tuple[Path, str]:
    """Map a repository script to the /tmp directory inside Docker."""
    source = SCRIPTS / relative
    return source, f"/tmp/{source.name}"


def convert(dataset: str, args: argparse.Namespace) -> None:
    """Create raw nnU-Net files unless a conversion already exists."""
    target = ROOT / "data/nnUNet_raw" / DATASET_CONFIG[dataset]["folder"]
    if target.exists() and any(target.iterdir()):
        print(f"SKIP conversion: {target} already contains files.", flush=True)
        print("Existing data will be checked by the validation stage.", flush=True)
        return
    source = SCRIPTS / "preprocessing" / f"convert_{dataset}_to_nnunet.py"
    command = [sys.executable, str(source)]
    if dataset == "bobs":
        command += [
            "--bobs-dir",
            str(args.bobs_dir),
            "--splits",
            str(ROOT / "data/splits/bobs_splits.json"),
        ]
    run(command, args.dry_run)


def plan_and_train_preprocess(dataset: str, args: argparse.Namespace) -> None:
    """Create standard plans, then adapt SSL3D plans and preprocess training cases."""
    config = DATASET_CONFIG[dataset]
    processed = ROOT / "data/nnUNet_preprocessed" / config["folder"]
    standard_plans = processed / "nnUNetPlans.json"
    common = dict(dry_run=args.dry_run, raw=True, results=True)

    if standard_plans.is_file():
        print(f"SKIP planning: {standard_plans} already exists.", flush=True)
    else:
        docker(
            [
                "nnUNetv2_plan_and_preprocess",
                "-d",
                str(config["id"]),
                "--no_pp",
                "-np",
                "2",
            ],
            **common,
        )

    plans = sorted(processed.glob(f"ptPlans__{config['name']}*.json"))
    train_dir = processed / "Spacing__1.00_1.00_1.00___Norm__Z"
    complete = (
        train_dir.is_dir()
        and len(list(train_dir.glob("*_seg.b2nd"))) == config["train"]
        and len(
            [p for p in train_dir.glob("*.b2nd") if not p.name.endswith("_seg.b2nd")]
        )
        == config["train"]
        and len(list(train_dir.glob("*.pkl"))) == config["train"]
    )
    if len(plans) == 1 and complete:
        print(
            f"SKIP training preprocessing: {config['train']} triplets and one adapted plans file found."
        )
        return
    if plans or (train_dir.exists() and any(train_dir.iterdir())):
        raise RuntimeError(
            f"Partial or ambiguous training preprocessing in {processed}. "
            "Inspect the folder before running again; nothing will be deleted."
        )
    fix = script_in_container("docker_fixes/like_nnssl.py")
    docker(
        [
            "nnUNetv2_preprocess_like_nnssl",
            "-d",
            str(config["id"]),
            "-n",
            config["name"],
            "-pc",
            args.checkpoint_in_image,
            "-am",
            "like_pretrained",
            "-np",
            "1",
        ],
        scripts=[
            (
                fix[0],
                "/opt/nnunet_adaptation/nnunetv2/experiment_planning/like_nnssl.py",
            )
        ],
        **common,
    )


def preprocess_val_test(dataset: str, args: argparse.Namespace) -> None:
    """Apply existing training-derived plans to validation/test data."""
    file = script_in_container(f"preprocessing/preprocess_{dataset}_val_test.py")
    docker(
        ["python", file[1]],
        dry_run=args.dry_run,
        raw=True,
        splits=(dataset == "feta"),
        scripts=[file],
    )


def validate(dataset: str, args: argparse.Namespace) -> None:
    """Run the dataset-specific structural and numerical QC script."""
    file = script_in_container(f"validation/check_{dataset}_preprocessing.py")
    docker(
        ["python", file[1]],
        dry_run=args.dry_run,
        raw=(dataset == "bobs"),
        splits=(dataset == "feta"),
        scripts=[file],
    )


def test_dataset(args: argparse.Namespace) -> None:
    """Run the CPU-only SSL3DDataset smoke test inside nnSSL Docker."""
    # Mount the project as /workspace so imports from src/ and scripts/ work.
    docker(
        [
            "python",
            "-m",
            "scripts.tests.test_ssl3d_dataset",
            "--preprocessed-root",
            "/workspace/data/nnUNet_preprocessed",
            "--patch-size",
            str(args.patch_size),
        ],
        dry_run=args.dry_run,
        preprocessed=False,
        scripts=[(ROOT, "/workspace")],
        workdir="/workspace",
    )


def main() -> None:
    """Parse CLI options and orchestrate an existing, non-destructive pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "test"))
    parser.add_argument("--dataset", choices=("feta", "bobs", "all"), default="all")
    parser.add_argument("--bobs-dir", type=Path, default=ROOT / "data/BOBs")
    parser.add_argument("--checkpoint-in-image", default=CHECKPOINT_IN_IMAGE)
    parser.add_argument("--patch-size", type=int, default=160)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running them"
    )
    args = parser.parse_args()

    if args.patch_size < 1:
        parser.error("--patch-size must be positive")
    selected = ("feta", "bobs") if args.dataset == "all" else (args.dataset,)
    if args.action == "test":
        if args.dataset != "all":
            parser.error(
                "The dataset smoke test always checks both FeTA and BOBs; use --dataset all"
            )
        test_dataset(args)
        return
    for dataset in selected:
        print(f"\n===== {args.action.upper()} {dataset.upper()} =====", flush=True)
        if args.action == "prepare":
            convert(dataset, args)
            plan_and_train_preprocess(dataset, args)
            preprocess_val_test(dataset, args)
        validate(dataset, args)
    print(f"\n{args.action.upper()} COMPLETED", flush=True)


if __name__ == "__main__":
    main()
