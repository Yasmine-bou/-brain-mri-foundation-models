"""Visual QC of SSL3D-preprocessed BOBs MRI and segmentation masks.

Reads Blosc2 volumes and saves axial/coronal/sagittal MRI+mask overlays.
Default: first 3 cases per split; --all: every case.
Run in the nnSSL Docker image (blosc2, numpy, matplotlib required).
"""

import argparse
from pathlib import Path

import blosc2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

DEFAULT_INPUT = Path("/opt/nnunet_resources/nnUNet_preprocessed/Dataset102_BOBs")
DEFAULT_OUTPUT = Path("/opt/qa_output")
SPLITS = {
    "train": "Spacing__1.00_1.00_1.00___Norm__Z",
    "val": "SSL3D_val",
    "test": "SSL3D_test",
}

# 0 = background; -1 = ignored region (not displayed).
# Labels 1..7 are the remapped BOBs9 class IDs, NOT the original IDs.
COLORS = ["#E74C3C", "#2E86C1", "#27AE60", "#F39C12", "#8E44AD", "#16A085", "#D35400"]
MASK_CMAP = ListedColormap(COLORS)
MASK_NORM = BoundaryNorm(np.arange(0.5, 8.5, 1), MASK_CMAP.N)


def read_volume(path: Path) -> np.ndarray:
    a = np.asarray(blosc2.open(str(path), mode="r")[:])
    if a.ndim != 4 or a.shape[0] != 1:
        raise ValueError(f"Expected (1,D,H,W), got {a.shape}: {path}")
    return a[0]


def make_figure(image: np.ndarray, seg: np.ndarray, case_id: str, split: str, outfile: Path) -> None:
    if image.shape != seg.shape:
        raise ValueError(f"Shape mismatch for {case_id}: {image.shape} != {seg.shape}")
    if not np.isfinite(image).all() or not np.isfinite(seg).all():
        raise ValueError(f"Non-finite values for {case_id}")
    bad_labels = set(np.unique(seg).tolist()) - set(range(-1, 8))
    if bad_labels:
        raise ValueError(f"Unexpected labels in {case_id}: {sorted(bad_labels)}")

    # Use the center of the labelled foreground when available, otherwise volume center.
    locations = np.argwhere(seg > 0)
    center = np.median(locations, axis=0).astype(int) if locations.size else np.array(image.shape) // 2

    views = [
        ("Axial (D)", 0, int(center[0])),
        ("Coronal (H)", 1, int(center[1])),
        ("Sagittal (W)", 2, int(center[2])),
    ]

    # Use robust display limits so a few outliers do not flatten contrast.
    finite_pixels = image[np.isfinite(image)]
    lo, hi = np.percentile(finite_pixels, [1, 99])
    if hi <= lo:
        lo, hi = float(finite_pixels.min()), float(finite_pixels.max())
        if hi <= lo:
            hi = lo + 1.0

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for col, (name, axis, index) in enumerate(views):
        mri_slice = np.rot90(np.take(image, index, axis=axis))
        seg_slice = np.rot90(np.take(seg, index, axis=axis))
        valid_seg = np.ma.masked_where(seg_slice <= 0, seg_slice)

        for row in range(2):
            ax = axes[row, col]
            ax.imshow(mri_slice, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            if row == 1:
                ax.imshow(valid_seg, cmap=MASK_CMAP, norm=MASK_NORM, alpha=0.5, interpolation="nearest")
            ax.set_title(f"{name} · slice {index}" + (" · overlay" if row else ""))
            ax.axis("off")

    fig.suptitle(f"BOBs SSL3D | {split} | {case_id} | MRI shape {tuple(image.shape)} | labels {sorted(np.unique(seg).tolist())}", fontsize=11)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=135, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-per-split", type=int, default=3)
    parser.add_argument("--all", action="store_true", help="Visualize every case in all splits")
    args = parser.parse_args()
    if args.num_per_split < 1:
        parser.error("--num-per-split must be >= 1")

    total = 0
    for split, folder_name in SPLITS.items():
        folder = args.input_dir / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing {split} directory: {folder}")
        images = sorted(p for p in folder.glob("*.b2nd") if not p.name.endswith("_seg.b2nd"))
        if not images:
            raise FileNotFoundError(f"No preprocessed images in {folder}")
        selected = images if args.all else images[: args.num_per_split]
        for img_path in selected:
            case_id = img_path.name[:-len(".b2nd")]
            seg_path = folder / f"{case_id}_seg.b2nd"
            if not seg_path.is_file():
                raise FileNotFoundError(f"Missing mask for {case_id}: {seg_path}")
            image = read_volume(img_path)
            mask = read_volume(seg_path)
            outfile = args.output_dir / split / f"{case_id}_overlay.png"
            make_figure(image, mask, case_id, split, outfile)
            print(f"SAVED {outfile}", flush=True)
            total += 1
        print(f"{split.upper()}: {len(selected)} visualizations", flush=True)
    print(f"DONE: {total} PNG files in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
