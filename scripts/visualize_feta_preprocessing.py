
import argparse
import json
from pathlib import Path

import blosc2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


ROOT = Path(
    "/opt/nnunet_resources/nnUNet_preprocessed/Dataset101_FeTA"
)

SPLITS_FILE = Path("/opt/project_splits/feta_splits.json")
OUTPUT_DIR = Path("/opt/qa_output")

SPLIT_DIRS = {
    "train": ROOT / "Spacing__1.00_1.00_1.00___Norm__Z",
    "val": ROOT / "SSL3D_val",
    "test": ROOT / "SSL3D_test",
}

# Label 0 = background; -1 = ignored region.
COLORS = [
    "#000000",  # 0 background
    "#36c8dc",  # 1
    "#ffa500",  # 2
    "#56c271",  # 3
    "#e95858",  # 4
    "#a16fe0",  # 5
    "#f0df55",  # 6
    "#f57cc0",  # 7
]

LABEL_CMAP = ListedColormap(COLORS)
LABEL_NORM = BoundaryNorm(
    np.arange(-0.5, 8.5, 1),
    LABEL_CMAP.N
)


def read_array(path):
    return np.asarray(
        blosc2.open(urlpath=str(path), mode="r")[:]
    )


def choose_subjects(ids, show_all):
    ids = sorted(ids)

    if show_all:
        return ids

    # Three subjects per split:
    # first, middle, and last.
    return [
        ids[0],
        ids[len(ids) // 2],
        ids[-1],
    ]


def get_plane(volume, axis, index):
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def visualize_case(subject, split_name, input_dir):
    image_path = input_dir / f"{subject}.b2nd"
    mask_path = input_dir / f"{subject}_seg.b2nd"

    image = read_array(image_path)[0]
    mask = read_array(mask_path)[0]

    if image.shape != mask.shape:
        raise ValueError(
            f"{subject}: MRI and mask shapes differ"
        )

    # Choose a location inside the segmented brain.
    foreground = np.argwhere((mask >= 1) & (mask <= 7))

    if len(foreground) == 0:
        raise ValueError(f"{subject}: no foreground labels")

    center = np.median(foreground, axis=0).astype(int)

    # Display contrast based on nonzero finite voxels.
    valid = image[np.isfinite(image) & (image != 0)]

    if valid.size == 0:
        raise ValueError(f"{subject}: empty MRI")

    vmin, vmax = np.percentile(valid, [1, 99])

    if vmax <= vmin:
        vmax = vmin + 1

    fig, axes = plt.subplots(
        2, 3,
        figsize=(13, 9),
        constrained_layout=True,
    )

    view_names = [
        "Axis 0",
        "Axis 1",
        "Axis 2",
    ]

    for axis in range(3):
        index = int(center[axis])

        img_slice = get_plane(image, axis, index)
        seg_slice = get_plane(mask, axis, index)

        # Row 1: MRI only.
        axes[0, axis].imshow(
            img_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        axes[0, axis].set_title(
            f"{view_names[axis]} | slice {index} | MRI"
        )

        # Row 2: MRI and segmentation overlay.
        axes[1, axis].imshow(
            img_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        overlay = np.ma.masked_where(
            (seg_slice <= 0) | (seg_slice > 7),
            seg_slice,
        )

        axes[1, axis].imshow(
            overlay,
            cmap=LABEL_CMAP,
            norm=LABEL_NORM,
            alpha=0.45,
            interpolation="nearest",
        )

        axes[1, axis].set_title(
            f"{view_names[axis]} | MRI + segmentation"
        )

        for row in range(2):
            axes[row, axis].axis("off")

    labels = sorted(
        set(np.unique(mask).astype(int).tolist())
    )

    fig.suptitle(
        f"{split_name.upper()} | {subject}\n"
        f"Shape: {image.shape} | Labels: {labels}",
        fontsize=13,
    )

    split_output = OUTPUT_DIR / split_name
    split_output.mkdir(parents=True, exist_ok=True)

    output_path = split_output / f"{subject}_visual_check.png"

    fig.savefig(
        output_path,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"[OK] {subject} -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="Visualize all 80 subjects instead of 9 samples",
    )
    args = parser.parse_args()

    with open(SPLITS_FILE, encoding="utf-8") as f:
        splits = json.load(f)

    total = 0

    for split_name, input_dir in SPLIT_DIRS.items():
        selected = choose_subjects(
            splits[split_name],
            show_all=args.all,
        )

        print(f"\n===== {split_name.upper()} =====")

        for subject in selected:
            visualize_case(
                subject,
                split_name,
                input_dir,
            )
            total += 1

    print("\n===== VISUALIZATION COMPLETE =====")
    print(f"Generated {total} PNG images")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
