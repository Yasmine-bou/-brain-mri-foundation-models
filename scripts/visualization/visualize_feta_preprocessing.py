"""
Visual quality control of SSL3D-preprocessed FeTA MRI volumes and masks.

This script generates visual quality-control figures for fetal brain
MRI volumes from the FeTA dataset after SSL3D/nnU-Net preprocessing.

Each figure contains three orthogonal MRI views and their
corresponding segmentation overlays.

Main processing steps:
    1. Load predefined FeTA train/validation/test subject splits.
    2. Select representative subjects from each split.
    3. Load preprocessed MRI and segmentation volumes.
    4. Verify image/mask spatial compatibility.
    5. Select representative slices inside the segmented brain.
    6. Determine robust MRI intensity display limits.
    7. Generate three anatomical views.
    8. Overlay segmentation labels using a discrete colormap.
    9. Save the resulting figures as PNG files.

Default subject selection:
    Three subjects per split:
        - First subject in sorted order.
        - Middle subject in sorted order.
        - Last subject in sorted order.

    Total: 9 visualizations.

Optional behavior:
    Use --all to generate figures for all 80 FeTA subjects.

Expected input:
    Preprocessed MRI and segmentation files stored as .b2nd arrays.

    Stored array shape: (1, D, H, W)

    After removing the channel dimension: (D, H, W)

Output:
    <output_dir>/<split>/<subject>_visual_check.png

Segmentation labels:
    -1: Ignored region, not displayed.
     0: Background, not displayed.
   1-7: FeTA anatomical foreground tissue classes.

Requirements:
    - Blosc2
    - NumPy
    - Matplotlib
    - Preprocessed FeTA SSL3D volumes
    - Predefined FeTA subject-level splits

Usage:
    Run inside the nnSSL Docker environment.

    python scripts/visualization/visualize_feta_preprocessing.py

    python scripts/visualization/visualize_feta_preprocessing.py --all

Notes:
    - Matplotlib uses the non-interactive Agg backend.
    - All figures are saved directly to disk.
    - Original MRI and segmentation data are never modified.
    - This script is intended for visual preprocessing inspection.
    - Visual inspection does not replace quantitative validation.
"""

import argparse
import json
from pathlib import Path

import blosc2
import matplotlib

# Use the non-interactive Agg backend.
# This allows figures to be generated in Docker or remote
# environments without requiring a graphical display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


# ---------------------------------------------------------------------
# Dataset paths and visualization configuration
# ---------------------------------------------------------------------

# Root directory containing preprocessed FeTA SSL3D volumes.
ROOT = Path("/opt/nnunet_resources/nnUNet_preprocessed/Dataset101_FeTA")

# Predefined FeTA subject-level train/validation/test splits.
SPLITS_FILE = Path("/opt/project_splits/feta_splits.json")

# Directory where visual quality-control figures will be saved.
OUTPUT_DIR = Path("/opt/qa_output")

# Map each split to its corresponding preprocessed directory.
#
# Training data uses the directory identified by its preprocessing
# configuration, while validation and test use dedicated folders.
SPLIT_DIRS = {
    "train": ROOT / "Spacing__1.00_1.00_1.00___Norm__Z",
    "val": ROOT / "SSL3D_val",
    "test": ROOT / "SSL3D_test",
}


# ---------------------------------------------------------------------
# Segmentation label visualization
# ---------------------------------------------------------------------

# Label 0 = background; -1 = ignored region.
#
# Assign one color to background and each of the seven foreground
# tissue classes defined by the FeTA segmentation convention.
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

# Create a discrete colormap for segmentation labels 0..7.
LABEL_CMAP = ListedColormap(COLORS)

# Define integer label boundaries for the discrete colormap.
# Background and ignored voxels are subsequently excluded
# from the displayed overlays.
LABEL_NORM = BoundaryNorm(np.arange(-0.5, 8.5, 1), LABEL_CMAP.N)


def read_array(path):
    """
    Load a complete preprocessed Blosc2 array.

    Parameters
    ----------
    path : Path
        Path to an MRI or segmentation .b2nd file.

    Returns
    -------
    np.ndarray
        Complete stored array, typically with shape (1, D, H, W).

    Notes
    -----
    The array is read without modifying the original file.

    Unlike the BOBs visualization loader, this function does
    not independently validate the number of dimensions.
    """

    # Open the Blosc2 file in read-only mode and load its contents.
    return np.asarray(blosc2.open(urlpath=str(path), mode="r")[:])


def choose_subjects(ids, show_all):
    """
    Select FeTA subjects for visual quality control.

    Parameters
    ----------
    ids : list
        Subject identifiers belonging to one dataset split.

    show_all : bool
        If True, select every subject in the split.

        If False, select three representative subjects based
        on their positions in sorted identifier order.

    Returns
    -------
    list
        Selected subject identifiers.

    Default selection
    -----------------
    - First subject.
    - Middle subject.
    - Last subject.

    Notes
    -----
    Selection is based on sorted identifiers rather than
    anatomical or demographic characteristics.

    The function assumes the supplied subject list is non-empty.
    """

    # Sort subject identifiers to obtain deterministic selection.
    ids = sorted(ids)

    # Return all subjects when complete visualization is requested.
    if show_all:
        return ids

    # Three subjects per split:
    # first, middle, and last.
    #
    # This provides a simple reproducible sample of each split.
    return [
        ids[0],
        ids[len(ids) // 2],
        ids[-1],
    ]


def get_plane(volume, axis, index):
    """
    Extract a two-dimensional slice from a 3D MRI or mask volume.

    Parameters
    ----------
    volume : np.ndarray
        Three-dimensional array with shape (D, H, W).

    axis : int
        Spatial axis along which the slice is extracted.

        - 0: Depth axis.
        - 1: Height axis.
        - 2: Width axis.

    index : int
        Position of the selected slice along the chosen axis.

    Returns
    -------
    np.ndarray
        Two-dimensional slice extracted from the input volume.

    Notes
    -----
    The returned slice is not rotated or reoriented.

    The axis interpretation follows the stored array dimensions
    and does not independently verify anatomical orientation.
    """

    # Extract a slice perpendicular to the depth axis.
    if axis == 0:
        return volume[index, :, :]

    # Extract a slice perpendicular to the height axis.
    if axis == 1:
        return volume[:, index, :]

    # Extract a slice perpendicular to the width axis.
    return volume[:, :, index]


def visualize_case(subject, split_name, input_dir):
    """
    Generate a visual quality-control figure for one FeTA subject.

    Parameters
    ----------
    subject : str
        FeTA subject identifier, such as "sub-006".

    split_name : str
        Dataset split ("train", "val", or "test").

    input_dir : Path
        Directory containing the preprocessed MRI and mask files.

    Returns
    -------
    None
        A PNG visualization is generated and saved.

    Visualization layout
    --------------------
    The figure contains two rows and three columns.

    Row 1:
        MRI slices without segmentation overlays.

    Row 2:
        The same MRI slices with segmentation overlays.

    Columns:
        Slices along spatial axes 0, 1, and 2.

    Slice selection:
        The median coordinate of foreground segmentation voxels
        is used to select representative slices.

    MRI contrast:
        Display limits are computed from the 1st and 99th
        percentiles of nonzero finite MRI intensities.

    Segmentation:
        Only foreground labels 1..7 are displayed.

        Background (0), ignored regions (-1), and values above
        the foreground label range are masked.

    Raises
    ------
    ValueError
        If MRI and segmentation shapes differ, if no foreground
        labels exist, or if the MRI contains no valid nonzero
        finite intensity values.
    """

    # -------------------------------------------------------------
    # 1. Locate and load the MRI and segmentation files
    # -------------------------------------------------------------

    # Construct the preprocessed MRI and segmentation file paths.
    image_path = input_dir / f"{subject}.b2nd"
    mask_path = input_dir / f"{subject}_seg.b2nd"

    # Load both arrays and remove the singleton channel dimension.
    #
    # Stored shape: (1, D, H, W)
    # Used shape:   (D, H, W)
    image = read_array(image_path)[0]
    mask = read_array(mask_path)[0]

    # -------------------------------------------------------------
    # 2. Validate spatial compatibility
    # -------------------------------------------------------------

    # MRI and segmentation must have identical spatial dimensions.
    if image.shape != mask.shape:
        raise ValueError(f"{subject}: MRI and mask shapes differ")

    # -------------------------------------------------------------
    # 3. Select representative slices inside the brain
    # -------------------------------------------------------------

    # Choose a location inside the segmented brain.
    #
    # Foreground voxels correspond to segmentation labels 1..7.
    foreground = np.argwhere((mask >= 1) & (mask <= 7))

    # Require at least one anatomical foreground voxel.
    if len(foreground) == 0:
        raise ValueError(f"{subject}: no foreground labels")

    # Use the median foreground coordinate along each spatial axis.
    # The resulting indices select the slices to display.
    center = np.median(foreground, axis=0).astype(int)

    # -------------------------------------------------------------
    # 4. Compute MRI display contrast
    # -------------------------------------------------------------

    # Display contrast based on nonzero finite voxels.
    #
    # Excluding zero intensities reduces the influence of empty
    # background regions on the displayed MRI contrast.
    valid = image[np.isfinite(image) & (image != 0)]

    # Reject volumes without usable nonzero finite intensities.
    if valid.size == 0:
        raise ValueError(f"{subject}: empty MRI")

    # Use robust intensity percentiles to reduce the influence
    # of extreme intensity values on image visualization.
    vmin, vmax = np.percentile(valid, [1, 99])

    # Ensure that the upper display limit is greater than the lower.
    if vmax <= vmin:
        vmax = vmin + 1

    # -------------------------------------------------------------
    # 5. Create the visualization figure
    # -------------------------------------------------------------

    # Create a 2x3 figure.
    #
    # Row 0: MRI only.
    # Row 1: MRI with segmentation overlay.
    #
    # Each column corresponds to one spatial axis.
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13, 9),
        constrained_layout=True,
    )

    # Names used to identify the selected spatial axes.
    view_names = [
        "Axis 0",
        "Axis 1",
        "Axis 2",
    ]

    # -------------------------------------------------------------
    # 6. Display MRI and segmentation slices
    # -------------------------------------------------------------

    for axis in range(3):
        # Select the slice corresponding to the median foreground
        # coordinate along the current spatial axis.
        index = int(center[axis])

        # Extract corresponding MRI and segmentation slices.
        #
        # Using the same axis and index preserves spatial alignment
        # between the displayed MRI and segmentation arrays.
        img_slice = get_plane(image, axis, index)
        seg_slice = get_plane(mask, axis, index)

        # ---------------------------------------------------------
        # Row 1: MRI only
        # ---------------------------------------------------------

        # Display the selected MRI slice using grayscale intensities.
        axes[0, axis].imshow(
            img_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        # Annotate the spatial axis and selected slice index.
        axes[0, axis].set_title(f"{view_names[axis]} | slice {index} | MRI")

        # ---------------------------------------------------------
        # Row 2: MRI and segmentation overlay
        # ---------------------------------------------------------

        # Draw the grayscale MRI as the background image.
        axes[1, axis].imshow(
            img_slice,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        # Mask background, ignored regions, and labels above 7.
        #
        # Only anatomical foreground labels 1..7 remain visible.
        overlay = np.ma.masked_where(
            (seg_slice <= 0) | (seg_slice > 7),
            seg_slice,
        )

        # Superimpose the segmentation mask using the predefined
        # label colors and partial transparency.
        axes[1, axis].imshow(
            overlay,
            cmap=LABEL_CMAP,
            norm=LABEL_NORM,
            alpha=0.45,
            interpolation="nearest",
        )

        # Identify the segmentation overlay.
        axes[1, axis].set_title(f"{view_names[axis]} | MRI + segmentation")

        # Hide coordinate axes for both rows.
        for row in range(2):
            axes[row, axis].axis("off")

    # -------------------------------------------------------------
    # 7. Add figure metadata
    # -------------------------------------------------------------

    # Collect all segmentation labels observed in the volume.
    labels = sorted(set(np.unique(mask).astype(int).tolist()))

    # Display split, subject identifier, MRI shape, and observed labels.
    fig.suptitle(
        f"{split_name.upper()} | {subject}\nShape: {image.shape} | Labels: {labels}",
        fontsize=13,
    )

    # -------------------------------------------------------------
    # 8. Save the visualization
    # -------------------------------------------------------------

    # Create a separate output directory for the current split.
    split_output = OUTPUT_DIR / split_name
    split_output.mkdir(parents=True, exist_ok=True)

    # Define the subject-specific output PNG filename.
    output_path = split_output / f"{subject}_visual_check.png"

    # Save the figure using the requested resolution.
    fig.savefig(
        output_path,
        dpi=130,
        bbox_inches="tight",
    )

    # Close the figure to release Matplotlib resources.
    plt.close(fig)

    # Report successful figure generation.
    print(f"[OK] {subject} -> {output_path}")


def main():
    """
    Generate visual quality-control figures for FeTA preprocessing.

    The function:
        1. Parses the --all command-line option.
        2. Loads predefined subject-level dataset splits.
        3. Selects representative subjects from each split.
        4. Generates MRI/segmentation visualizations.
        5. Saves figures into split-specific directories.
        6. Reports the total number of generated PNG images.

    Default behavior:
        Three subjects per split, producing nine figures.

    With --all:
        Visualize all subjects listed in the split JSON.

    Notes
    -----
    Subject selection is deterministic because identifiers
    are sorted before choosing representative cases.
    """

    # -------------------------------------------------------------
    # 1. Parse command-line arguments
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser()

    # Enable visualization of all 80 FeTA subjects instead
    # of the default three subjects per split.
    parser.add_argument(
        "--all",
        action="store_true",
        help="Visualize all 80 subjects instead of 9 samples",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # 2. Load predefined FeTA subject splits
    # -------------------------------------------------------------

    # Read the train/validation/test subject identifiers.
    with open(SPLITS_FILE, encoding="utf-8") as f:
        splits = json.load(f)

    # Track the number of generated figures.
    total = 0

    # -------------------------------------------------------------
    # 3. Generate visualizations for each split
    # -------------------------------------------------------------

    for split_name, input_dir in SPLIT_DIRS.items():
        # Select all subjects or three representative cases,
        # depending on the --all argument.
        selected = choose_subjects(
            splits[split_name],
            show_all=args.all,
        )

        print(f"\n===== {split_name.upper()} =====")

        # Generate one quality-control figure per selected subject.
        for subject in selected:
            visualize_case(
                subject,
                split_name,
                input_dir,
            )

            # Update the total number of generated PNG files.
            total += 1

    # -------------------------------------------------------------
    # 4. Report visualization summary
    # -------------------------------------------------------------

    print("\n===== VISUALIZATION COMPLETE =====")
    print(f"Generated {total} PNG images")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
