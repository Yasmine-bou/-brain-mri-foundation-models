"""
Visual quality control of SSL3D-preprocessed BOBs MRI volumes and masks.

This script generates visualizations of preprocessed Baby Open Brains
(BOBs) MRI volumes and their corresponding segmentation masks.

For each selected MRI case, it produces a PNG figure containing:
    - Axial MRI slice.
    - Coronal MRI slice.
    - Sagittal MRI slice.
    - The same three slices with segmentation overlays.

Main processing steps:
    1. Locate preprocessed BOBs train/validation/test directories.
    2. Select a configurable number of MRI cases per split.
    3. Load MRI and segmentation volumes from Blosc2 files.
    4. Validate image/mask shapes and numerical values.
    5. Verify that segmentation labels belong to {-1, 0, ..., 7}.
    6. Select representative slices based on foreground locations.
    7. Apply robust MRI intensity display limits.
    8. Overlay segmentation labels using a discrete color map.
    9. Save the resulting figures as PNG images.

Default behavior:
    Visualize the first three MRI cases from each split.

Optional behavior:
    Use --all to visualize every available case.

Input format:
    MRI:  (1, D, H, W)
    Mask: (1, D, H, W)

Output:
    <output_dir>/<split>/<case_id>_overlay.png

Segmentation labels:
    -1: Ignored region, not displayed.
     0: Background, not displayed.
   1-7: Remapped BOBs9 foreground segmentation classes.

Requirements:
    - Blosc2
    - NumPy
    - Matplotlib
    - Existing SSL3D-preprocessed BOBs volumes

Usage:
    Run inside the nnSSL Docker environment.

    python scripts/visualization/visualize_bobs_preprocessing.py

    python scripts/visualization/visualize_bobs_preprocessing.py --all

Notes:
    - The script uses Matplotlib's non-interactive Agg backend.
    - Figures are saved directly to disk without opening a GUI.
    - Original MRI and segmentation files are never modified.
    - The output is intended for visual quality control.
    - Visual inspection does not replace quantitative validation
      or establish complete voxel-wise anatomical correctness.
"""

import argparse
from pathlib import Path

import blosc2
import matplotlib

# Use a non-interactive Matplotlib backend.
# This allows figures to be generated in Docker or remote
# computing environments without a graphical display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


# ---------------------------------------------------------------------
# Dataset paths and visualization configuration
# ---------------------------------------------------------------------

# Root directory containing preprocessed BOBs MRI volumes.
DEFAULT_INPUT = Path("/opt/nnunet_resources/nnUNet_preprocessed/Dataset102_BOBs")

# Default directory where visual quality-control figures are saved.
DEFAULT_OUTPUT = Path("/opt/qa_output")

# Map each dataset split to its SSL3D preprocessing directory.
#
# Training data uses the spacing and normalization identifier.
# Validation and test data use dedicated SSL3D directories.
SPLITS = {
    "train": "Spacing__1.00_1.00_1.00___Norm__Z",
    "val": "SSL3D_val",
    "test": "SSL3D_test",
}


# ---------------------------------------------------------------------
# Segmentation color configuration
# ---------------------------------------------------------------------

# 0 = background; -1 = ignored region (not displayed).
# Labels 1..7 are the remapped BOBs9 class IDs, NOT the original IDs.
#
# Assign one distinct color to each foreground segmentation class.
COLORS = ["#E74C3C", "#2E86C1", "#27AE60", "#F39C12", "#8E44AD", "#16A085", "#D35400"]

# Create a discrete colormap for the seven foreground classes.
MASK_CMAP = ListedColormap(COLORS)

# Define the boundaries used to associate integer labels 1..7
# with their corresponding colors.
MASK_NORM = BoundaryNorm(np.arange(0.5, 8.5, 1), MASK_CMAP.N)


def read_volume(path: Path) -> np.ndarray:
    """
    Load a preprocessed MRI or segmentation volume from a Blosc2 file.

    Parameters
    ----------
    path : Path
        Path to the preprocessed .b2nd volume.

    Returns
    -------
    np.ndarray
        Three-dimensional volume with shape (D, H, W).

        The singleton channel dimension is removed.

    Raises
    ------
    ValueError
        If the stored array does not have shape (1, D, H, W).

    Notes
    -----
    The file is opened in read-only mode.

    This function does not modify the source volume or perform
    any additional preprocessing.
    """

    # Load the complete Blosc2 array into memory.
    a = np.asarray(blosc2.open(str(path), mode="r")[:])

    # Verify that the volume contains exactly one channel
    # and three spatial dimensions.
    if a.ndim != 4 or a.shape[0] != 1:
        raise ValueError(f"Expected (1,D,H,W), got {a.shape}: {path}")

    # Remove the singleton channel dimension.
    # Output shape: (D, H, W).
    return a[0]


def make_figure(
    image: np.ndarray, seg: np.ndarray, case_id: str, split: str, outfile: Path
) -> None:
    """
    Generate and save MRI slices with segmentation overlays.

    Parameters
    ----------
    image : np.ndarray
        Three-dimensional MRI volume with shape (D, H, W).

    seg : np.ndarray
        Corresponding segmentation mask with shape (D, H, W).

    case_id : str
        Unique identifier of the BOBs MRI case or session.

    split : str
        Dataset split ("train", "val", or "test").

    outfile : Path
        Destination path of the generated PNG image.

    Returns
    -------
    None
        The figure is saved to disk.

    Visualization
    -------------
    The figure contains two rows and three columns.

    First row:
        MRI-only slices.

    Second row:
        The same MRI slices with segmentation overlays.

    Columns:
        1. Axial view (depth axis).
        2. Coronal view (height axis).
        3. Sagittal view (width axis).

    Slice selection:
        The median location of foreground segmentation voxels
        is used to select representative slices.

        If no foreground labels are present, the geometric
        center of the MRI volume is used instead.

    Intensity display:
        MRI contrast is determined using the 1st and 99th
        intensity percentiles to reduce sensitivity to outliers.

    Segmentation display:
        Only foreground labels 1..7 are visualized.

        Background and ignored regions are masked.

    Raises
    ------
    ValueError
        If image/mask shapes differ, non-finite values are
        detected, or unexpected segmentation labels are found.
    """

    # -------------------------------------------------------------
    # 1. Validate MRI and segmentation compatibility
    # -------------------------------------------------------------

    # MRI and segmentation must have identical spatial dimensions.
    if image.shape != seg.shape:
        raise ValueError(f"Shape mismatch for {case_id}: {image.shape} != {seg.shape}")

    # Reject arrays containing NaN or infinity values.
    if not np.isfinite(image).all() or not np.isfinite(seg).all():
        raise ValueError(f"Non-finite values for {case_id}")

    # Identify labels outside the expected segmentation convention.
    # Allowed labels: -1, 0, 1, 2, 3, 4, 5, 6, 7.
    bad_labels = set(np.unique(seg).tolist()) - set(range(-1, 8))

    if bad_labels:
        raise ValueError(f"Unexpected labels in {case_id}: {sorted(bad_labels)}")

    # -------------------------------------------------------------
    # 2. Select representative anatomical slice locations
    # -------------------------------------------------------------

    # Use the center of the labelled foreground when available, otherwise volume center.
    #
    # Foreground voxels have segmentation labels greater than zero.
    locations = np.argwhere(seg > 0)

    # Calculate the median foreground coordinate along each
    # spatial dimension.
    #
    # If foreground voxels are absent, use the geometric center
    # of the complete volume instead.
    center = (
        np.median(locations, axis=0).astype(int)
        if locations.size
        else np.array(image.shape) // 2
    )

    # Define the three orthogonal anatomical views.
    #
    # Each entry contains:
    #     - View name.
    #     - Spatial axis used for slicing.
    #     - Selected slice index.
    views = [
        ("Axial (D)", 0, int(center[0])),
        ("Coronal (H)", 1, int(center[1])),
        ("Sagittal (W)", 2, int(center[2])),
    ]

    # -------------------------------------------------------------
    # 3. Determine robust MRI display contrast
    # -------------------------------------------------------------

    # Use robust display limits so a few outliers do not flatten contrast.
    #
    # Consider only finite MRI intensity values.
    finite_pixels = image[np.isfinite(image)]

    # Use the 1st and 99th intensity percentiles as display limits.
    lo, hi = np.percentile(finite_pixels, [1, 99])

    # Handle cases where the two percentile values are identical.
    if hi <= lo:
        # Fall back to the complete finite intensity range.
        lo, hi = float(finite_pixels.min()), float(finite_pixels.max())

        # Ensure a nonzero display range, even for constant images.
        if hi <= lo:
            hi = lo + 1.0

    # -------------------------------------------------------------
    # 4. Create the visualization figure
    # -------------------------------------------------------------

    # Create a 2x3 layout:
    #     Row 0: MRI only.
    #     Row 1: MRI with segmentation overlay.
    #
    # constrained_layout automatically adjusts subplot spacing.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)

    # Process axial, coronal, and sagittal views.
    for col, (name, axis, index) in enumerate(views):
        # Extract the selected MRI slice and rotate it for display.
        mri_slice = np.rot90(np.take(image, index, axis=axis))

        # Extract the corresponding segmentation slice.
        # Identical slicing and rotation preserve visual alignment.
        seg_slice = np.rot90(np.take(seg, index, axis=axis))

        # Mask background and ignored regions.
        # Only segmentation labels strictly greater than zero remain visible.
        valid_seg = np.ma.masked_where(seg_slice <= 0, seg_slice)

        # ---------------------------------------------------------
        # 5. Draw MRI-only and MRI+segmentation views
        # ---------------------------------------------------------

        for row in range(2):
            ax = axes[row, col]

            # Display the grayscale MRI slice.
            ax.imshow(mri_slice, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")

            # Add the segmentation overlay only in the second row.
            if row == 1:
                ax.imshow(
                    valid_seg,
                    cmap=MASK_CMAP,
                    norm=MASK_NORM,
                    alpha=0.5,
                    interpolation="nearest",
                )

            # Display the anatomical view and selected slice index.
            # The second row is explicitly identified as an overlay.
            ax.set_title(f"{name} · slice {index}" + (" · overlay" if row else ""))

            # Hide coordinate axes for clearer anatomical inspection.
            ax.axis("off")

    # -------------------------------------------------------------
    # 6. Add figure metadata
    # -------------------------------------------------------------

    # Include dataset, split, case ID, spatial dimensions,
    # and the segmentation labels present in the volume.
    fig.suptitle(
        f"BOBs SSL3D | {split} | {case_id} | MRI shape {tuple(image.shape)} | labels {sorted(np.unique(seg).tolist())}",
        fontsize=11,
    )

    # -------------------------------------------------------------
    # 7. Save the generated figure
    # -------------------------------------------------------------

    # Create the destination directory if necessary.
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Save the visualization as a PNG image.
    # bbox_inches="tight" removes unnecessary surrounding whitespace.
    fig.savefig(outfile, dpi=135, bbox_inches="tight")

    # Close the figure to release Matplotlib resources.
    # This is important when visualizing many MRI volumes.
    plt.close(fig)


def main() -> None:
    """
    Generate visual quality-control figures for BOBs preprocessing.

    The function:
        1. Parses input and output directory arguments.
        2. Selects a configurable number of cases per split.
        3. Loads paired MRI and segmentation volumes.
        4. Generates a visualization for each selected case.
        5. Saves PNG figures organized by split.
        6. Reports the number of generated figures.

    Default behavior:
        Three cases are selected from each split.

    With --all:
        Every available MRI case is visualized.

    Notes
    -----
    Input MRI cases are selected in sorted filename order.

    Visualization is performed independently for the training,
    validation, and test splits.
    """

    # -------------------------------------------------------------
    # 1. Parse command-line arguments
    # -------------------------------------------------------------

    parser = argparse.ArgumentParser(description=__doc__)

    # Root directory containing preprocessed BOBs split folders.
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)

    # Directory where visualization figures will be saved.
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    # Number of cases to visualize per split by default.
    parser.add_argument("--num-per-split", type=int, default=3)

    # Optionally visualize all available cases.
    parser.add_argument(
        "--all", action="store_true", help="Visualize every case in all splits"
    )

    args = parser.parse_args()

    # The requested number of visualizations must be positive.
    if args.num_per_split < 1:
        parser.error("--num-per-split must be >= 1")

    # -------------------------------------------------------------
    # 2. Process training, validation, and test splits
    # -------------------------------------------------------------

    # Count the total number of generated PNG figures.
    total = 0

    for split, folder_name in SPLITS.items():
        # Locate the current preprocessed split directory.
        folder = args.input_dir / folder_name

        # Ensure that the split directory exists.
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing {split} directory: {folder}")

        # Find all MRI .b2nd files.
        #
        # Segmentation files are excluded using the "_seg.b2nd" suffix.
        # Sorting ensures reproducible case selection.
        images = sorted(
            p for p in folder.glob("*.b2nd") if not p.name.endswith("_seg.b2nd")
        )

        # Require at least one preprocessed MRI volume.
        if not images:
            raise FileNotFoundError(f"No preprocessed images in {folder}")

        # Select either every case or the first requested number
        # of MRI volumes in the current split.
        selected = images if args.all else images[: args.num_per_split]

        # ---------------------------------------------------------
        # 3. Generate visualizations for selected cases
        # ---------------------------------------------------------

        for img_path in selected:
            # Extract the case identifier from the MRI filename.
            case_id = img_path.name[: -len(".b2nd")]

            # Construct the corresponding segmentation file path.
            seg_path = folder / f"{case_id}_seg.b2nd"

            # Every selected MRI must have a segmentation mask.
            if not seg_path.is_file():
                raise FileNotFoundError(f"Missing mask for {case_id}: {seg_path}")

            # Load the MRI and segmentation as 3D NumPy arrays.
            image = read_volume(img_path)
            mask = read_volume(seg_path)

            # Define the output PNG path.
            # Figures are organized into train/val/test subdirectories.
            outfile = args.output_dir / split / f"{case_id}_overlay.png"

            # Generate and save the three-view visualization.
            make_figure(image, mask, case_id, split, outfile)

            # Report the generated output file.
            print(f"SAVED {outfile}", flush=True)

            # Update the total number of generated figures.
            total += 1

        # Display the number of visualizations generated for this split.
        print(f"{split.upper()}: {len(selected)} visualizations", flush=True)

    # -------------------------------------------------------------
    # 4. Report visualization summary
    # -------------------------------------------------------------

    print(f"DONE: {total} PNG files in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
