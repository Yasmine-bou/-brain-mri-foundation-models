"""Evaluation metrics for 3D brain MRI segmentation.

Computes per-class Dice score and symmetric 95th percentile Hausdorff
distance (HD95). Intended for evaluating FeTA and BOBs segmentations.
"""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_score(
    pred: np.ndarray,
    target: np.ndarray,
    label: int,
    ignore_index: int = -1,
) -> float:
    """Compute the Dice coefficient for one segmentation class.

    Voxels with the ignore label in the ground truth are excluded.

    Args:
        pred: Predicted segmentation with integer class labels.
        target: Ground-truth segmentation with the same shape as pred.
        label: Class ID to evaluate.
        ignore_index: Ground-truth label to exclude (default: -1).

    Returns:
        Dice score between 0 and 1, or NaN if the class is absent
        from both valid segmentations.
    """
    if pred.shape != target.shape:
        raise ValueError("Prediction and target shapes must match.")

    valid = target != ignore_index

    pred_mask = (pred == label) & valid
    target_mask = (target == label) & valid

    intersection = np.logical_and(pred_mask, target_mask).sum()
    total = pred_mask.sum() + target_mask.sum()

    if total == 0:
        return float("nan")

    return float(2 * intersection / total)


def hd95_score(
    pred: np.ndarray,
    target: np.ndarray,
    label: int,
    spacing: tuple[float, ...],
    ignore_index: int = -1,
) -> float:
    """Compute symmetric HD95 for one segmentation class, in millimeters.

    Evaluates the 95th percentile of distances between the predicted
    and ground-truth boundaries. Requires full-volume masks without
    ignored voxels to avoid artificial boundaries from patch padding.

    Args:
        pred: Predicted segmentation with integer class labels.
        target: Ground-truth segmentation with the same shape as pred.
        label: Class ID to evaluate.
        spacing: Voxel spacing in mm, one value per spatial dimension.
        ignore_index: Ground-truth ignore label (default: -1).

    Returns:
        HD95 in mm, NaN if both masks are empty, or Inf if only one
        mask is empty.

    Raises:
        ValueError: If shapes, spacing, or ground-truth labels are invalid.
    """
    if pred.shape != target.shape:
        raise ValueError("Prediction and target shapes must match.")

    if len(spacing) != pred.ndim or any(s <= 0 or not np.isfinite(s) for s in spacing):
        raise ValueError("Invalid voxel spacing.")

    if np.any(target == ignore_index):
        raise ValueError(
            "HD95 requires a full-volume mask without ignored voxels. "
            "Do not evaluate directly on padded patches."
        )

    pred_mask = pred == label
    target_mask = target == label

    if not pred_mask.any() and not target_mask.any():
        return float("nan")

    if not pred_mask.any() or not target_mask.any():
        return float("inf")

    # Extract boundaries and measure distances in physical units (mm).
    pred_boundary = pred_mask & ~binary_erosion(pred_mask)
    target_boundary = target_mask & ~binary_erosion(target_mask)

    dist_to_target = distance_transform_edt(~target_boundary, sampling=spacing)
    dist_to_pred = distance_transform_edt(~pred_boundary, sampling=spacing)

    distances = np.concatenate(
        [
            dist_to_target[pred_boundary],
            dist_to_pred[target_boundary],
        ]
    )

    return float(np.percentile(distances, 95))


def evaluate_segmentation(
    pred: np.ndarray,
    target: np.ndarray,
    labels: list[int],
    spacing: tuple[float, ...],
) -> dict:
    """Evaluate Dice and HD95 for the specified segmentation classes.

    Args:
        pred: Predicted segmentation with integer class labels.
        target: Ground-truth segmentation with the same shape as pred.
        labels: Class IDs to evaluate (typically 1-7, excluding background).
        spacing: Voxel spacing in mm.

    Returns:
        Dictionary mapping each class ID to its Dice and HD95 scores.
    """
    results = {}

    for label in labels:
        results[label] = {
            "dice": dice_score(pred, target, label),
            "hd95": hd95_score(pred, target, label, spacing),
        }

    return results
