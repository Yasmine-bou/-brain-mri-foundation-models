
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_score(pred: np.ndarray, target: np.ndarray, label: int) -> float:
    """
    Compute the Dice score for one segmentation label.

    pred: predicted segmentation
    target: ground-truth segmentation
    label: class to evaluate
    """
    pred_mask = pred == label
    target_mask = target == label

    intersection = np.logical_and(pred_mask, target_mask).sum()
    total = pred_mask.sum() + target_mask.sum()

    if total == 0:
        return 1.0

    return float(2 * intersection / total)




def hd95_score(
    pred: np.ndarray,
    target: np.ndarray,
    label: int,
    spacing: tuple[float, ...],
) -> float:
    """
    Compute the symmetric 95th percentile Hausdorff distance
    for one segmentation class, in millimeters.
    """
    pred_mask = pred == label
    target_mask = target == label

    if not pred_mask.any() and not target_mask.any():
        return float("nan")

    if not pred_mask.any() or not target_mask.any():
        return float("inf")

    # Extract the boundaries of both regions
    pred_boundary = pred_mask & ~binary_erosion(pred_mask)
    target_boundary = target_mask & ~binary_erosion(target_mask)

    # Distances to the nearest boundary voxel
    dist_to_target = distance_transform_edt(
        ~target_boundary, sampling=spacing
    )
    dist_to_pred = distance_transform_edt(
        ~pred_boundary, sampling=spacing
    )

    # Boundary distances in both directions
    distances = np.concatenate([
        dist_to_target[pred_boundary],
        dist_to_pred[target_boundary],
    ])

    return float(np.percentile(distances, 95))


def evaluate_segmentation(
    pred: np.ndarray,
    target: np.ndarray,
    labels: list[int],
    spacing: tuple[float, ...],
) -> dict:
    """
    Compute Dice and HD95 for each segmentation label.
    """
    results = {}

    for label in labels:
        results[label] = {
            "dice": dice_score(pred, target, label),
            "hd95": hd95_score(pred, target, label, spacing),
        }

    return results
