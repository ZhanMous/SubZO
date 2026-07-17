"""Evaluation metrics for FSCIL experiments."""

import torch
import numpy as np


def harmonic_accuracy(seen_acc: float, unseen_acc: float) -> float:
    """Compute harmonic mean of seen and unseen class accuracy.

    Args:
        seen_acc: Accuracy on all seen classes
        unseen_acc: Accuracy on new (incremental) classes

    Returns:
        Harmonic mean (HAcc)
    """
    if seen_acc + unseen_acc == 0:
        return 0.0
    return 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc)


def old_class_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    base_class_ids: list[int],
) -> float:
    """Compute accuracy on old (base) classes only.

    Args:
        predictions: Predicted class labels [N]
        labels: True class labels [N]
        base_class_ids: List of base class IDs

    Returns:
        Accuracy on base-class samples
    """
    base_mask = torch.tensor([l.item() in base_class_ids for l in labels])
    if not base_mask.any():
        return 1.0  # No base-class samples → vacuously perfect
    return (predictions[base_mask] == labels[base_mask]).float().mean().item()


def per_class_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[int, float]:
    """Compute per-class accuracy.

    Returns:
        Dict mapping class_id → accuracy
    """
    acc = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            acc[c] = (predictions[mask] == c).float().mean().item()
    return acc


def session_report(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    base_classes: list[int],
    incremental_classes: list[list[int]],
    current_session: int,
) -> dict:
    """Generate a full session report.

    Args:
        predictions: Predicted labels
        labels: True labels
        base_classes: Base class IDs
        incremental_classes: List of incremental class lists per session
        current_session: Current session number

    Returns:
        Dict with all metrics for this session
    """
    # All seen classes
    seen_classes = list(base_classes)
    for s in range(current_session):
        seen_classes.extend(incremental_classes[s])

    # New classes (this session only)
    new_classes = incremental_classes[current_session - 1] if current_session > 0 else []

    # Masks
    seen_mask = torch.tensor([l.item() in seen_classes for l in labels])
    base_mask = torch.tensor([l.item() in base_classes for l in labels])
    new_mask = torch.tensor([l.item() in new_classes for l in labels]) if new_classes else torch.zeros_like(seen_mask, dtype=torch.bool)

    # Overall seen accuracy
    seen_acc = (predictions[seen_mask] == labels[seen_mask]).float().mean().item() if seen_mask.any() else 0.0

    # Base class accuracy
    base_acc = (predictions[base_mask] == labels[base_mask]).float().mean().item() if base_mask.any() else 1.0

    # New class accuracy
    new_acc = (predictions[new_mask] == labels[new_mask]).float().mean().item() if new_mask.any() else 0.0

    # Harmonic accuracy
    h_acc = harmonic_accuracy(seen_acc, new_acc) if new_classes else seen_acc

    return {
        "session": current_session,
        "seen_accuracy": seen_acc,
        "base_accuracy": base_acc,
        "new_accuracy": new_acc,
        "harmonic_accuracy": h_acc,
        "old_class_drop": 0.0,  # Computed relative to base session
        "num_seen_classes": len(seen_classes),
    }
