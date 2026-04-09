from __future__ import annotations

from typing import Any

import torch


def logits_to_prediction(logits: torch.Tensor, num_classes: int, threshold: float = 0.5) -> torch.Tensor:
    if num_classes == 1:
        probs = torch.sigmoid(logits)
        return (probs > threshold).long().squeeze(1)
    return torch.argmax(logits, dim=1)


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / torch.clamp(denominator, min=1e-6)


def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    threshold: float = 0.5,
) -> dict[str, Any]:
    predictions = logits_to_prediction(logits, num_classes=num_classes, threshold=threshold)
    targets = targets.long()

    if num_classes == 1:
        targets = (targets > 0).long()
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = ((predictions == 1) & (targets == 1)).sum().float()
        pred_area = (predictions == 1).sum().float()
        target_area = (targets == 1).sum().float()
        union = pred_area + target_area - intersection
        iou = _safe_divide(intersection, union)
        dice = _safe_divide(2 * intersection, pred_area + target_area)
        accuracy = (predictions == targets).float().mean()
        return {
            "pixel_accuracy": accuracy.item(),
            "miou": iou.item(),
            "dice": dice.item(),
        }

    predictions = predictions.view(-1)
    targets = targets.view(-1)
    ious = []
    dices = []
    for class_index in range(num_classes):
        pred_mask = predictions == class_index
        target_mask = targets == class_index
        intersection = (pred_mask & target_mask).sum().float()
        pred_area = pred_mask.sum().float()
        target_area = target_mask.sum().float()
        union = pred_area + target_area - intersection
        ious.append(_safe_divide(intersection, union))
        dices.append(_safe_divide(2 * intersection, pred_area + target_area))

    accuracy = (predictions == targets).float().mean()
    return {
        "pixel_accuracy": accuracy.item(),
        "miou": torch.stack(ious).mean().item(),
        "dice": torch.stack(dices).mean().item(),
    }
