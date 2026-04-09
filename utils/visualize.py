from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def denormalize_image(image: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    mean_array = np.array(mean).reshape(1, 1, 3)
    std_array = np.array(std).reshape(1, 1, 3)
    restored = (image * std_array + mean_array) * 255.0
    return np.clip(restored, 0, 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],
            [220, 20, 60],
            [0, 128, 0],
            [30, 144, 255],
            [255, 165, 0],
            [148, 0, 211],
        ],
        dtype=np.uint8,
    )
    return palette[mask % len(palette)]


def save_prediction_visualization(
    image: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
    target: np.ndarray | None = None,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = np.transpose(image, (1, 2, 0))

    if mean is not None and std is not None:
        image = denormalize_image(image, mean=mean, std=std)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)

    columns = 3 if target is not None else 2
    fig, axes = plt.subplots(1, columns, figsize=(5 * columns, 5))
    axes = np.atleast_1d(axes)
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(colorize_mask(prediction))
    axes[1].set_title("Prediction")
    if target is not None:
        axes[2].imshow(colorize_mask(target))
        axes[2].set_title("Target")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
