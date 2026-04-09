from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path | None = None,
        transform: Callable | None = None,
        num_classes: int = 1,
        image_suffixes: Sequence[str] | None = None,
        mask_suffix: str | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.transform = transform
        self.num_classes = num_classes
        self.image_suffixes = {suffix.lower() for suffix in (image_suffixes or IMAGE_SUFFIXES)}
        self.mask_suffix = mask_suffix

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")

        self.image_paths = sorted(
            path for path in self.image_dir.iterdir() if path.is_file() and path.suffix.lower() in self.image_suffixes
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in: {self.image_dir}")

        self.mask_paths: list[Path] | None = None
        if self.mask_dir is not None:
            if not self.mask_dir.exists():
                raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")
            self.mask_paths = [self._resolve_mask_path(image_path) for image_path in self.image_paths]

    def _resolve_mask_path(self, image_path: Path) -> Path:
        assert self.mask_dir is not None
        if self.mask_suffix is not None:
            candidate = self.mask_dir / f"{image_path.stem}{self.mask_suffix}"
            if candidate.exists():
                return candidate

        for suffix in [image_path.suffix, ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            candidate = self.mask_dir / f"{image_path.stem}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Mask for image not found: {image_path.name}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        sample = {
            "image": image_array,
            "image_path": str(image_path),
            "image_name": image_path.name,
        }

        if self.mask_paths is not None:
            mask = Image.open(self.mask_paths[index])
            mask_array = np.array(mask, dtype=np.int64)
            if mask_array.ndim == 3:
                mask_array = mask_array[..., 0]
            if self.num_classes == 1:
                mask_array = (mask_array > 0).astype(np.int64)
            sample["mask"] = mask_array

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
