from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class Compose:
    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = transforms

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class Resize:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image = torch.from_numpy(sample["image"]).permute(2, 0, 1).float().unsqueeze(0)
        image = F.interpolate(image, size=self.size, mode="bilinear", align_corners=False)
        sample["image"] = image.squeeze(0).permute(1, 2, 0).byte().numpy()

        mask = sample.get("mask")
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            mask_tensor = F.interpolate(mask_tensor, size=self.size, mode="nearest")
            sample["mask"] = mask_tensor.squeeze(0).squeeze(0).long().numpy()
        return sample


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if random.random() < self.p:
            sample["image"] = np.ascontiguousarray(np.flip(sample["image"], axis=1))
            if "mask" in sample:
                sample["mask"] = np.ascontiguousarray(np.flip(sample["mask"], axis=1))
        return sample


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if random.random() < self.p:
            sample["image"] = np.ascontiguousarray(np.flip(sample["image"], axis=0))
            if "mask" in sample:
                sample["mask"] = np.ascontiguousarray(np.flip(sample["mask"], axis=0))
        return sample


class RandomRotate90:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if random.random() < self.p:
            k = random.randint(1, 3)
            sample["image"] = np.ascontiguousarray(np.rot90(sample["image"], k=k))
            if "mask" in sample:
                sample["mask"] = np.ascontiguousarray(np.rot90(sample["mask"], k=k))
        return sample


class Normalize:
    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image = sample["image"].astype(np.float32) / 255.0
        sample["image"] = (image - self.mean) / self.std
        return sample


class ToTensor:
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image = torch.from_numpy(sample["image"]).permute(2, 0, 1).float()
        sample["image"] = image
        if "mask" in sample:
            sample["mask"] = torch.from_numpy(sample["mask"]).long()
        return sample


def build_transforms(config: dict[str, Any], train: bool) -> Compose:
    image_size = tuple(config.get("image_size", [256, 256]))
    mean = config.get("mean", [0.485, 0.456, 0.406])
    std = config.get("std", [0.229, 0.224, 0.225])
    augmentation = config.get("augmentation", {})

    transforms: list[Any] = [Resize(image_size)]
    if train:
        if augmentation.get("horizontal_flip", True):
            transforms.append(RandomHorizontalFlip(p=augmentation.get("horizontal_flip_p", 0.5)))
        if augmentation.get("vertical_flip", False):
            transforms.append(RandomVerticalFlip(p=augmentation.get("vertical_flip_p", 0.5)))
        if augmentation.get("rotate90", False):
            transforms.append(RandomRotate90(p=augmentation.get("rotate90_p", 0.5)))
    transforms.extend([Normalize(mean=mean, std=std), ToTensor()])
    return Compose(transforms)
