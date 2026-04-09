from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SegmentationDataset
from data.transforms import build_transforms
from models.unet import UNet
from utils.metrics import compute_segmentation_metrics


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    def resolve_relative(path_str: str) -> str:
        candidate = (config_path.parent / path_str).resolve()
        if candidate.exists():
            return str(candidate)
        fallback = (config_path.parent.parent / path_str).resolve()
        return str(fallback)

    if "checkpoint_dir" in config and isinstance(config["checkpoint_dir"], str):
        config["checkpoint_dir"] = resolve_relative(config["checkpoint_dir"])
    if "output_dir" in config and isinstance(config["output_dir"], str):
        config["output_dir"] = resolve_relative(config["output_dir"])

    for key, value in config.get("data", {}).items():
        if isinstance(value, str) and value.startswith("./"):
            config["data"][key] = resolve_relative(value)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_model(config: dict[str, Any]) -> UNet:
    return UNet(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        base_channels=config["model"]["base_channels"],
    )


def build_loss_function(num_classes: int) -> nn.Module:
    if num_classes == 1:
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def prepare_targets(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes == 1:
        return (targets > 0).float().unsqueeze(1)
    return targets.long()


def create_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_config = config["data"]
    train_dataset = SegmentationDataset(
        image_dir=data_config["train_image_dir"],
        mask_dir=data_config["train_mask_dir"],
        transform=build_transforms(data_config, train=True),
        num_classes=config["model"]["num_classes"],
    )
    val_dataset = SegmentationDataset(
        image_dir=data_config["val_image_dir"],
        mask_dir=data_config["val_mask_dir"],
        transform=build_transforms(data_config, train=False),
        num_classes=config["model"]["num_classes"],
    )

    common = {
        "batch_size": config["train"]["batch_size"],
        "num_workers": config["train"]["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: Adam | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_metrics = {"pixel_accuracy": 0.0, "miou": 0.0, "dice": 0.0}

    progress = tqdm(loader, leave=False, desc="train" if is_train else "eval")
    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            targets = prepare_targets(masks, num_classes=num_classes)
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        metrics = compute_segmentation_metrics(
            logits.detach(),
            masks.detach(),
            num_classes=num_classes,
            threshold=threshold,
        )
        total_loss += loss.item()
        for key, value in metrics.items():
            total_metrics[key] += value
        progress.set_postfix(loss=f"{loss.item():.4f}", miou=f"{metrics['miou']:.4f}")

    num_batches = max(len(loader), 1)
    return {
        "loss": total_loss / num_batches,
        **{key: value / num_batches for key, value in total_metrics.items()},
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: Adam,
    epoch: int,
    metrics: dict[str, float],
    checkpoint_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = resolve_device(config.get("device", "auto"))

    train_loader, val_loader = create_dataloaders(config)
    model = build_model(config).to(device)
    criterion = build_loss_function(config["model"]["num_classes"])
    optimizer = Adam(model.parameters(), lr=config["train"]["learning_rate"])

    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_miou = float("-inf")

    for epoch in range(1, config["train"]["epochs"] + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            num_classes=config["model"]["num_classes"],
            optimizer=optimizer,
            threshold=config["train"].get("threshold", 0.5),
        )
        print(f"Epoch {epoch} train: {train_metrics}")

        if epoch % config["train"].get("val_interval", 1) != 0:
            continue

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=config["model"]["num_classes"],
            threshold=config["train"].get("threshold", 0.5),
        )
        print(f"Epoch {epoch} val: {val_metrics}")

        latest_path = checkpoint_dir / "latest.pt"
        save_checkpoint(model, optimizer, epoch, val_metrics, latest_path)

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_dir / "best.pt")


if __name__ == "__main__":
    main()
