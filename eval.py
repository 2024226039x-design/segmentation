from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from data.dataset import SegmentationDataset
from data.transforms import build_transforms
from models.unet import UNet
from utils.metrics import compute_segmentation_metrics


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a semantic segmentation model.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
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


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_model(config: dict[str, Any], checkpoint_path: str, device: torch.device) -> nn.Module:
    model = UNet(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        base_channels=config["model"]["base_channels"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def create_loader(config: dict[str, Any], split: str) -> DataLoader:
    data_config = config["data"]
    dataset = SegmentationDataset(
        image_dir=data_config[f"{split}_image_dir"],
        mask_dir=data_config[f"{split}_mask_dir"],
        transform=build_transforms(data_config, train=False),
        num_classes=config["model"]["num_classes"],
    )
    return DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config.get("device", "auto"))
    checkpoint_path = args.checkpoint or str(Path(config["checkpoint_dir"]) / "best.pt")
    model = build_model(config, checkpoint_path, device=device)
    loader = create_loader(config, args.split)

    criterion = nn.BCEWithLogitsLoss() if config["model"]["num_classes"] == 1 else nn.CrossEntropyLoss()
    total_loss = 0.0
    total_metrics = {"pixel_accuracy": 0.0, "miou": 0.0, "dice": 0.0}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            targets = (masks > 0).float().unsqueeze(1) if config["model"]["num_classes"] == 1 else masks.long()
            loss = criterion(logits, targets)
            metrics = compute_segmentation_metrics(
                logits,
                masks,
                num_classes=config["model"]["num_classes"],
                threshold=config["predict"].get("threshold", 0.5),
            )
            total_loss += loss.item()
            for key, value in metrics.items():
                total_metrics[key] += value

    num_batches = max(len(loader), 1)
    summary = {
        "loss": total_loss / num_batches,
        **{key: value / num_batches for key, value in total_metrics.items()},
    }
    print(summary)


if __name__ == "__main__":
    main()
