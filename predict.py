from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import SegmentationDataset
from data.transforms import build_transforms
from models.unet import UNet
from utils.metrics import logits_to_prediction
from utils.visualize import save_prediction_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for semantic segmentation.")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent / "configs" / "config.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
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


def build_model(config: dict[str, Any], checkpoint_path: str, device: torch.device) -> UNet:
    model = UNet(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        base_channels=config["model"]["base_channels"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def create_loader(config: dict[str, Any], input_dir: str | None) -> DataLoader:
    data_config = config["data"]
    image_dir = input_dir or data_config["test_image_dir"]
    dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=None,
        transform=build_transforms(data_config, train=False),
        num_classes=config["model"]["num_classes"],
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    if mask.max() <= 1:
        mask = mask * 255
    Image.fromarray(mask.astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config.get("device", "auto"))
    checkpoint_path = args.checkpoint or str(Path(config["checkpoint_dir"]) / "best.pt")
    model = build_model(config, checkpoint_path, device=device)
    loader = create_loader(config, args.input_dir)
    output_dir = Path(args.output_dir or Path(config["output_dir"]) / "predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = config["predict"].get("threshold", 0.5)
    mean = config["data"].get("mean")
    std = config["data"].get("std")

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images)
            predictions = logits_to_prediction(
                logits,
                num_classes=config["model"]["num_classes"],
                threshold=threshold,
            )
            prediction = predictions[0].cpu().numpy().astype(np.uint8)

            image_name = Path(batch["image_name"][0]).stem
            mask_output_path = output_dir / f"{image_name}_mask.png"
            save_mask(prediction, mask_output_path)

            if config["predict"].get("save_visualization", True):
                visualization_path = output_dir / f"{image_name}_vis.png"
                save_prediction_visualization(
                    image=batch["image"][0].cpu().numpy(),
                    prediction=prediction,
                    output_path=visualization_path,
                    mean=mean,
                    std=std,
                )


if __name__ == "__main__":
    main()
