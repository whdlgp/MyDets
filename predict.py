from pathlib import Path
from typing import NoReturn
import argparse
import random

import cv2
import numpy as np
import torch
from torch import nn

from datatools.loader import IMAGE_SUFFIXES
from datatools.transforms import Compose, create_transform
from models.base import Detections
from models.builder import build_model
from shared.device import Device
from shared.output_directory import OutputDirectory
from shared.settings import CommonSettings, PredictSettings, read_config, read_settings

CONFIG = {
    "weights": ["best_loss.pt", "best_map.pt"],
    "num_images": 16,
    "tile_size": 512,
    "gap": 2,
    "seed": 0
}


def exit_with_usage(error: str | None = None) -> NoReturn:
    """Exit with the usage and the expected directory structure.

    Args:
        error: error message printed first.
    """
    lines = [error, ""] if error else []
    lines += [
        "usage: uv run predict.py <output_dir> <image or image_dir> [--weights ...] [--num-images N] [--seed S] [--device D]",
        "",
        "output_dir",
        *[f"├─ {name}" for name in CONFIG["weights"]],
        "├─ config.yaml",
        "└─ names.txt"
    ]
    raise SystemExit("\n".join(lines))


def load_model(config: dict, num_classes: int, weights_path: Path, device: torch.device) -> nn.Module:
    """Build the model of a config and load its weights.

    Args:
        config: config of the training.
        num_classes: number of object classes.
        weights_path: weights saved by train.py.
        device: device to run the model on.

    Returns:
        Model in eval mode.
    """
    model = build_model(config["model"], num_classes)
    model.load_state_dict(OutputDirectory.load(weights_path))
    return model.to(device).eval()


@torch.no_grad()
def predict_image(model: nn.Module, image: np.ndarray, transform: Compose, device: torch.device) -> Detections:
    """Detect the objects of one image.

    Args:
        model: model in eval mode.
        image: BGR image.
        transform: predict transform from samples to the model input.
        device: device of the model.

    Returns:
        Detections with boxes normalized to the image.
    """
    empty = np.zeros((0, 5), np.float32)
    images, _ = transform([(image, empty, empty)])
    outputs = model(images.to(device))
    return model.head.decode(outputs)[0]


def draw_detections(image: np.ndarray, detections: Detections, names: list[str], score_threshold: float) -> np.ndarray:
    """Draw the boxes above the score threshold with their class and score.

    Args:
        image: BGR image.
        detections: detections with boxes normalized to the image.
        names: class names.
        score_threshold: minimum score of the drawn boxes.

    Returns:
        Copy of the image with the boxes drawn.
    """
    height, width = image.shape[:2]
    drawn = image.copy()
    keep = detections.scores >= score_threshold
    boxes = detections.convert("xyxy", (width, height))[keep].round().int().tolist()
    scores = detections.scores[keep].tolist()
    labels = detections.labels[keep].tolist()

    # line and text sizes follow the image size
    thickness = max(round(max(height, width) / 300), 1)
    font_scale = 0.4 * thickness

    for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
        hsv = np.uint8([[[label * 180 // len(names), 200, 255]]])
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        text = f"{names[label]} {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = max(y1 - text_height - baseline, 0)

        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, thickness)
        cv2.rectangle(drawn, (x1, top), (x1 + text_width, top + text_height + baseline), color, -1)
        cv2.putText(drawn, text, (x1, top + text_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return drawn


def make_grid(images: list[np.ndarray], num_images: int) -> np.ndarray:
    """Fit each image in a square tile keeping its aspect ratio and tile them in a square grid.

    Args:
        images: BGR images, at most num_images.
        num_images: number of tiles, which sets the grid size.

    Returns:
        Grid image, with empty tiles when there are fewer images.
    """
    tile, gap = CONFIG["tile_size"], CONFIG["gap"]
    columns = round(num_images ** 0.5)
    rows = (num_images + columns - 1) // columns
    grid = np.full((rows * tile + (rows - 1) * gap, columns * tile + (columns - 1) * gap, 3), 255, np.uint8)

    for i, image in enumerate(images):
        height, width = image.shape[:2]
        scale = tile / max(height, width)
        new_width, new_height = max(round(width * scale), 1), max(round(height * scale), 1)

        # pad with gray
        boxed = np.full((tile, tile, 3), 128, np.uint8)
        left, top = (tile - new_width) // 2, (tile - new_height) // 2
        boxed[top:top + new_height, left:left + new_width] = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        x, y = (i % columns) * (tile + gap), (i // columns) * (tile + gap)
        grid[y:y + tile, x:x + tile] = boxed

    return grid


def read_image(path: Path) -> np.ndarray:
    """Read an image as BGR, also from non-ascii paths on windows."""
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise SystemExit(f"cannot read image {path}")

    return image


def main():
    """Draw the detections of the best models of a training on one image, or on random images of a folder as a grid.

    Usage: uv run predict.py <output_dir> <image or image_dir> [--weights best_map.pt] [--num-images 16] [--seed 0] [--device cuda:1]
    Saved in <output_dir>/predict/ with the weights name at the end.
    """
    # read the arguments
    parser = argparse.ArgumentParser(description="Draw the detections of the best models of a training.")
    parser.add_argument("output_dir", type=Path, help="output directory of train.py, such as runs\\<config name>")
    parser.add_argument("input", type=Path, help="image, or folder to pick random images from")
    parser.add_argument("--weights", nargs="+", choices=CONFIG["weights"], default=CONFIG["weights"], help="weights to draw with")
    parser.add_argument("--num-images", type=int, default=CONFIG["num_images"], help="number of random images in the grid")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="seed of the random images, the same seed picks the same images")
    parser.add_argument("--device", help="torch device such as cuda:1 or cpu, the first gpu when available by default")
    args = parser.parse_args()

    output = OutputDirectory(args.output_dir)
    input_path = args.input
    missing = [name for name in args.weights if not (output.path / name).exists()]

    if missing:
        exit_with_usage(f"no {', '.join(missing)} in {output.path}")

    if not input_path.exists():
        exit_with_usage(f"no such image or directory: {input_path}")

    if args.num_images < 1:
        exit_with_usage(f"--num-images must be at least 1, got {args.num_images}")

    # read the training
    config = read_config(output.config)
    names = output.names.read_text().splitlines()
    common = read_settings(CommonSettings, config, "common")
    settings = read_settings(PredictSettings, config, "predict")
    transform = create_transform(settings.transforms, common.image_size, common.mean, common.std)
    device = Device(args.device).torch_device

    # read the images
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

        if not paths:
            exit_with_usage(f"no images in {input_path}")

        # the same seed picks the same images
        paths = random.Random(args.seed).sample(paths, min(args.num_images, len(paths)))

    images = [read_image(path) for path in paths]
    output.predict.mkdir(exist_ok=True)
    saved = []

    # draw with each best model
    for weights_name in args.weights:
        model = load_model(config, len(names), output.path / weights_name, device)
        drawn = [draw_detections(image, predict_image(model, image, transform, device), names, settings.score_threshold) for image in images]
        stem = input_path.stem if input_path.is_file() else "grid"
        out_path = output.predict / f"{stem}_{Path(weights_name).stem}.jpg"
        cv2.imencode(".jpg", drawn[0] if input_path.is_file() else make_grid(drawn, args.num_images))[1].tofile(out_path)
        saved.append(out_path.name)

    print(f"\nSaved: {output.predict} ({', '.join(saved)})")


if __name__ == "__main__":
    main()
