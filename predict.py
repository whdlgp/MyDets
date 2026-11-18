from pathlib import Path
from typing import NoReturn
import random
import sys

import cv2
import numpy as np
import torch
import yaml
from torch import nn

from datatools.loader import IMAGE_SUFFIXES
from datatools.transforms import Compose, create_transform
from models.base import Detections
from models.builder import build_model

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
        "usage: uv run predict.py <run_dir> <image or image_dir>",
        "",
        "run_dir",
        *[f"├─ {name}" for name in CONFIG["weights"]],
        "├─ config.yaml",
        "└─ names.txt"
    ]
    raise SystemExit("\n".join(lines))


def load_model(config: dict, num_classes: int, weights_path: Path, device: torch.device) -> nn.Module:
    """Build the model of a config and load its weights.

    Args:
        config: config of the run.
        num_classes: number of object classes.
        weights_path: weights saved by train.py.
        device: device to run the model on.

    Returns:
        Model in eval mode.
    """
    model = build_model(config["model"], num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
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


def make_grid(images: list[np.ndarray]) -> np.ndarray:
    """Fit each image in a square tile keeping its aspect ratio and tile them in a square grid.

    Args:
        images: BGR images, at most num_images.

    Returns:
        Grid image, with empty tiles when there are fewer images.
    """
    tile, gap = CONFIG["tile_size"], CONFIG["gap"]
    columns = round(CONFIG["num_images"] ** 0.5)
    rows = (CONFIG["num_images"] + columns - 1) // columns
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
    """Draw the detections of the best models of a run on one image, or on random images of a folder as a grid.

    Usage: uv run predict.py <run_dir> <image or image_dir>
    Saved in <run_dir>/predict/ with the weights name at the end.
    """
    # read the arguments
    if len(sys.argv) != 3:
        exit_with_usage()

    run_dir, input_path = Path(sys.argv[1]), Path(sys.argv[2])
    missing = [name for name in CONFIG["weights"] if not (run_dir / name).exists()]

    if missing:
        exit_with_usage(f"no {', '.join(missing)} in {run_dir}")

    if not input_path.exists():
        exit_with_usage(f"no such image or directory: {input_path}")

    # read the run
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    names = (run_dir / "names.txt").read_text().splitlines()
    common, predict_config = config["common"], config["predict"]
    transform = create_transform(predict_config["transforms"], common["image_size"], common["mean"], common["std"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read the images
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

        if not paths:
            exit_with_usage(f"no images in {input_path}")

        # the same random images every run
        paths = random.Random(CONFIG["seed"]).sample(paths, min(CONFIG["num_images"], len(paths)))

    images = [read_image(path) for path in paths]
    out_dir = run_dir / "predict"
    out_dir.mkdir(exist_ok=True)
    saved = []

    # draw with each best model
    for weights_name in CONFIG["weights"]:
        model = load_model(config, len(names), run_dir / weights_name, device)
        drawn = [draw_detections(image, predict_image(model, image, transform, device), names, predict_config["score_threshold"]) for image in images]
        stem = input_path.stem if input_path.is_file() else "grid"
        out_path = out_dir / f"{stem}_{Path(weights_name).stem}.jpg"
        cv2.imencode(".jpg", drawn[0] if input_path.is_file() else make_grid(drawn))[1].tofile(out_path)
        saved.append(out_path.name)

    print(f"\nSaved: {out_dir} ({', '.join(saved)})")


if __name__ == "__main__":
    main()
