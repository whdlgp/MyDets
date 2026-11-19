from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn
import argparse
import copy
import sys

# the scripts of onnx_tools find the modules of the project root like the scripts at the root do
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor, nn

from datatools.transforms import Compose, create_transform
from models.builder import build_model
from predict import draw_detections, read_image
from report import format_tree
from shared.output_directory import OutputDirectory
from shared.settings import CommonSettings, PredictSettings, TrainSettings, read_config, read_settings


@dataclass
class OnnxModel:
    """Onnx file opened on onnxruntime with the cpu, with the weights name of its metadata and its input size."""
    session: ort.InferenceSession
    weights_name: str
    image_size: tuple[int, int] | None


def load_onnx_model(onnx_path: Path) -> OnnxModel:
    """Open an onnx file of export.py, whose input size is None when it is dynamic."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    weights_name = session.get_modelmeta().custom_metadata_map.get("weights", "")
    _, _, height, width = session.get_inputs()[0].shape
    image_size = None if isinstance(height, str) else (width, height)
    return OnnxModel(session, weights_name, image_size)


def prepare_input(image: np.ndarray, config: dict, image_size: tuple[int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    """Resize a BGR image like the training does and convert it to the onnx input.

    Args:
        image: BGR image.
        config: config of the training.
        image_size: (width, height) of a fixed size onnx, None for a dynamic onnx.

    Returns:
        Resized BGR image, and its RGB pixel values in [0, 255] as float32 of shape (1, 3, height, width).
    """
    common = read_settings(CommonSettings, config, "common")

    # the fixed size takes the letterbox of the validation, the dynamic size the resize of predict
    if image_size is None:
        blocks, transform_size = read_settings(PredictSettings, config, "predict").transforms, common.image_size
    else:
        blocks, transform_size = read_settings(TrainSettings, config, "train").val_transforms, image_size

    # the transforms of the config without the last Normalize, which is inside the onnx
    transform = create_transform(blocks, transform_size, common.mean, common.std)
    resize = Compose(transform.transforms[:-1])
    empty = np.zeros((0, 5), np.float32)
    resized = resize([(image, empty, empty)]).images[0]

    # bgr to rgb, (1, 3, height, width)
    rgb = resized[:, :, ::-1].transpose(2, 0, 1)
    images = np.ascontiguousarray(rgb[None], dtype=np.float32)
    return resized, images


def run_onnx(model: OnnxModel, images: np.ndarray) -> dict[str, Tensor]:
    """Run the onnx model on onnxruntime and return the raw head outputs by name."""
    names = [output.name for output in model.session.get_outputs()]
    outputs = model.session.run(names, {"images": images})
    return {name: torch.from_numpy(output) for name, output in zip(names, outputs)}


def build_head(config: dict, num_classes: int) -> nn.Module:
    """Build the head of a config for its decode, which takes no weights."""
    model_config = copy.deepcopy(config["model"])
    model_config["backbone"].setdefault("args", {})["pretrained"] = False
    return build_model(model_config, num_classes).head.eval()


def exit_with_usage(error: str | None = None) -> NoReturn:
    """Exit with the usage and the expected directory structure.

    Args:
        error: error message printed first.
    """
    lines = [error, ""] if error else []
    lines += [
        "usage: uv run onnx_tools\\inference.py <output_dir> <onnx file> <image>",
        "",
        "output_dir",
        "├─ export",
        "│  └─ <weights name>_<size>.onnx",
        "├─ config.yaml",
        "└─ names.txt"
    ]
    raise SystemExit("\n".join(lines))


def parse_arguments() -> argparse.Namespace:
    """Read the arguments and exit with the usage when they are wrong."""
    parser = argparse.ArgumentParser(description="Detect the objects of an image with an onnx export.")
    parser.add_argument("output_dir", type=Path, help="output directory of train.py, such as runs\\<config name>")
    parser.add_argument("onnx_path", type=Path, help="onnx file of onnx_tools\\export.py")
    parser.add_argument("image_path", type=Path, help="image to detect the objects of")
    args = parser.parse_args()

    required = [args.output_dir / "config.yaml", args.output_dir / "names.txt", args.onnx_path, args.image_path]
    missing = [path for path in required if not path.exists()]

    if missing:
        exit_with_usage(f"no such file: {', '.join(str(path) for path in missing)}")

    return args


def main():
    """Detect the objects of an image with an onnx export on onnxruntime, decoded by the head of the config.

    Usage: uv run onnx_tools\\inference.py <output_dir> <onnx file> <image>
    Saved in <output_dir>/export/ as <image name>_<onnx name>_inference.jpg.
    """
    args = parse_arguments()
    output = OutputDirectory(args.output_dir)

    # read the training
    config = read_config(output.config)
    names = output.names.read_text().splitlines()
    score_threshold = read_settings(PredictSettings, config, "predict").score_threshold

    # run the onnx and decode its raw outputs
    model = load_onnx_model(args.onnx_path)
    head = build_head(config, len(names))
    image = read_image(args.image_path)
    resized, images = prepare_input(image, config, model.image_size)
    outputs = run_onnx(model, images)
    detections = head.decode(outputs)[0]

    # draw and save
    drawn = draw_detections(resized, detections, names, score_threshold)
    out_path = output.export / f"{args.image_path.stem}_{args.onnx_path.stem}_inference.jpg"
    output.export.mkdir(exist_ok=True)
    cv2.imencode(".jpg", drawn)[1].tofile(out_path)

    # print the summary
    size_kind = "dynamic" if model.image_size is None else "fixed"
    boxes = int((detections.scores >= score_threshold).sum())
    rows = [
        ("input", f"{images.shape[3]}x{images.shape[2]}, {size_kind} onnx"),
        ("boxes", f"{boxes} above {score_threshold}")
    ]
    title = f"MyDets onnx inference: {args.onnx_path.name}"
    print(format_tree(title, [(args.image_path.name, rows)]))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
