from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn
import argparse
import copy
import math
import sys

# the scripts of onnx_tools find the modules of the project root like the scripts at the root do
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from models.base import Detections
from models.builder import build_model
from onnx_tools.export import ExportModel
from onnx_tools.inference import OnnxModel, load_onnx_model, prepare_input, run_onnx
from predict import draw_detections, read_image
from report import format_tree
from shared.output_directory import OutputDirectory
from shared.settings import CommonSettings, PredictSettings, read_config, read_settings

CONFIG = {
    "absolute_tolerance": 1e-4,
    "relative_tolerance": 1e-3
}


@dataclass
class OutputCheck:
    """Comparison of one raw output of the onnx model with the torch model, failures as (index, torch, onnx, diff, tolerance)."""
    name: str
    shape: tuple[int, ...]
    max_diff: float
    mean_diff: float
    total: int
    failures: list[tuple[tuple[int, ...], float, float, float, float]]
    allclose: bool


def load_torch_model(output: OutputDirectory, config: dict, num_classes: int, onnx_model: OnnxModel) -> ExportModel:
    """Build the model of a config with the weights that the onnx file was exported from, its outputs in the order of the onnx.

    Args:
        output: output directory of the training.
        config: config of the training.
        num_classes: number of object classes.
        onnx_model: onnx file to compare with.

    Returns:
        Model in eval mode.
    """
    model_config = copy.deepcopy(config["model"])
    model_config["backbone"].setdefault("args", {})["pretrained"] = False
    detector = build_model(model_config, num_classes)
    detector.load_state_dict(OutputDirectory.load(output.path / onnx_model.weights_name))
    common = read_settings(CommonSettings, config, "common")
    output_names = [o.name for o in onnx_model.session.get_outputs()]
    return ExportModel(detector.eval(), common.mean, common.std, output_names).eval()


def run_torch(model: ExportModel, images: np.ndarray) -> dict[str, Tensor]:
    """Run the torch model and return the raw head outputs by name."""
    with torch.no_grad():
        outputs = model(torch.from_numpy(images))

    return dict(zip(model.output_names, outputs))


def compare_outputs(torch_outputs: dict[str, Tensor], onnx_outputs: dict[str, Tensor]) -> list[OutputCheck]:
    """Compare each raw output, failing the values outside |diff| <= absolute + relative * |onnx|."""
    atol, rtol = CONFIG["absolute_tolerance"], CONFIG["relative_tolerance"]
    checks = []

    for name, expected in torch_outputs.items():
        actual = onnx_outputs[name]

        if expected.shape != actual.shape:
            checks.append(OutputCheck(name, tuple(actual.shape), math.inf, math.inf, expected.numel(), [], False))
            continue

        expected, actual = expected.double(), actual.double()
        diff = (expected - actual).abs()
        tolerance = atol + rtol * actual.abs()
        failures = []

        for index in (diff > tolerance).nonzero().tolist():
            index = tuple(index)
            failures.append((index, expected[index].item(), actual[index].item(), diff[index].item(), tolerance[index].item()))

        # an empty output has no max
        max_diff, mean_diff = 0.0, 0.0

        if diff.numel():
            max_diff, mean_diff = diff.max().item(), diff.mean().item()

        allclose = torch.allclose(expected, actual, rtol, atol)
        checks.append(OutputCheck(name, tuple(actual.shape), max_diff, mean_diff, diff.numel(), failures, allclose))

    return checks


def format_checks(checks: list[OutputCheck]) -> list[tuple[str, str]]:
    """Format the diff, the failed values and allclose of each output as rows of a tree."""
    rows = []

    for c in checks:
        if c.max_diff == math.inf:
            rows.append((c.name, f"FAIL  onnx shape {list(c.shape)} differs from torch"))
            continue

        status = "ok  " if not c.failures and c.allclose else "FAIL"
        ratio = len(c.failures) / max(c.total, 1)
        diffs = f"max diff {c.max_diff:.2e}  mean diff {c.mean_diff:.2e}"
        failed = f"failed {len(c.failures)}/{c.total} ({ratio:.2%})"
        rows.append((c.name, f"{status}  {diffs}  {failed}  allclose {c.allclose}  {list(c.shape)}"))

    return rows


def format_failures(checks: list[OutputCheck]) -> list[str]:
    """Format each value outside the tolerance with its index, both values, the diff and the tolerance."""
    return [f"{c.name}{list(index)}: torch={a:.8f}, onnx={b:.8f}, diff={diff:.8f}, tolerance={tolerance:.8f}"
            for c in checks for index, a, b, diff, tolerance in c.failures]


def draw_comparison(image: np.ndarray, torch_detections: Detections, onnx_detections: Detections, names: list[str], score_threshold: float) -> np.ndarray:
    """Draw the torch and the onnx detections side by side, each labeled."""
    panels = []

    for label, detections in [("PyTorch", torch_detections), ("ONNX", onnx_detections)]:
        panel = draw_detections(image, detections, names, score_threshold)
        cv2.putText(panel, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)

    return np.hstack(panels)


def exit_with_usage(error: str | None = None) -> NoReturn:
    """Exit with the usage and the expected directory structure.

    Args:
        error: error message printed first.
    """
    lines = [error, ""] if error else []
    lines += [
        "usage: uv run onnx_tools\\verify.py <output_dir> <onnx file> <image> [--visualize]",
        "",
        "output_dir",
        "├─ export",
        "│  └─ <weights name>_<size>.onnx",
        "├─ <weights name>.pt",
        "├─ config.yaml",
        "└─ names.txt"
    ]
    raise SystemExit("\n".join(lines))


def parse_arguments() -> argparse.Namespace:
    """Read the arguments and exit with the usage when they are wrong."""
    parser = argparse.ArgumentParser(description="Compare an onnx export with its torch model on an image.")
    parser.add_argument("output_dir", type=Path, help="output directory of train.py, such as runs\\<config name>")
    parser.add_argument("onnx_path", type=Path, help="onnx file of onnx_tools\\export.py")
    parser.add_argument("image_path", type=Path, help="image to compare on")
    parser.add_argument("--visualize", action="store_true", help="show the drawn comparison in a window")
    args = parser.parse_args()

    required = [args.output_dir / "config.yaml", args.output_dir / "names.txt", args.onnx_path, args.image_path]
    missing = [path for path in required if not path.exists()]

    if missing:
        exit_with_usage(f"no such file: {', '.join(str(path) for path in missing)}")

    return args


def main():
    """Compare an onnx export with its torch model on an image, the raw outputs and the decoded boxes.

    Usage: uv run onnx_tools\\verify.py <output_dir> <onnx file> <image> [--visualize]
    Saved in <output_dir>/export/ as <image name>_<onnx name>_verify.jpg.
    """
    args = parse_arguments()
    output = OutputDirectory(args.output_dir)

    # read the training
    config = read_config(output.config)
    names = output.names.read_text().splitlines()
    score_threshold = read_settings(PredictSettings, config, "predict").score_threshold

    # load both models
    onnx_model = load_onnx_model(args.onnx_path)

    if not (output.path / onnx_model.weights_name).is_file():
        exit_with_usage(f"the weights '{onnx_model.weights_name}' of {args.onnx_path.name} are not in {output.path}")

    torch_model = load_torch_model(output, config, len(names), onnx_model)

    # run both on the same input and compare the raw outputs
    image = read_image(args.image_path)
    resized, images = prepare_input(image, config, onnx_model.image_size)
    torch_outputs = run_torch(torch_model, images)
    onnx_outputs = run_onnx(onnx_model, images)
    checks = compare_outputs(torch_outputs, onnx_outputs)

    # decode both with the head and draw them side by side
    torch_detections = torch_model.model.head.decode(torch_outputs)[0]
    onnx_detections = torch_model.model.head.decode(onnx_outputs)[0]
    drawn = draw_comparison(resized, torch_detections, onnx_detections, names, score_threshold)
    out_path = output.export / f"{args.image_path.stem}_{args.onnx_path.stem}_verify.jpg"
    output.export.mkdir(exist_ok=True)
    cv2.imencode(".jpg", drawn)[1].tofile(out_path)

    # print the comparison
    size_kind = "dynamic" if onnx_model.image_size is None else "fixed"
    torch_boxes = int((torch_detections.scores >= score_threshold).sum())
    onnx_boxes = int((onnx_detections.scores >= score_threshold).sum())
    box_rows = [
        ("input", f"{images.shape[3]}x{images.shape[2]}, {size_kind} onnx"),
        ("PyTorch", f"{torch_boxes} above {score_threshold}"),
        ("ONNX", f"{onnx_boxes} above {score_threshold}")
    ]
    title = f"MyDets onnx verify: {args.onnx_path.name}, {onnx_model.weights_name}"
    print(format_tree(title, [("raw outputs", format_checks(checks)), ("boxes", box_rows)]))
    failures = format_failures(checks)

    if failures:
        print("\nvalues outside the tolerance")
        print("\n".join(failures))

    # opencv-python-headless has no window, so matplotlib shows it
    if args.visualize:
        plt.figure(args.onnx_path.name)
        plt.imshow(drawn[:, :, ::-1])
        plt.axis("off")
        plt.show()

    if any(c.failures or not c.allclose for c in checks):
        raise SystemExit(f"\nSaved, but onnx does not match torch: {out_path}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
