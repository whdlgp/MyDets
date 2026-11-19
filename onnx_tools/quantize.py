from pathlib import Path
from typing import NoReturn
import argparse
import random
import sys
import time

# the scripts of onnx_tools find the modules of the project root like the scripts at the root do
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import onnx
import onnxruntime.quantization as ortq
from onnxruntime.transformers.float16 import convert_float_to_float16

from onnx_tools.inference import load_onnx_model, prepare_input
from predict import read_image
from report import format_tree
from shared.output_directory import OutputDirectory
from shared.settings import TrainSettings, read_config, read_settings

CONFIG = {
    "mode": "int8_static",
    "per_channel": False,
    "reduce_range": False,
    "calibration_method": "minmax",
    "calibration_samples": 500,
    "calibration_seed": 0
}

MODES = {
    "fp16": ("FP16", "Float16 conversion"),
    "int8_static": ("INT8", "Static quantization (QDQ)"),
    "int8_dynamic": ("INT8", "Dynamic quantization (QOperator)")
}

CALIBRATION_METHODS = {
    "minmax": ortq.CalibrationMethod.MinMax,
    "entropy": ortq.CalibrationMethod.Entropy
}

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class CalibrationImages(ortq.CalibrationDataReader):
    """Read validation images as the input of an onnx export for the calibration of a static quantization."""

    def __init__(self, paths: list[Path], config: dict, image_size: tuple[int, int] | None) -> None:
        """Initialize the reader.

        Args:
            paths: validation images.
            config: config of the training.
            image_size: (width, height) of the onnx input, None when it is dynamic.
        """
        self.paths = paths
        self.config = config
        self.image_size = image_size
        self.index = 0

    def get_next(self) -> dict | None:
        """Prepare the next image, or return None when all images were read."""
        if self.index == len(self.paths):
            return None

        _, images = prepare_input(read_image(self.paths[self.index]), self.config, self.image_size)
        self.index += 1
        print(f"\rCalibration: {self.index}/{len(self.paths)} ({self.index / len(self.paths):.0%})", end="\n" if self.index == len(self.paths) else "", flush=True)

        return {"images": images}

    def rewind(self) -> None:
        """Start reading the validation images again."""
        self.index = 0


def exit_with_usage(error: str | None = None) -> NoReturn:
    """Exit with the usage and the expected directory structure.

    Args:
        error: error message printed first.
    """
    lines = [error, ""] if error else []
    lines += [
        "usage: uv run onnx_tools\\quantize.py <output_dir> <onnx_file> [--mode fp16 | int8_static | int8_dynamic]",
        "",
        "output_dir",
        "├─ export",
        "│  └─ <onnx_file>",
        "├─ config.yaml",
        "└─ names.txt",
        "",
        "dataset_dir (from config.yaml: train.dataset_dir), required for int8_static",
        "└─ val",
        "   └─ images"
    ]
    raise SystemExit("\n".join(lines))


def parse_arguments() -> argparse.Namespace:
    """Read the arguments and exit with the usage when the files are missing."""
    parser = argparse.ArgumentParser(description="Convert an ONNX export to FP16 or INT8 for inference.")
    parser.add_argument("output_dir", type=Path, help="output directory of train.py, such as runs\\<config name>")
    parser.add_argument("onnx_path", type=Path, help="ONNX file of onnx_tools\\export.py")
    parser.add_argument("--mode", choices=MODES, default=CONFIG["mode"], help="fp16, int8_static or int8_dynamic")
    parser.add_argument("--per-channel", action="store_true", default=CONFIG["per_channel"], help="quantize weights per channel")
    parser.add_argument("--reduce-range", action="store_true", default=CONFIG["reduce_range"], help="use the reduced quantization range")
    parser.add_argument("--calibration-method", choices=CALIBRATION_METHODS, default=CONFIG["calibration_method"], help="calibration of int8_static")
    args = parser.parse_args()

    missing = [path for path in [args.output_dir / "config.yaml", args.output_dir / "names.txt", args.onnx_path] if not path.exists()]

    if missing:
        exit_with_usage(f"no such file: {', '.join(str(path) for path in missing)}")

    return args


def find_calibration_images(config: dict) -> list[Path]:
    """Pick the random validation images of the dataset in the config.

    Args:
        config: config of the training.

    Returns:
        Sorted paths of at most CONFIG["calibration_samples"] images.
    """
    dataset_dir = Path(read_settings(TrainSettings, config, "train").dataset_dir)
    image_dir = (dataset_dir if dataset_dir.is_absolute() else ROOT / dataset_dir) / "val" / "images"

    if not image_dir.is_dir():
        exit_with_usage(f"no such directory: {image_dir}")

    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

    if not paths:
        exit_with_usage(f"no supported images in {image_dir}")

    return sorted(random.Random(CONFIG["calibration_seed"]).sample(paths, min(CONFIG["calibration_samples"], len(paths))))


def main():
    """Convert an onnx export to fp16 or int8 and save it in the export directory.

    Usage: uv run onnx_tools\\quantize.py <output_dir> <onnx_file> [--mode fp16 | int8_static | int8_dynamic]
    Saved in <output_dir>/export/ as <onnx name>_<mode>.onnx.
    """
    args = parse_arguments()
    output = OutputDirectory(args.output_dir)
    precision, method = MODES[args.mode]
    output_path = output.export / f"{args.onnx_path.stem}_{args.mode}.onnx"
    output.export.mkdir(parents=True, exist_ok=True)

    # convert
    start = time.perf_counter()

    if args.mode == "fp16":
        model = convert_float_to_float16(onnx.load(args.onnx_path), keep_io_types=True)
        onnx.save(model, output_path)
    elif args.mode == "int8_dynamic":
        ortq.quantize_dynamic(args.onnx_path, output_path, per_channel=args.per_channel, reduce_range=args.reduce_range, weight_type=ortq.QuantType.QInt8)
    else:
        config = read_config(output.config)
        paths = find_calibration_images(config)
        reader = CalibrationImages(paths, config, load_onnx_model(args.onnx_path).image_size)
        ortq.quantize_static(
            args.onnx_path, output_path, reader, quant_format=ortq.QuantFormat.QDQ, activation_type=ortq.QuantType.QUInt8, weight_type=ortq.QuantType.QInt8,
            calibrate_method=CALIBRATION_METHODS[args.calibration_method], per_channel=args.per_channel, reduce_range=args.reduce_range, calibration_providers=["CPUExecutionProvider"]
        )

    elapsed = time.perf_counter() - start

    # print the summary
    before, after = args.onnx_path.stat().st_size, output_path.stat().st_size
    rows = [("mode", method), ("precision", precision), ("size", f"{before / 2 ** 20:.2f} MB -> {after / 2 ** 20:.2f} MB ({after / before:.2%})")]

    if args.mode == "int8_static":
        rows.append(("calibration", f"{len(paths)} validation images, {args.calibration_method}"))

    if args.mode != "fp16":
        rows += [("per-channel", str(args.per_channel)), ("reduce-range", str(args.reduce_range))]

    print(format_tree(f"MyDets ONNX conversion: {args.onnx_path.name}", [("conversion", rows)]))
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
