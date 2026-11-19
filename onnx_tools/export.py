from pathlib import Path
from typing import NoReturn
import argparse
import copy
import json
import sys

# the scripts of onnx_tools find the modules of the project root like the scripts at the root do
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import onnx
import onnxsim
import torch
from torch import Tensor, Value, nn
from torch.onnx import symbolic_helper
from torch.onnx._internal.torchscript_exporter.jit_utils import GraphContext

from models.builder import build_model
from report import format_tree
from shared.output_directory import OutputDirectory
from shared.settings import CommonSettings, read_config, read_settings

CONFIG = {
    "weights": ["best_loss.pt", "best_map.pt"],
    "opset": 19
}


class ExportModel(nn.Module):
    """Detector with the input normalization inside, from RGB pixel values in [0, 255] to the raw head outputs."""

    def __init__(self, model: nn.Module, mean: tuple[float, float, float], std: tuple[float, float, float], output_names: list[str]) -> None:
        """Initialize the model.

        Args:
            model: detector in eval mode.
            mean: mean of each RGB channel, for pixel values in [0, 1].
            std: standard deviation of each RGB channel, for pixel values in [0, 1].
            output_names: keys of the head outputs, in the order of the onnx outputs.
        """
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).reshape(1, 3, 1, 1))
        self.output_names = output_names

    def forward(self, images: Tensor) -> tuple[Tensor, ...]:
        """Return the raw head outputs in the order of output_names."""
        outputs = self.model((images / 255 - self.mean) / self.std)
        return tuple(outputs[name] for name in self.output_names)


class DeformConv2dExport:
    """Exports torchvision deform_conv2d to onnx, as the builtin DeformConv from opset 19 and as GridSample with Conv from opset 16."""

    BUILTIN_OPSET = 19
    MANUAL_OPSET = 16

    # the arguments of torchvision deform_conv2d: 5 tensors, 8 ints and use_mask
    ARGUMENTS = ("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "b")

    @staticmethod
    def count(model: ExportModel, images: Tensor) -> int:
        """Trace the model and count the deform_conv2d calls."""
        graph = torch.jit.trace(model, (images,), check_trace=False).inlined_graph
        return sum(node.kind() == "torchvision::deform_conv2d" for node in graph.nodes())

    @staticmethod
    def convert(g: GraphContext, *args: Value) -> Value:
        """Write the onnx nodes of one deform_conv2d call, builtin from BUILTIN_OPSET and manual before, the function registered to torch.onnx."""
        if g.opset >= DeformConv2dExport.BUILTIN_OPSET:
            return DeformConv2dExport.builtin(g, *args)

        return DeformConv2dExport.manual(g, *args)

    @staticmethod
    @symbolic_helper.parse_args(*ARGUMENTS)
    def builtin(g: GraphContext, features: Value, weight: Value, offset: Value, mask: Value, bias: Value,
                stride_h: int, stride_w: int, pad_h: int, pad_w: int, dilation_h: int, dilation_w: int,
                groups: int, offset_groups: int, use_mask: bool) -> Value:
        """Write deform_conv2d as the onnx DeformConv, which takes the offsets and the mask in the same layout."""
        kernel_shape = symbolic_helper._get_tensor_sizes(weight)[-2:]
        inputs = [features, weight, offset, bias] + ([mask] if use_mask else [])
        return g.op("DeformConv", *inputs, dilations_i=[dilation_h, dilation_w], group_i=groups, kernel_shape_i=kernel_shape,
                    offset_group_i=offset_groups, pads_i=[pad_h, pad_w, pad_h, pad_w], strides_i=[stride_h, stride_w])

    @staticmethod
    @symbolic_helper.parse_args(*ARGUMENTS)
    def manual(g: GraphContext, features: Value, weight: Value, offset: Value, mask: Value, bias: Value,
               stride_h: int, stride_w: int, pad_h: int, pad_w: int, dilation_h: int, dilation_w: int,
               groups: int, offset_groups: int, use_mask: bool) -> Value:
        """Write deform_conv2d as GridSample at every kernel position and a grouped 1x1 Conv.

        The sampled values are the columns of the conv, (batch, channels * kernel size, out height, out width).
        GridSample with zero padding gives 0 to each bilinear corner outside the input, like torchvision.
        """
        out_channels, group_channels, kernel_height, kernel_width = symbolic_helper._get_tensor_sizes(weight)
        channels = symbolic_helper._get_tensor_sizes(features)[1]
        group_size = channels // offset_groups
        kernel_size = kernel_height * kernel_width
        float_type = onnx.TensorProto.FLOAT
        one, two = torch.tensor(1.0), torch.tensor(2.0)

        # sizes as 1-d tensors for the shapes, scalars for Range and floats for the grid
        batch = g.op("Shape", offset, start_i=0, end_i=1)
        out_height = g.op("Shape", offset, start_i=2, end_i=3)
        out_width = g.op("Shape", offset, start_i=3, end_i=4)
        out_height_scalar = g.op("Squeeze", out_height)
        out_width_scalar = g.op("Squeeze", out_width)
        in_height = g.op("Shape", features, start_i=2, end_i=3)
        in_width = g.op("Shape", features, start_i=3, end_i=4)
        in_height = g.op("Cast", in_height, to_i=float_type)
        in_width = g.op("Cast", in_width, to_i=float_type)

        # output pixel to input pixel, (out height, 1) rows and (out width,) columns
        rows = g.op("Range", torch.tensor(0), out_height_scalar, torch.tensor(1))
        rows = g.op("Cast", rows, to_i=float_type)
        rows = g.op("Mul", rows, torch.tensor(float(stride_h)))
        rows = g.op("Sub", rows, torch.tensor(float(pad_h)))
        row_axis = g.op("Constant", value_t=torch.tensor([1]))
        rows = g.op("Unsqueeze", rows, row_axis)
        columns = g.op("Range", torch.tensor(0), out_width_scalar, torch.tensor(1))
        columns = g.op("Cast", columns, to_i=float_type)
        columns = g.op("Mul", columns, torch.tensor(float(stride_w)))
        columns = g.op("Sub", columns, torch.tensor(float(pad_w)))

        # kernel positions, (kernel size, 1, 1, 1)
        kernel_rows = [float(i * dilation_h) for i in range(kernel_height) for _ in range(kernel_width)]
        kernel_columns = [float(j * dilation_w) for _ in range(kernel_height) for j in range(kernel_width)]
        kernel_rows = g.op("Constant", value_t=torch.tensor(kernel_rows).reshape(-1, 1, 1, 1))
        kernel_columns = g.op("Constant", value_t=torch.tensor(kernel_columns).reshape(-1, 1, 1, 1))

        # offsets as (batch, offset groups, kernel size, (y, x), out height, out width)
        offset_layout = g.op("Constant", value_t=torch.tensor([offset_groups, kernel_size, 2]))
        offset_shape = g.op("Concat", batch, offset_layout, out_height, out_width, axis_i=0)
        offset = g.op("Reshape", offset, offset_shape)
        split_sizes = g.op("Constant", value_t=torch.tensor([1, 1]))
        offset_y, offset_x = g.op("Split", offset, split_sizes, axis_i=3, outputs=2)

        # sampling position, (batch, offset groups, kernel size, 1, out height, out width)
        y = g.op("Add", offset_y, kernel_rows)
        y = g.op("Add", y, rows)
        x = g.op("Add", offset_x, kernel_columns)
        x = g.op("Add", x, columns)

        # pixel position to the grid of GridSample without align_corners, (2 * position + 1) / size - 1
        grid_y = g.op("Mul", y, two)
        grid_y = g.op("Add", grid_y, one)
        grid_y = g.op("Div", grid_y, in_height)
        grid_y = g.op("Sub", grid_y, one)
        grid_x = g.op("Mul", x, two)
        grid_x = g.op("Add", grid_x, one)
        grid_x = g.op("Div", grid_x, in_width)
        grid_x = g.op("Sub", grid_x, one)

        # grid of (batch * offset groups, kernel size * out height, out width, (x, y))
        grid = g.op("Concat", grid_x, grid_y, axis_i=3)
        grid = g.op("Transpose", grid, perm_i=[0, 1, 2, 4, 5, 3])
        grid_batch = g.op("Constant", value_t=torch.tensor([-1]))
        grid_height = g.op("Mul", out_height, torch.tensor(kernel_size))
        grid_last = g.op("Constant", value_t=torch.tensor([2]))
        grid_shape = g.op("Concat", grid_batch, grid_height, out_width, grid_last, axis_i=0)
        grid = g.op("Reshape", grid, grid_shape)

        # each offset group samples its own channels, (batch * offset groups, group size, kernel size * out height, out width)
        group_layout = g.op("Constant", value_t=torch.tensor([-1, group_size]))
        in_size = g.op("Shape", features, start_i=2)
        group_shape = g.op("Concat", group_layout, in_size, axis_i=0)
        grouped = g.op("Reshape", features, group_shape)
        sampled = g.op("GridSample", grouped, grid, mode_s="bilinear", padding_mode_s="zeros", align_corners_i=0)

        # sampled values as (batch, offset groups, group size, kernel size, out height, out width), with the masks of each group
        sampled_layout = g.op("Constant", value_t=torch.tensor([offset_groups, group_size, kernel_size]))
        sampled_shape = g.op("Concat", batch, sampled_layout, out_height, out_width, axis_i=0)
        sampled = g.op("Reshape", sampled, sampled_shape)

        if use_mask:
            mask_layout = g.op("Constant", value_t=torch.tensor([offset_groups, 1, kernel_size]))
            mask_shape = g.op("Concat", batch, mask_layout, out_height, out_width, axis_i=0)
            mask = g.op("Reshape", mask, mask_shape)
            sampled = g.op("Mul", sampled, mask)

        # the columns in the order of the flattened weight, channel then kernel position
        columns_layout = g.op("Constant", value_t=torch.tensor([channels * kernel_size]))
        columns_shape = g.op("Concat", batch, columns_layout, out_height, out_width, axis_i=0)
        sampled = g.op("Reshape", sampled, columns_shape)
        weight_shape = g.op("Constant", value_t=torch.tensor([out_channels, group_channels * kernel_size, 1, 1]))
        weight = g.op("Reshape", weight, weight_shape)
        return g.op("Conv", sampled, weight, bias, group_i=groups, kernel_shape_i=[1, 1])


def build_export_model(config: dict, num_classes: int, image_size: tuple[int, int]) -> ExportModel:
    """Build the model of a config without pretrained weights, with the output names read from one run.

    Args:
        config: config of the training.
        num_classes: number of object classes.
        image_size: (width, height) of the run.

    Returns:
        Model in eval mode, whose weights are loaded after.
    """
    model_config = copy.deepcopy(config["model"])
    model_config["backbone"].setdefault("args", {})["pretrained"] = False
    detector = build_model(model_config, num_classes).eval()
    common = read_settings(CommonSettings, config, "common")
    width, height = image_size

    images = torch.zeros(1, 3, height, width)

    with torch.no_grad():
        output_names = list(detector(images))

    return ExportModel(detector, common.mean, common.std, output_names).eval()


def find_dynamic_axes(model: ExportModel, image_size: tuple[int, int]) -> dict[str, dict[int, str]]:
    """Find the axes of the input and the outputs that change with the batch size and the input size.

    Args:
        model: model to export.
        image_size: (width, height) of the export, compared with batch 2 at a larger size of multiples of the largest stride.

    Returns:
        Dynamic axes of each input and output name, in the form torch.onnx.export takes.
    """
    width, height = image_size
    stride = max(model.model.neck.out_strides)

    with torch.no_grad():
        first = model(torch.zeros(1, 3, height, width))
        second = model(torch.zeros(2, 3, height + stride, width + 2 * stride))

    axes = {"images": {0: "batch", 2: "height", 3: "width"}}

    # the batch axis goes from 1 to 2, the other changed axes follow the input size
    for name, a, b in zip(model.output_names, first, second):
        changed = [i for i in range(a.dim()) if a.shape[i] != b.shape[i]]

        if changed:
            axes[name] = {i: "batch" if (a.shape[i], b.shape[i]) == (1, 2) else f"{name}_{i}" for i in changed}

    return axes


def create_metadata(config: dict, names: list[str], weights_name: str, image_size: tuple[int, int], dynamic: bool, strides: list[int]) -> dict[str, str]:
    """Create the metadata saved in the onnx file, which verify.py and inference.py read.

    Args:
        config: config of the training.
        names: class names.
        weights_name: weights file the onnx is exported from.
        image_size: (width, height) of the export.
        dynamic: dynamic batch size and input size.
        strides: strides of the levels of the head.

    Returns:
        Metadata of string keys and values.
    """
    return {
        "weights": weights_name,
        "input": "images: RGB pixel values in [0, 255], float32, (batch, 3, height, width), normalized inside",
        "image_size": "dynamic" if dynamic else json.dumps(image_size),
        "strides": json.dumps(strides),
        "head": config["model"]["head"]["type"],
        "names": json.dumps(names)
    }


def export_onnx(model: ExportModel, path: Path, image_size: tuple[int, int], dynamic: bool, opset: int, metadata: dict[str, str], simplify: bool) -> None:
    """Export the model to an onnx file with a fixed input size, or a dynamic batch size and input size.

    Args:
        model: model to export.
        path: onnx file to save.
        image_size: (width, height) of the traced input.
        dynamic: dynamic batch size and input size.
        opset: onnx opset version.
        metadata: values saved in the metadata of the onnx file.
        simplify: simplify the graph with onnxsim.
    """
    width, height = image_size
    images = torch.zeros(1, 3, height, width)
    dynamic_axes = None

    if dynamic:
        dynamic_axes = find_dynamic_axes(model, image_size)

    # torch.onnx does not know the torchvision op
    torch.onnx.register_custom_op_symbolic("torchvision::deform_conv2d", DeformConv2dExport.convert, DeformConv2dExport.MANUAL_OPSET)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (images,), str(path),
        input_names=["images"], output_names=model.output_names, dynamic_axes=dynamic_axes,
        opset_version=opset, do_constant_folding=True, dynamo=False
    )
    onnx_model = onnx.load(str(path))

    if simplify:
        onnx_model, valid = onnxsim.simplify(onnx_model)

        if not valid:
            raise SystemExit(f"onnxsim could not check the simplified graph of {path.name}, export again without --simplify")

    onnx.helper.set_model_props(onnx_model, metadata)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(path))


def exit_with_usage(error: str | None = None) -> NoReturn:
    """Exit with the usage and the expected directory structure.

    Args:
        error: error message printed first.
    """
    lines = [error, ""] if error else []
    lines += [
        "usage: uv run onnx_tools\\export.py <output_dir> [--weights ...] [--image-size W H | --dynamic] [--opset N] [--simplify]",
        "",
        "output_dir",
        *[f"├─ {name}" for name in CONFIG["weights"]],
        "├─ config.yaml",
        "└─ names.txt"
    ]
    raise SystemExit("\n".join(lines))


def parse_arguments() -> argparse.Namespace:
    """Read the arguments and exit with the usage when they are wrong."""
    parser = argparse.ArgumentParser(description="Export the best models of a training to onnx.")
    parser.add_argument("output_dir", type=Path, help="output directory of train.py, such as runs\\<config name>")
    parser.add_argument("--weights", nargs="+", choices=CONFIG["weights"], default=CONFIG["weights"], help="weights to export")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), help="fixed input size, the image_size of the config by default")
    parser.add_argument("--dynamic", action="store_true", help="dynamic batch size and input size instead of a fixed input size")
    parser.add_argument("--opset", type=int, default=CONFIG["opset"], help="onnx opset version")
    parser.add_argument("--simplify", action="store_true", help="simplify the graph with onnxsim")
    args = parser.parse_args()

    if args.image_size and args.dynamic:
        exit_with_usage("--image-size is the fixed input size, it cannot go with --dynamic")

    if args.image_size and min(args.image_size) < 1:
        exit_with_usage(f"--image-size must be positive, got {args.image_size}")

    missing = [name for name in [*args.weights, "config.yaml", "names.txt"] if not (args.output_dir / name).exists()]

    if missing:
        exit_with_usage(f"no {', '.join(missing)} in {args.output_dir}")

    return args


def main():
    """Export the best models of a training to onnx with the raw head outputs.

    Usage: uv run onnx_tools\\export.py <output_dir> [--weights best_map.pt] [--image-size 640 480 | --dynamic] [--opset 19] [--simplify]
    Saved in <output_dir>/export/ as <weights name>_<width>x<height>.onnx, or <weights name>_dynamic.onnx.
    """
    args = parse_arguments()
    output = OutputDirectory(args.output_dir)

    # read the training
    config = read_config(output.config)
    names = output.names.read_text().splitlines()
    image_size = tuple(args.image_size or read_settings(CommonSettings, config, "common").image_size)
    width, height = image_size

    # deform_conv2d needs DeformConv, or GridSample for the manual version
    model = build_export_model(config, len(names), image_size)
    calls = 0

    if args.opset < DeformConv2dExport.BUILTIN_OPSET:
        calls = DeformConv2dExport.count(model, torch.zeros(1, 3, height, width))

    if calls and args.opset < DeformConv2dExport.MANUAL_OPSET:
        exit_with_usage(f"opset {args.opset} cannot export deform_conv2d, use --opset {DeformConv2dExport.MANUAL_OPSET} or higher")

    # export each best model
    size_name = "dynamic" if args.dynamic else f"{width}x{height}"
    saved = []

    for weights_name in args.weights:
        model.model.load_state_dict(OutputDirectory.load(output.path / weights_name))
        path = output.export / f"{Path(weights_name).stem}_{size_name}.onnx"
        metadata = create_metadata(config, names, weights_name, image_size, args.dynamic, model.model.neck.out_strides)
        export_onnx(model, path, image_size, args.dynamic, args.opset, metadata, args.simplify)
        saved.append(path.name)

    # print the summary
    input_shape = "(batch, 3, height, width)" if args.dynamic else f"(1, 3, {height}, {width})"
    simplified = ", simplified" if args.simplify else ""
    sections = [("onnx", [
        ("input", f"images  RGB [0, 255]  {input_shape}"),
        ("outputs", ", ".join(model.output_names)),
        ("opset", f"{args.opset}{simplified}")
    ])]

    if calls:
        sections.append((f"opset {args.opset} has no DeformConv, deform_conv2d replaced", [
            ("calls", f"{calls}, as GridSample at each kernel position and a grouped 1x1 Conv"),
            ("cost", "slower than DeformConv, and the samples take kernel size times the memory of the conv input, 9 for 3x3"),
            ("DeformConv", f"--opset {DeformConv2dExport.BUILTIN_OPSET} or higher")
        ]))

    print(format_tree(f"MyDets onnx export: {output.path.name}", sections))
    print(f"\nSaved: {output.export} ({', '.join(saved)})")


if __name__ == "__main__":
    main()
