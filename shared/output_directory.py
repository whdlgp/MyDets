from pathlib import Path
from typing import Self

import torch
from torch import Tensor


class OutputDirectory:
    """Files that training, prediction, the report and the onnx tools save in the output directory of a config."""

    def __init__(self, path: Path) -> None:
        """Initialize the paths.

        Args:
            path: output directory.
        """
        self.path = Path(path)
        self.config = self.path / "config.yaml"
        self.names = self.path / "names.txt"
        self.last = self.path / "last.pt"
        self.best_loss = self.path / "best_loss.pt"
        self.best_map = self.path / "best_map.pt"
        self.log = self.path / "log.csv"
        self.curve = self.path / "curve.png"
        self.result = self.path / "result.json"
        self.report = self.path / "report.txt"
        self.predict = self.path / "predict"
        self.export = self.path / "export"
        self.pid = self.path / "train.pid"
        self.console = self.path / "console.log"

    @classmethod
    def from_config(cls, config_path: Path) -> Self:
        """Return the output directory of a config, runs/<config name>."""
        return cls(Path("runs") / Path(config_path).stem)

    @staticmethod
    def save(value: object, path: Path) -> None:
        """Save weights or a checkpoint with every tensor moved to the cpu, so that it loads on any device.

        Args:
            value: tensor, or dicts, lists and tuples of tensors and other values.
            path: file to save.
        """
        torch.save(OutputDirectory.move_to_cpu(value), path)

    @staticmethod
    def move_to_cpu(value: object) -> object:
        """Move every tensor in nested dicts, lists and tuples to the cpu."""
        if isinstance(value, Tensor):
            return value.cpu()

        if isinstance(value, dict):
            return {k: OutputDirectory.move_to_cpu(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return type(value)(OutputDirectory.move_to_cpu(v) for v in value)

        return value

    @staticmethod
    def load(path: Path, weights_only: bool = True) -> object:
        """Load weights or a checkpoint on the cpu, also files that earlier versions saved with gpu tensors.

        Args:
            path: file saved by save.
            weights_only: refuse pickled objects other than tensors and plain containers.

        Returns:
            Loaded value, with the tensors on the cpu.
        """
        return torch.load(path, map_location="cpu", weights_only=weights_only)
