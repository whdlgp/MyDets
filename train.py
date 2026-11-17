from datetime import timedelta
from pathlib import Path
import csv
import json
import math
import random
import shutil
import sys
import time

import numpy as np
import torch
import yaml
from matplotlib.figure import Figure
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from datatools.loader import create_loader
from datatools.transforms import create_transform
from metrics.evaluator import Evaluator
from models.base import Targets
from models.builder import build_model


class Trainer:
    """Train a detector from a config and save the weights, logs and curves."""

    def __init__(self, config: dict, out_dir: Path) -> None:
        """Initialize the trainer, resuming from last.pt in out_dir if it exists.

        Args:
            config: config with the common, model and train sections.
            out_dir: directory to save the outputs.
        """
        self.config = config
        self.train_config = config["train"]
        self.out_dir = out_dir
        self.dataset_dir = Path(self.train_config["dataset_dir"])
        self.names = (self.dataset_dir / "names.txt").read_text().split()
        self.epochs = self.train_config["epochs"]
        self.is_main = True

        random.seed(self.train_config["seed"])
        np.random.seed(self.train_config["seed"])
        torch.manual_seed(self.train_config["seed"])

        # model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(config["model"], len(self.names)).to(self.device)
        self.network = self.prepare_model()
        self.ema = AveragedModel(self.model, multi_avg_fn=self.update_ema, use_buffers=True) if self.train_config["ema"] else None

        # bf16 when the gpu supports it, otherwise fp16 with gradient scaling
        self.amp_dtype = None

        if self.train_config["amp"] and self.device.type == "cuda":
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp_dtype == torch.float16)

        # data
        self.train_mode = "train"
        self.train_loader, self.val_loader = self.create_loaders()

        # no weight decay on 1d parameters such as norms and biases
        decay = [p for p in self.model.parameters() if p.ndim > 1]
        no_decay = [p for p in self.model.parameters() if p.ndim <= 1]
        self.optimizer = AdamW([{"params": decay, "weight_decay": self.train_config["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}], lr=self.train_config["lr"])
        self.scheduler = self.create_scheduler()

        self.history = []
        self.best = {"val_loss": {"value": math.inf, "epoch": 0}, "map50_95": {"value": -math.inf, "epoch": 0}}

        if (self.out_dir / "last.pt").exists():
            self.load_checkpoint()

    def fit(self) -> None:
        """Train for the remaining epochs, saving the outputs after every epoch."""
        start_epoch = len(self.history)
        epoch_times = []

        if self.is_main:
            print(f"amp: {str(self.amp_dtype).removeprefix('torch.') if self.amp_dtype else 'off'}, device: {self.device}")

            if start_epoch:
                print(f"resumed from epoch {start_epoch}: {self.out_dir}")

        for epoch in range(start_epoch, self.epochs):
            start = time.time()
            self.set_frozen(epoch < self.train_config["freeze_epochs"])

            # the last epochs train without the mosaic
            mode = "train_no_mosaic" if epoch >= self.epochs - self.train_config["close_mosaic_epochs"] else "train"

            if mode != self.train_mode:
                self.train_mode = mode
                self.train_loader, _ = self.create_loaders()

            record = {"epoch": epoch + 1, **self.train_one_epoch(epoch), **self.evaluate(epoch)}
            record["time"] = time.time() - start
            self.history.append(record)
            epoch_times.append(record["time"])
            improved = self.update_best(record)

            if self.is_main:
                self.save_checkpoint()
                self.save_log()
                self.plot_curve()
                self.print_summary(record, improved, epoch_times)

        if self.is_main:
            self.save_result()

    # ==========================================================
    # One epoch
    # ==========================================================

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: 0-based epoch.

        Returns:
            Mean of each loss over the images, with keys prefixed by train_.
        """
        self.model.train()
        state = {"epoch": epoch, "num_epochs": self.epochs}
        sums, count = {}, 0
        progress = tqdm(self.train_loader, desc=f"epoch {epoch + 1}/{self.epochs} train", disable=not self.is_main, leave=False)

        for images, targets in progress:
            images = images.to(self.device, non_blocking=True)
            targets = self.move_targets(targets)

            # only the forward runs in mixed precision, and the losses in float32
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                outputs = self.network(images, targets)

            outputs = {k: v.float() if v.is_floating_point() else v for k, v in outputs.items()}
            losses = self.model.head.loss(outputs, targets, state)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            if self.ema is not None:
                self.ema.update_parameters(self.model)

            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + value.item() * len(targets)

            count += len(targets)
            progress.set_postfix(loss=f"{sums['loss'] / count:.3f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

        return {f"train_{name}": value / count for name, value in sums.items()}

    @torch.no_grad()
    def evaluate(self, epoch: int) -> dict[str, float]:
        """Compute the val losses and the mAP, with the EMA weights when EMA is on.

        Args:
            epoch: 0-based epoch.

        Returns:
            Mean of each val loss with keys prefixed by val_, map50 and map50_95.
        """
        # the head of the model computes the losses and the detections from the outputs
        network = self.ema.module if self.ema is not None else self.network
        network.eval()
        self.model.eval()
        state = {"epoch": epoch, "num_epochs": self.epochs}
        evaluator = Evaluator(len(self.names))
        sums, count = {}, 0

        for images, targets in tqdm(self.val_loader, desc=f"epoch {epoch + 1}/{self.epochs} val", disable=not self.is_main, leave=False):
            images = images.to(self.device, non_blocking=True)
            targets = self.move_targets(targets)

            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                outputs = network(images)

            outputs = {k: v.float() if v.is_floating_point() else v for k, v in outputs.items()}
            losses = self.model.head.loss(outputs, targets, state)
            evaluator.update(self.model.head.decode(outputs), targets, (images.shape[-1], images.shape[-2]))

            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + value.item() * len(targets)

            count += len(targets)

        return {**{f"val_{name}": value / count for name, value in sums.items()}, **evaluator.compute()}

    def update_best(self, record: dict[str, float]) -> list[str]:
        """Update the bests with an epoch and save the new best weights.

        Args:
            record: losses and mAP of one epoch.

        Returns:
            Names of the improved values, val_loss and map50_95.
        """
        improved = []

        if record["val_loss"] < self.best["val_loss"]["value"]:
            improved.append("val_loss")

        if record["map50_95"] > self.best["map50_95"]["value"]:
            improved.append("map50_95")

        weights = (self.ema.module if self.ema is not None else self.model).state_dict()

        for name in improved:
            self.best[name] = {"value": record[name], "epoch": record["epoch"]}

            if self.is_main:
                torch.save(weights, self.out_dir / {"val_loss": "best_loss.pt", "map50_95": "best_map.pt"}[name])

        return improved

    # ==========================================================
    # Saving and printing
    # ==========================================================

    def save_checkpoint(self) -> None:
        """Save the state to resume (last.pt)."""
        torch.save({
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "history": self.history,
            "best": self.best,
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }, self.out_dir / "last.pt")

    def load_checkpoint(self) -> None:
        """Restore the state from last.pt."""
        state = torch.load(self.out_dir / "last.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])

        if self.ema is not None:
            self.ema.load_state_dict(state["ema"])

        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.history = state["history"]
        self.best = state["best"]
        random.setstate(state["random"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])

        if state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def save_log(self) -> None:
        """Save the records of the epochs (log.csv)."""
        with open(self.out_dir / "log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)

    def plot_curve(self) -> None:
        """Save the loss curves and the mAP curves (curve.png)."""
        epochs = [h["epoch"] for h in self.history]
        fig = Figure(figsize=(10, 4))
        loss_ax, map_ax = fig.subplots(1, 2)

        for name in ["train_loss", "val_loss"]:
            loss_ax.plot(epochs, [h[name] for h in self.history], label=name)

        for name in ["map50", "map50_95"]:
            map_ax.plot(epochs, [h[name] for h in self.history], label=name)

        for ax, label in [(loss_ax, "loss"), (map_ax, "mAP")]:
            ax.set_xlabel("epoch")
            ax.set_ylabel(label)
            ax.legend()

        fig.savefig(self.out_dir / "curve.png", dpi=120, bbox_inches="tight")

    def print_summary(self, record: dict[str, float], improved: list[str], epoch_times: list[float]) -> None:
        """Print the losses, the mAP and the bests of an epoch as a tree.

        Args:
            record: losses and mAP of the epoch.
            improved: names of the values that became the best.
            epoch_times: seconds of the epochs trained in this run, for the remaining time.
        """
        elapsed = sum(h["time"] for h in self.history)
        remaining = (self.epochs - record["epoch"]) * sum(epoch_times) / len(epoch_times)
        lines = []

        for split in ["train", "val"]:
            parts = [f"{name.removeprefix(f'{split}_loss_')} {value:.3f}" for name, value in record.items() if name.startswith(f"{split}_loss_")]
            lines.append((f"{split} loss", f"{record[f'{split}_loss']:.3f}  ({', '.join(parts)})", f"{split}_loss"))

        lines += [("map50", f"{record['map50']:.3f}", "map50"), ("map50_95", f"{record['map50_95']:.3f}", "map50_95")]
        lines += [(f"best {name.replace('_loss', ' loss')}", f"{best['value']:.3f}  (epoch {best['epoch']})", None) for name, best in self.best.items()]

        width = max(len(name) for name, _, _ in lines)
        text = [f"epoch {record['epoch']}/{self.epochs}  {record['time']:.0f}s, elapsed {timedelta(seconds=round(elapsed))}, eta {timedelta(seconds=round(remaining))}"]

        for i, (name, value, key) in enumerate(lines):
            mark = "  new best" if key in improved else ""
            text.append(f"{'└─' if i == len(lines) - 1 else '├─'} {name:<{width}}  {value}{mark}")

        tqdm.write("\n".join(text))

    def save_result(self) -> None:
        """Save the records of the two bests (result.json)."""
        result = {f"best_{name}": self.history[best["epoch"] - 1] for name, best in self.best.items()}

        with open(self.out_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

    # ==========================================================
    # Setup
    # ==========================================================

    def prepare_model(self) -> nn.Module:
        """Return the model that runs forward, which multi gpu training would wrap."""
        return self.model

    def create_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Create the train loader of the current mode and the val loader."""
        common = self.config["common"]
        transform_args = (common["image_size"], common["mean"], common["std"])
        loader_args = (self.train_config["batch_size"], self.train_config["num_workers"])
        train_loader = create_loader(self.dataset_dir / "train", create_transform(self.train_mode, *transform_args), loader_args[0], True, loader_args[1])
        val_loader = create_loader(self.dataset_dir / "val", create_transform("val", *transform_args), loader_args[0], False, loader_args[1])
        return train_loader, val_loader

    def create_scheduler(self) -> SequentialLR:
        """Create a linear warmup followed by a cosine decay to 0, stepped every batch."""
        steps_per_epoch = len(self.train_loader)
        warmup_steps = max(self.train_config["warmup_epochs"] * steps_per_epoch, 1)
        cosine_steps = max(self.epochs * steps_per_epoch - warmup_steps, 1)
        warmup = LinearLR(self.optimizer, start_factor=1 / warmup_steps, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=cosine_steps)
        return SequentialLR(self.optimizer, [warmup, cosine], milestones=[warmup_steps])

    def set_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze the backbone when the state changes."""
        if self.model.backbone.frozen != frozen:
            self.model.backbone.freeze(frozen)

    # ==========================================================
    # Helpers
    # ==========================================================

    def move_targets(self, targets: list[Targets]) -> list[Targets]:
        """Move the targets of a batch to the device."""
        return [Targets(t.boxes.to(self.device), t.labels.to(self.device), {k: v.to(self.device) for k, v in t.extras.items()}) for t in targets]

    @staticmethod
    def update_ema(averaged: list[Tensor], current: list[Tensor], num_averaged: Tensor) -> None:
        """Move the averaged weights toward the current ones, with a decay growing to 0.9998.

        Args:
            averaged: averaged weights of one device and dtype, updated in place.
            current: current weights of the model.
            num_averaged: number of updates so far.
        """
        # integer buffers such as num_batches_tracked follow the model
        if not averaged[0].is_floating_point():
            for a, c in zip(averaged, current):
                a.copy_(c)

            return

        decay = 0.9998 * (1 - math.exp(-(num_averaged.item() + 1) / 2000))

        for a, c in zip(averaged, current):
            a.lerp_(c, 1 - decay)


def main():
    """Train a detector from a config file.

    The outputs and copies of the config and the class names are saved in runs/<config name>/, and an existing last.pt there is resumed.

    Usage: uv run train.py configs\\<config>.yaml
    """
    # read the arguments
    if len(sys.argv) != 2:
        raise SystemExit("usage: uv run train.py configs\\<config>.yaml")

    config_path = Path(sys.argv[1])
    config = yaml.safe_load(config_path.read_text())
    out_dir = Path("runs") / config_path.stem

    # keep what inference needs next to the weights
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.yaml")
    shutil.copy(Path(config["train"]["dataset_dir"]) / "names.txt", out_dir / "names.txt")

    Trainer(config, out_dir).fit()
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
