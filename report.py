from datetime import timedelta
from pathlib import Path
from typing import NoReturn
import copy
import csv
import json
import sys

import yaml

from datatools.loader import IMAGE_SUFFIXES
from models.builder import build_model


def exit_with_usage(error: str | None = None) -> NoReturn:
    """Exit with the usage and the expected directory structure.

    Args:
        error: error message printed first.
    """
    lines = [error, ""] if error else []
    lines += [
        "usage: uv run report.py <run_dir>",
        "",
        "run_dir",
        "├─ config.yaml",
        "├─ names.txt",
        "├─ log.csv",
        "└─ result.json"
    ]
    raise SystemExit("\n".join(lines))


def format_tree(title: str, sections: list[tuple[str, list[tuple[str, str]]]]) -> str:
    """Format sections of key and value rows as a tree.

    Args:
        title: first line.
        sections: (section name, rows of (key, value)).

    Returns:
        Tree text.
    """
    width = max(len(key) for _, rows in sections for key, _ in rows)
    lines = [title, ""]

    for i, (name, rows) in enumerate(sections):
        last_section = i == len(sections) - 1
        lines.append(f"{'└─' if last_section else '├─'} {name}")

        for j, (key, value) in enumerate(rows):
            branch = "└─" if j == len(rows) - 1 else "├─"
            lines.append(f"{'   ' if last_section else '│  '}{branch} {key.ljust(width)}  {value}")

    return "\n".join(lines)


def count_images(dataset_dir: Path) -> str | None:
    """Count the train and val images of a dataset, or None when the dataset is not there anymore."""
    image_dirs = [dataset_dir / split / "images" for split in ["train", "val"]]

    if not all(d.is_dir() for d in image_dirs):
        return None

    train, val = [sum(p.suffix.lower() in IMAGE_SUFFIXES for p in d.iterdir()) for d in image_dirs]
    return f"train {train}, val {val}"


def count_parameters(model_config: dict, num_classes: int) -> str:
    """Count the parameters of the model, built without downloading pretrained weights."""
    model_config = copy.deepcopy(model_config)
    model_config["backbone"].setdefault("args", {})["pretrained"] = False
    model = build_model(model_config, num_classes)
    return f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"


def describe_module(module_config: dict) -> str:
    """Name a module by its type, with the timm model name for a backbone."""
    name = (module_config.get("args") or {}).get("name")
    return f"{module_config['type']} ({name})" if name else module_config["type"]


def main():
    """Summarize a training run in a report, from the files train.py saved.

    Usage: uv run report.py <run_dir>
    Saved as <run_dir>/report.txt.
    """
    # read the arguments
    if len(sys.argv) != 2:
        exit_with_usage()

    run_dir = Path(sys.argv[1])
    missing = [name for name in ["config.yaml", "names.txt", "log.csv", "result.json"] if not (run_dir / name).exists()]

    if missing:
        exit_with_usage(f"no {', '.join(missing)} in {run_dir}")

    # read the run
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    names = (run_dir / "names.txt").read_text().splitlines()
    result = json.loads((run_dir / "result.json").read_text())

    with open(run_dir / "log.csv") as f:
        log = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]

    model, train, common = config["model"], config["train"], config["common"]
    best_map, best_loss, last = result["best_map50_95"], result["best_val_loss"], log[-1]
    width, height = common["image_size"]

    # the image counts of the dataset folder now, only when it is still there
    data_rows = [("dataset", f"{train['dataset_dir']}, {len(names)} classes")]
    images = count_images(Path(train["dataset_dir"]))

    if images:
        data_rows.append(("images", images))

    sections = [
        ("model", [
            ("backbone", describe_module(model["backbone"])),
            ("neck", describe_module(model["neck"])),
            ("head", describe_module(model["head"])),
            ("parameters", count_parameters(model, len(names)))
        ]),
        ("data", data_rows),
        ("training", [
            ("epochs", f"{len(log)} of {train['epochs']}, batch {train['batch_size']}, input {width}x{height}"),
            ("schedule", f"warmup {train['warmup_epochs']}, freeze {train['freeze_epochs']}, close mosaic {train['close_mosaic_epochs']} (last {train['close_mosaic_epochs'] + 1} epochs)"),
            ("optimizer", f"AdamW, lr {train['lr']}, weight decay {train['weight_decay']}"),
            ("ema / amp", f"{'on' if train['ema'] else 'off'} / {'on' if train['amp'] else 'off'}"),
            ("time", str(timedelta(seconds=round(sum(row["time"] for row in log)))))
        ]),
        ("results", [
            ("best map50_95", f"{best_map['map50_95']:.3f}  (epoch {int(best_map['epoch'])}, map50 {best_map['map50']:.3f}, best_map.pt)"),
            ("best val loss", f"{best_loss['val_loss']:.3f}  (epoch {int(best_loss['epoch'])}, best_loss.pt)"),
            ("last epoch", f"map50 {last['map50']:.3f}, map50_95 {last['map50_95']:.3f}")
        ])
    ]

    report = format_tree(f"MyDets report: {run_dir.name}", sections)
    (run_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSaved: {run_dir / 'report.txt'}")


if __name__ == "__main__":
    main()