from dataclasses import dataclass
from pathlib import Path
import csv
import json
import os
import subprocess
import sys

import psutil

from shared.output_directory import OutputDirectory
from shared.settings import read_config

RUNS_DIR = Path("runs")


@dataclass
class TrainStatus:
    """State and progress of a training, read from its output directory."""
    name: str
    state: str
    epoch: int
    epochs: int
    best_map: float | None
    elapsed: float
    device: str | None
    model: str


def list_devices() -> list[tuple[str, str]]:
    """List the devices as (name, description), asked in a short process so that the server never opens a gpu."""
    code = "import json; from shared.device import Device; print(json.dumps(Device.list_devices()))"
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return [tuple(device) for device in json.loads(completed.stdout.strip().splitlines()[-1])]


def read_pid(pid_path: Path) -> dict | None:
    """Read a pid file, the process id, creation time and device that the server saved, None without one."""
    return json.loads(pid_path.read_text()) if pid_path.exists() else None


def find_process(pid_path: Path) -> psutil.Process | None:
    """Return the process of a pid file, None when it is not running."""
    info = read_pid(pid_path)

    if info is None or not psutil.pid_exists(info["pid"]):
        return None

    process = psutil.Process(info["pid"])

    # another process may have taken the id after the process ended, and on linux an ended child stays as a zombie until the server exits
    if abs(process.create_time() - info["create_time"]) >= 1 or process.status() == psutil.STATUS_ZOMBIE:
        return None

    return process


def read_log(output: OutputDirectory) -> list[dict[str, float]]:
    """Read log.csv, one record per finished epoch, empty when there is none."""
    if not output.log.exists():
        return []

    with open(output.log, newline="") as f:
        rows = list(csv.DictReader(f))

    # the training rewrites the whole file every epoch, so the last row may be cut while it writes
    return [{k: float(v) for k, v in row.items()} for row in rows if None not in row.values() and "" not in row.values()]


def read_status(output_dir: Path) -> TrainStatus:
    """Read the state and the progress of a training.

    Args:
        output_dir: output directory of the training.

    Returns:
        Status, with the state training, done, failed, stopped or new.
    """
    output = OutputDirectory(output_dir)
    config = read_config(output.config) if output.config.exists() else {}
    log = read_log(output)
    info = read_pid(output.pid)

    # a pid file left without its process means the training ended without the stop button
    if find_process(output.pid):
        state = "training"
    elif output.result.exists():
        state = "done"
    elif info is not None:
        state = "failed"
    elif output.last.exists():
        state = "stopped"
    else:
        state = "new"

    model = config.get("model", {})
    model_name = " · ".join(model[block]["type"] for block in ["backbone", "neck", "head"] if block in model)
    best_map = max((row["map50_95"] for row in log), default=None)
    epochs = config.get("train", {}).get("epochs", 0)
    return TrainStatus(output.path.name, state, len(log), epochs, best_map, sum(row["time"] for row in log), info["device"] if info else None, model_name)


def list_statuses() -> list[TrainStatus]:
    """Read the status of every training in runs/."""
    if not RUNS_DIR.is_dir():
        return []

    return [read_status(path) for path in sorted(RUNS_DIR.iterdir()) if (path / "config.yaml").exists() or (path / "train.pid").exists()]


def list_busy_devices() -> dict[str, str]:
    """Return the devices that running trainings use, with the training name of each."""
    busy = {}

    for path in sorted(RUNS_DIR.iterdir()) if RUNS_DIR.is_dir() else []:
        output = OutputDirectory(path)

        if find_process(output.pid):
            busy[read_pid(output.pid)["device"]] = path.name

    return busy


def start_process(arguments: list[str], console: Path, pid_path: Path, extra: dict) -> None:
    """Start a python script of the project detached from the server, with its output in a console file.

    Args:
        arguments: script and its arguments.
        console: file that gets the output of the script.
        pid_path: file that keeps the process id and creation time, to find the process again later.
        extra: other values kept in the pid file, such as the device.
    """
    # the child writes at once, in utf-8 on windows, and the progress bars less often
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "TQDM_MININTERVAL": "5"}
    detach = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {"start_new_session": True}
    # the child keeps its own handle of the console file
    with open(console, "a", encoding="utf-8") as stdout:
        process = subprocess.Popen([sys.executable, *arguments], stdout=stdout, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, **detach)

    info = {"pid": process.pid, "create_time": psutil.Process(process.pid).create_time(), **extra}
    pid_path.write_text(json.dumps(info))


def start_train(config_path: Path, device: str) -> None:
    """Start train.py on a config, detached from the server so that it keeps going when the server stops.

    Args:
        config_path: config to train, whose last.pt is resumed if there is one.
        device: device to train on, refused when a running training uses it.
    """
    busy = list_busy_devices()

    if device in busy:
        raise ValueError(f"{device} is in use by {busy[device]}")

    output = OutputDirectory.from_config(config_path)
    output.path.mkdir(parents=True, exist_ok=True)
    start_process(["train.py", str(config_path), "--device", device], output.console, output.pid, {"device": device})


def stop_process(pid_path: Path) -> None:
    """Stop the process of a pid file with its children, such as data loader workers, and remove the pid file."""
    process = find_process(pid_path)

    if process is not None:
        processes = [*process.children(recursive=True), process]

        for p in processes:
            p.terminate()

        psutil.wait_procs(processes, timeout=10)

    pid_path.unlink(missing_ok=True)


def read_console(console_path: Path, num_lines: int) -> list[str]:
    """Read the last lines of a console file as a terminal shows them, each line after its last carriage return."""
    if not console_path.exists():
        return []

    with open(console_path, encoding="utf-8", errors="replace", newline="") as f:
        text = f.read()

    # windows ends printed lines with \r\n, which must not be taken for the carriage return of a progress bar
    lines = [line.split("\r")[-1] for line in text.replace("\r\n", "\n").split("\n")]

    return [line for line in lines if line.strip()][-num_lines:]


def predict_images(output_dir: Path, input_path: Path, weights: list[str], num_images: int, seed: int, device: str) -> list[Path]:
    """Run predict.py on an image or a folder and wait for it.

    Args:
        output_dir: output directory with the best weights.
        input_path: image, or folder to pick random images from.
        weights: weights files to draw with.
        num_images: number of random images in the grid.
        seed: seed of the random images.
        device: device to run on, refused when a running training uses it.

    Returns:
        Images saved in the predict folder, one per weights file.
    """
    if device in list_busy_devices():
        raise ValueError(f"{device} is in use")

    arguments = ["predict.py", str(output_dir), str(input_path), "--weights", *weights, "--num-images", str(num_images), "--seed", str(seed), "--device", device]
    completed = subprocess.run([sys.executable, *arguments], capture_output=True, text=True, encoding="utf-8", errors="replace")

    if completed.returncode != 0:
        lines = (completed.stderr or completed.stdout).strip().splitlines()
        raise RuntimeError(lines[-1] if lines else f"predict.py exited with {completed.returncode}")

    stem = input_path.stem if input_path.is_file() else "grid"
    return [OutputDirectory(output_dir).predict / f"{stem}_{Path(name).stem}.jpg" for name in weights]


def make_report(output_dir: Path) -> Path:
    """Run report.py and wait for it, returning report.txt."""
    completed = subprocess.run([sys.executable, "report.py", str(output_dir)], capture_output=True, text=True, encoding="utf-8", errors="replace")

    if completed.returncode != 0:
        lines = (completed.stderr or completed.stdout).strip().splitlines()
        raise RuntimeError(lines[-1] if lines else f"report.py exited with {completed.returncode}")

    return OutputDirectory(output_dir).report
