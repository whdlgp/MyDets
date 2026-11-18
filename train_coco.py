from pathlib import Path
import subprocess
import sys

import yaml


def main():
    """Download COCO 2017, train a model on it, draw its detections on random val images and report the run.

    Usage: uv run train_coco.py <config>
    Saved in runs/<config name>/.
    """
    # read the arguments
    if len(sys.argv) != 2:
        raise SystemExit("usage: uv run train_coco.py <config>")

    config_path = Path(sys.argv[1])
    dataset_dir = Path(yaml.safe_load(config_path.read_text())["train"]["dataset_dir"])
    run_dir = Path("runs") / config_path.stem

    print("Preparing COCO 2017")
    subprocess.run([sys.executable, "datatools/download_coco.py"], check=True)

    print("\nTraining")
    subprocess.run([sys.executable, "train.py", str(config_path)], check=True)

    print("\nPredicting")
    subprocess.run([sys.executable, "predict.py", str(run_dir), str(dataset_dir / "val" / "images")], check=True)

    print("\nReport")
    subprocess.run([sys.executable, "report.py", str(run_dir)], check=True)


if __name__ == "__main__":
    main()
