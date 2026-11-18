from pathlib import Path
import subprocess
import sys

import yaml

from shared.output_directory import OutputDirectory
from shared.settings import TrainSettings, read_settings


def main():
    """Download PASCAL VOC, train a model on it, draw its detections on random val images and report the training.

    Usage: uv run train_voc.py <config>
    Saved in runs/<config name>/.
    """
    # read the arguments
    if len(sys.argv) != 2:
        raise SystemExit("usage: uv run train_voc.py <config>")

    config_path = Path(sys.argv[1])
    config = yaml.safe_load(config_path.read_text())
    dataset_dir = Path(read_settings(TrainSettings, config, "train").dataset_dir)
    output = OutputDirectory.from_config(config_path)

    print("Preparing PASCAL VOC")
    subprocess.run([sys.executable, "datatools/download_voc.py"], check=True)

    print("\nTraining")
    subprocess.run([sys.executable, "train.py", str(config_path)], check=True)

    print("\nPredicting")
    subprocess.run([sys.executable, "predict.py", str(output.path), str(dataset_dir / "val" / "images")], check=True)

    print("\nReport")
    subprocess.run([sys.executable, "report.py", str(output.path)], check=True)


if __name__ == "__main__":
    main()