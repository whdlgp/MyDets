from pathlib import Path
import subprocess
import sys

from shared.output_directory import OutputDirectory
from shared.settings import TrainSettings, read_config, read_settings


def main():
    """Download COCO 2017, train a model on it, draw its detections on random val images and report the training.

    Usage: uv run train_coco.py <config>
    Saved in runs/<config name>/.
    """
    # read the arguments
    if len(sys.argv) != 2:
        raise SystemExit("usage: uv run train_coco.py <config>")

    config_path = Path(sys.argv[1])
    config = read_config(config_path)
    dataset_dir = Path(read_settings(TrainSettings, config, "train").dataset_dir)
    output = OutputDirectory.from_config(config_path)

    print("Preparing COCO 2017")
    subprocess.run([sys.executable, "datatools/download_coco.py"], check=True)

    print("\nTraining")
    subprocess.run([sys.executable, "train.py", str(config_path)], check=True)

    print("\nPredicting")
    subprocess.run([sys.executable, "predict.py", str(output.path), str(dataset_dir / "val" / "images")], check=True)

    print("\nReport")
    subprocess.run([sys.executable, "report.py", str(output.path)], check=True)


if __name__ == "__main__":
    main()
