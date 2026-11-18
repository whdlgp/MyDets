import subprocess
import sys


def main():
    """Download COCO 2017 and train a model on it.

    Usage: uv run train_coco.py <config>
    """
    print("Preparing COCO 2017")
    subprocess.run([sys.executable, "datatools/download_coco.py"], check=True)

    print("\nTraining")
    subprocess.run([sys.executable, "train.py", *sys.argv[1:]], check=True)


if __name__ == "__main__":
    main()
