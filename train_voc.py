import subprocess
import sys


def main():
    """Download PASCAL VOC and train a model on it.

    Usage: uv run train_voc.py <config>
    """
    subprocess.run([sys.executable, "datatools/download_voc.py"], check=True)
    subprocess.run([sys.executable, "train.py", *sys.argv[1:]], check=True)


if __name__ == "__main__":
    main()