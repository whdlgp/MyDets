from dataclasses import dataclass
from pathlib import Path
import random

import cv2

from datatools.loader import IMAGE_SUFFIXES
from webui.bridge.processes import find_process, start_process, stop_process

DATA_DIR = Path("data")
SCRIPT_DIR = Path("datatools")
SPLITS = ["train", "val"]
SAMPLE_SIDE = 640


@dataclass
class SplitInfo:
    """Files of one split of a dataset directory."""
    name: str
    images: int
    labels: int
    ignores: int
    unlabeled: int


@dataclass
class DatasetInfo:
    """A dataset directory, its download script and what it holds."""
    name: str
    path: Path
    classes: list[str] | None
    splits: list[SplitInfo]
    script: Path | None
    state: str


def list_scripts() -> dict[str, Path]:
    """Return the download scripts in datatools/ by dataset name, such as voc for download_voc.py."""
    return {path.stem.removeprefix("download_"): path for path in sorted(SCRIPT_DIR.glob("download_*.py"))}


def list_stems(folder: Path, suffixes: set[str]) -> set[str]:
    """Return the file names without suffix of a folder, only the given suffixes, empty when the folder is missing."""
    if not folder.is_dir():
        return set()

    return {path.stem for path in folder.iterdir() if path.suffix.lower() in suffixes}


def read_dataset(name: str) -> DatasetInfo:
    """Read the classes and the file counts of a dataset directory in data/.

    Args:
        name: folder name in data/.

    Returns:
        Info, with the state downloading, ready, incomplete or missing.
    """
    path = DATA_DIR / name
    names_path = path / "names.txt"
    classes = names_path.read_text(encoding="utf-8").splitlines() if names_path.exists() else None
    splits = []

    for split in SPLITS:
        images = list_stems(path / split / "images", IMAGE_SUFFIXES)
        labels = list_stems(path / split / "labels", {".txt"})
        ignores = list_stems(path / split / "ignores", {".txt"})
        splits.append(SplitInfo(split, len(images), len(labels), len(ignores), len(images - labels)))

    if find_process(path / "download.pid"):
        state = "downloading"
    elif classes and all(s.images and s.unlabeled == 0 for s in splits):
        state = "ready"
    elif path.is_dir():
        state = "incomplete"
    else:
        state = "missing"

    return DatasetInfo(name, path, classes, splits, list_scripts().get(name), state)


def list_datasets() -> list[DatasetInfo]:
    """Read every folder in data/ and every dataset that a download script can make."""
    folders = {path.name for path in DATA_DIR.iterdir() if path.is_dir()} if DATA_DIR.is_dir() else set()
    return [read_dataset(name) for name in sorted(folders | set(list_scripts()))]


def list_downloads() -> set[str]:
    """Return the names of the datasets whose download is running, without counting their files."""
    folders = sorted(path for path in DATA_DIR.iterdir() if path.is_dir()) if DATA_DIR.is_dir() else []
    return {path.name for path in folders if find_process(path / "download.pid")}


def start_download(name: str) -> None:
    """Start the download script of a dataset detached from the server, writing into data/<name>/."""
    path = DATA_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    start_process([str(list_scripts()[name]), str(path)], path / "download.log", path / "download.pid", {})


def stop_download(name: str) -> None:
    """Stop the download of a dataset, which continues from where it stopped when started again."""
    stop_process(DATA_DIR / name / "download.pid")


def read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read a darknet label file, one (class id, cx, cy, w, h) per line, empty when the file is missing."""
    if not path.exists():
        return []

    boxes = []

    for i, line in enumerate(path.read_text().splitlines()):
        values = line.split()

        if not values:
            continue

        if len(values) != 5:
            raise ValueError(f"{path} line {i + 1}: expected 5 values, got {len(values)}")

        boxes.append((int(values[0]), *map(float, values[1:])))

    return boxes


def draw_sample(info: DatasetInfo, split: str, seed: int) -> tuple[bytes, str] | None:
    """Draw the labels of one random image of a split, with the ignored boxes thin and gray.

    Args:
        info: dataset to draw from.
        split: train or val.
        seed: seed of the random image.

    Returns:
        JPEG bytes of the drawn image and the image file name, None when the split has no images.
    """
    split_dir = info.path / split
    image_paths = sorted(p for p in (split_dir / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if (split_dir / "images").is_dir() else []

    if not image_paths:
        return None

    image_path = random.Random(seed).choice(image_paths)
    image = cv2.imread(str(image_path))
    scale = SAMPLE_SIDE / max(image.shape[:2])
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    classes = info.classes or []
    colors = random.Random(0)
    palette = [tuple(colors.randint(40, 230) for _ in range(3)) for _ in range(max(len(classes), 1))]

    for folder in ["labels", "ignores"]:
        for class_id, cx, cy, w, h in read_boxes(split_dir / folder / f"{image_path.stem}.txt"):
            x1, y1 = round((cx - w / 2) * width), round((cy - h / 2) * height)
            x2, y2 = round((cx + w / 2) * width), round((cy + h / 2) * height)
            ignored = folder == "ignores"
            color = (150, 150, 150) if ignored else palette[class_id % len(palette)]
            name = classes[class_id] if 0 <= class_id < len(classes) else f"? {class_id}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1 if ignored else 2)
            (text_width, text_height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, max(y1 - text_height - 6, 0)), (x1 + text_width + 6, max(y1, text_height + 6)), color, -1)
            cv2.putText(image, name, (x1 + 3, max(y1 - 4, text_height + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(), image_path.name
