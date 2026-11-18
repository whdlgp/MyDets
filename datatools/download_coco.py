from pathlib import Path
from typing import NoReturn
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
import json
import shutil
import sys
import time

from torchvision.datasets.utils import extract_archive

from download_voc import download_file

CONFIG = {
    "out_dir": "data/coco",
    "archives": [
        ("annotations_trainval2017.zip", "annotations"),
        ("val2017.zip", "zips"),
        ("train2017.zip", "zips")
    ],
    "mirror_url": "https://huggingface.co/datasets/presencesw/coco/resolve/main/{name}",
    "official_url": "http://images.cocodataset.org/{folder}/{name}",
    "attempts": 5,
    "retry_seconds": 10,
    "splits": {"train": "train2017", "val": "val2017"}
}


def exit_with_usage() -> NoReturn:
    """Exit with the usage and the output directory structure."""
    lines = [
        "usage: uv run datatools\\download_coco.py [out_dir]",
        "",
        f"out_dir (default {CONFIG['out_dir']})",
        "├─ names.txt",
        "├─ train",
        "│  ├─ images",
        "│  ├─ labels",
        "│  └─ ignores    (crowd objects)",
        "└─ val",
        "   ├─ images",
        "   ├─ labels",
        "   └─ ignores    (crowd objects)"
    ]
    raise SystemExit("\n".join(lines))


def read_size(url: str) -> int | None:
    """Read the size of a file from its headers, or None when the server cannot tell."""
    try:
        with urlopen(Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
            return int(response.headers["Content-Length"])
    except (URLError, OSError, KeyError, TypeError, ValueError):
        return None


def check_archive(path: Path, size: int | None) -> bool:
    """Check that an archive has the official size and that every file in it passes its CRC check.

    Args:
        path: zip archive.
        size: size of the official file, None when unknown.

    Returns:
        Whether the archive is complete and unbroken.
    """
    if not path.exists() or (size is not None and path.stat().st_size != size):
        return False

    try:
        with ZipFile(path) as archive:
            return archive.testzip() is None
    except BadZipFile:
        return False


def download_archive(name: str, folder: str, download_dir: Path) -> Path:
    """Download an archive from the mirror, or from the official site when the mirror fails.

    Args:
        name: archive file name.
        folder: folder of the archive on the official site.
        download_dir: directory to save the archive.

    Returns:
        Archive with the official size and no broken file.
    """
    path = download_dir / name
    official_url = CONFIG["official_url"].format(folder=folder, name=name)

    # coco has no published checksum, so the size of the official file stands in for it
    size = read_size(official_url)

    if check_archive(path, size):
        return path

    for url in [CONFIG["mirror_url"].format(name=name), official_url]:
        # a mirror file of another size is not the official one
        if size is not None and url != official_url and read_size(url) != size:
            print(f"{url} differs from the official file, skipped")
            continue

        for attempt in range(1, CONFIG["attempts"] + 1):
            try:
                download_file(url, path)
            except (URLError, OSError) as error:
                print(f"{url} failed ({error}), attempt {attempt}/{CONFIG['attempts']}")

                # the requested range is past a complete but broken file
                if isinstance(error, HTTPError) and error.code == 416:
                    path.unlink()

                # a refused or missing file does not change on retry, so the next url is tried
                elif isinstance(error, HTTPError) and 400 <= error.code < 500 and error.code != 429:
                    break

                time.sleep(CONFIG["retry_seconds"])
                continue

            if check_archive(path, size):
                return path

            # a broken file comes again from the same url, so the next url is tried
            print(f"{name} is broken, trying the next url")
            path.unlink()
            break

    raise SystemExit(f"cannot download {name}")


def convert_split(annotation_path: Path, image_dir: Path, split_dir: Path, category_ids: list[int]) -> tuple[int, int, int]:
    """Convert a COCO split to images, label lines and crowd lines.

    Args:
        annotation_path: COCO instances json of the split.
        image_dir: extracted images of the split.
        split_dir: output split directory.
        category_ids: COCO category ids in the order of the class indices.

    Returns:
        Number of images, of objects and of crowd objects.
    """
    data = json.loads(annotation_path.read_text())
    images = {image["id"]: image for image in data["images"]}
    lines = {image_id: {False: [], True: []} for image_id in images}

    for annotation in data["annotations"]:
        image = images[annotation["image_id"]]
        width, height = image["width"], image["height"]
        left, top, box_width, box_height = annotation["bbox"]

        # a few boxes have no area
        if box_width <= 0 or box_height <= 0:
            continue

        # coco boxes are left, top, width, height in continuous pixel coordinates
        values = [(left + box_width / 2) / width, (top + box_height / 2) / height, box_width / width, box_height / height]
        line = f"{category_ids.index(annotation['category_id'])} " + " ".join(f"{v:.6f}" for v in values)
        lines[image["id"]][annotation["iscrowd"] == 1].append(line)

    for folder in ["images", "labels"]:
        (split_dir / folder).mkdir(parents=True, exist_ok=True)

    # every image is kept, also the ones without objects
    for image_id, image in images.items():
        stem = Path(image["file_name"]).stem
        labels, crowds = lines[image_id][False], lines[image_id][True]
        (split_dir / "labels" / f"{stem}.txt").write_text("".join(f"{line}\n" for line in labels))

        if crowds:
            (split_dir / "ignores").mkdir(exist_ok=True)
            (split_dir / "ignores" / f"{stem}.txt").write_text("".join(f"{line}\n" for line in crowds))

        (image_dir / image["file_name"]).replace(split_dir / "images" / image["file_name"])

    num_objects = sum(len(v[False]) for v in lines.values())
    num_crowds = sum(len(v[True]) for v in lines.values())
    return len(images), num_objects, num_crowds


def main():
    """Download COCO 2017 and save it as a dataset directory.

    Train is train2017 and val is val2017, with the crowd objects in ignores for both.
    The archives come from a mirror on Hugging Face, or from the official site when the mirror fails, and are kept in out_dir/downloads.
    Interrupted downloads continue from where they stopped, and a finished dataset is not prepared again.

    Usage: uv run datatools\\download_coco.py [out_dir]
    """
    # read the arguments
    if len(sys.argv) > 2:
        exit_with_usage()

    out_dir = Path(sys.argv[1] if len(sys.argv) == 2 else CONFIG["out_dir"])
    download_dir = out_dir / "downloads"

    # names.txt is written last, so it marks a finished conversion
    if (out_dir / "names.txt").exists():
        print(f"Already prepared: {out_dir}")
        return

    # download and extract
    download_dir.mkdir(parents=True, exist_ok=True)

    for name, folder in CONFIG["archives"]:
        extract_archive(str(download_archive(name, folder, download_dir)), str(download_dir))

    # the class indices follow the category ids, which skip some numbers
    categories = json.loads((download_dir / "annotations" / "instances_val2017.json").read_text())["categories"]
    categories = sorted(categories, key=lambda c: c["id"])
    category_ids = [c["id"] for c in categories]

    # convert each split
    for split, coco_split in CONFIG["splits"].items():
        annotation_path = download_dir / "annotations" / f"instances_{coco_split}.json"
        num_images, num_objects, num_crowds = convert_split(annotation_path, download_dir / coco_split, out_dir / split, category_ids)
        print(f"{split}: {coco_split}, {num_images} images, {num_objects} objects, {num_crowds} crowd objects")

    for folder in ["annotations", *CONFIG["splits"].values()]:
        shutil.rmtree(download_dir / folder)

    (out_dir / "names.txt").write_text("".join(f"{c['name']}\n" for c in categories))
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
