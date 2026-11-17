from pathlib import Path
from typing import NoReturn
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import shutil
import sys
import time
import xml.etree.ElementTree as ET

from torchvision.datasets.utils import check_integrity, extract_archive
from tqdm import tqdm

CONFIG = {
    "out_dir": "data/voc",
    "archives": [
        ("VOCtrainval_06-Nov-2007.tar", "c52e279531787c972589f7e41ab4ae64", "voc2007"),
        ("VOCtest_06-Nov-2007.tar", "b6e924de25625d8de591ea690078ad9f", "voc2007"),
        ("VOCtrainval_11-May-2012.tar", "6cd6e144f989b92b3379bac3b3de84fd", "voc2012")
    ],
    "mirror_url": "https://data.pjreddie.com/files/{name}",
    "official_url": "https://thor.robots.ox.ac.uk/pascal/VOC/{year}/{name}",
    "attempts": 5,
    "retry_seconds": 10,
    "splits": {
        "train": [("VOC2007", "trainval"), ("VOC2012", "trainval")],
        "val": [("VOC2007", "test")]
    },
    "classes": ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
}


def exit_with_usage() -> NoReturn:
    """Exit with the usage and the output directory structure."""
    lines = [
        "usage: uv run datatools\\download_voc.py [out_dir]",
        "",
        f"out_dir (default {CONFIG['out_dir']})",
        "├─ names.txt",
        "├─ train",
        "│  ├─ images",
        "│  └─ labels",
        "└─ val",
        "   ├─ images",
        "   ├─ labels",
        "   └─ ignores    (difficult objects)"
    ]
    raise SystemExit("\n".join(lines))


def download_file(url: str, path: Path) -> None:
    """Download a file, continuing after the bytes already in path.

    Args:
        url: file url.
        path: file to write.
    """
    done = path.stat().st_size if path.exists() else 0

    # some servers refuse the default user agent of urllib
    headers = {"Range": f"bytes={done}-", "User-Agent": "Mozilla/5.0"}

    with urlopen(Request(url, headers=headers), timeout=60) as response:
        # a server without range requests sends the whole file
        if response.status != 206:
            done = 0

        total = done + int(response.headers.get("Content-Length", 0))

        with open(path, "ab" if done else "wb") as f, tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=path.name) as progress:
            while chunk := response.read(1 << 20):
                f.write(chunk)
                progress.update(len(chunk))

    # a closed connection ends the read early, and the next attempt continues
    if path.stat().st_size < total:
        raise ConnectionError(f"connection closed at {path.stat().st_size} of {total} bytes")


def download_archive(name: str, md5: str, year: str, download_dir: Path) -> Path:
    """Download an archive from the mirror, or from the official site when the mirror fails.

    Args:
        name: archive file name.
        md5: checksum of the archive.
        year: folder of the archive on the official site.
        download_dir: directory to save the archive.

    Returns:
        Archive with the right checksum.
    """
    path = download_dir / name

    if check_integrity(str(path), md5):
        return path

    for url in [CONFIG["mirror_url"].format(name=name), CONFIG["official_url"].format(year=year, name=name)]:
        for attempt in range(1, CONFIG["attempts"] + 1):
            try:
                download_file(url, path)
            except (URLError, OSError) as error:
                print(f"{url} failed ({error}), attempt {attempt}/{CONFIG['attempts']}")

                # the requested range is past a complete but corrupted file
                if isinstance(error, HTTPError) and error.code == 416:
                    path.unlink()

                # a refused or missing file does not change on retry, so the next url is tried
                elif isinstance(error, HTTPError) and 400 <= error.code < 500 and error.code != 429:
                    break

                time.sleep(CONFIG["retry_seconds"])
                continue

            if check_integrity(str(path), md5):
                return path

            # a corrupted file starts over
            print(f"{name} has a wrong checksum, attempt {attempt}/{CONFIG['attempts']}")
            path.unlink()

    raise SystemExit(f"cannot download {name}")


def convert_annotation(xml_path: Path) -> tuple[list[str], list[str]]:
    """Convert a VOC annotation to label lines.

    Args:
        xml_path: VOC annotation file.

    Returns:
        Lines of the normal objects and of the difficult objects, as "class cx cy w h" normalized.
    """
    root = ET.parse(xml_path).getroot()
    width, height = float(root.find("size/width").text), float(root.find("size/height").text)
    lines = {False: [], True: []}

    for obj in root.iter("object"):
        box = obj.find("bndbox")
        difficult = obj.find("difficult")

        # voc boxes are 1-based inclusive pixel indices
        left, top = float(box.find("xmin").text) - 1, float(box.find("ymin").text) - 1
        right, bottom = float(box.find("xmax").text), float(box.find("ymax").text)
        values = [(left + right) / 2 / width, (top + bottom) / 2 / height, (right - left) / width, (bottom - top) / height]
        line = f"{CONFIG['classes'].index(obj.find('name').text)} " + " ".join(f"{v:.6f}" for v in values)
        lines[difficult is not None and int(difficult.text) == 1].append(line)

    return lines[False], lines[True]


def main():
    """Download PASCAL VOC and save it as a dataset directory.

    Train is VOC2007 and VOC2012 trainval with the difficult objects, and val is VOC2007 test with the difficult objects in ignores.
    The archives come from the mirror of pjreddie, or from the official site when the mirror fails, and are kept in out_dir/downloads.
    Interrupted downloads continue from where they stopped, and a finished dataset is not prepared again.

    Usage: uv run datatools\\download_voc.py [out_dir]
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

    for name, md5, year in CONFIG["archives"]:
        extract_archive(str(download_archive(name, md5, year, download_dir)), str(download_dir))

    # convert each split
    for split, sets in CONFIG["splits"].items():
        for folder in ["images", "labels"]:
            (out_dir / split / folder).mkdir(parents=True, exist_ok=True)

        for year, image_set in sets:
            voc_dir = download_dir / "VOCdevkit" / year
            image_ids = (voc_dir / "ImageSets" / "Main" / f"{image_set}.txt").read_text().split()

            for image_id in image_ids:
                labels, difficults = convert_annotation(voc_dir / "Annotations" / f"{image_id}.xml")

                # train learns the difficult objects, and val ignores them
                if split == "train":
                    labels, difficults = labels + difficults, []

                (out_dir / split / "labels" / f"{image_id}.txt").write_text("".join(f"{line}\n" for line in labels))

                if difficults:
                    (out_dir / split / "ignores").mkdir(exist_ok=True)
                    (out_dir / split / "ignores" / f"{image_id}.txt").write_text("".join(f"{line}\n" for line in difficults))

                (voc_dir / "JPEGImages" / f"{image_id}.jpg").replace(out_dir / split / "images" / f"{image_id}.jpg")

            print(f"{split}: {year} {image_set}, {len(image_ids)} images")

    shutil.rmtree(download_dir / "VOCdevkit")
    (out_dir / "names.txt").write_text("\n".join(CONFIG["classes"]) + "\n")
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
