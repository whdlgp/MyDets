from pathlib import Path
from typing import NoReturn
import shutil
import sys
import xml.etree.ElementTree as ET

from torchvision.datasets.utils import download_and_extract_archive

CONFIG = {
    "out_dir": "data/voc",
    "archives": [
        ("https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar", "c52e279531787c972589f7e41ab4ae64"),
        ("https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar", "b6e924de25625d8de591ea690078ad9f"),
        ("https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar", "6cd6e144f989b92b3379bac3b3de84fd")
    ],
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
    The archives are kept in out_dir/downloads.

    Usage: uv run datatools\\download_voc.py [out_dir]
    """
    # read the arguments
    if len(sys.argv) > 2:
        exit_with_usage()

    out_dir = Path(sys.argv[1] if len(sys.argv) == 2 else CONFIG["out_dir"])
    download_dir = out_dir / "downloads"

    # download and extract
    for url, md5 in CONFIG["archives"]:
        download_and_extract_archive(url, str(download_dir), md5=md5)

    (out_dir / "names.txt").write_text("\n".join(CONFIG["classes"]) + "\n")

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
    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()
