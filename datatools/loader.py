from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}


class DetectionDataset(Dataset):
    """Images, labels and ignored boxes of one split of a dataset directory."""

    def __init__(self, split_dir: Path) -> None:
        """Initialize the dataset.

        Args:
            split_dir: split directory with images, labels and optional ignores, next to names.txt.
        """
        self.split_dir = Path(split_dir)
        self.names = (self.split_dir.parent / "names.txt").read_text().split()
        self.image_paths = sorted(p for p in (self.split_dir / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read an image as BGR, its labels and its ignored boxes."""
        path = self.image_paths[index]

        # imdecode also reads non-ascii paths on windows
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"cannot read image {path}")

        return image, self.read_boxes("labels", path.stem), self.read_boxes("ignores", path.stem)

    def read_boxes(self, folder: str, stem: str) -> np.ndarray:
        """Read the boxes of a label file, empty when the file does not exist.

        Args:
            folder: labels or ignores.
            stem: file name of the image without the suffix.

        Returns:
            Boxes of shape (N, 5) as (class, cx, cy, w, h) normalized.
        """
        path = self.split_dir / folder / f"{stem}.txt"

        if not path.exists():
            return np.zeros((0, 5), np.float32)

        return np.array(path.read_text().split(), np.float32).reshape(-1, 5)


def init_worker(worker_id: int) -> None:
    """Stop the opencv threads of a data loader worker."""
    cv2.setNumThreads(0)


def create_loader(split_dir: Path, transform: Callable, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Create a data loader that runs the transform on each batch in the workers.

    In distributed training, the images are split over the processes.

    Args:
        split_dir: split directory with images, labels and optional ignores, next to names.txt.
        transform: transform from a list of dataset samples to the model input and the targets.
        batch_size: number of images per batch.
        shuffle: whether to shuffle the images every epoch.
        num_workers: number of worker processes.

    Returns:
        Data loader of (images, targets).
    """
    dataset = DetectionDataset(split_dir)
    sampler = None

    # evaluation needs every image exactly once, which the padding of DistributedSampler breaks
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset) if shuffle else range(dist.get_rank(), len(dataset), dist.get_world_size())
        shuffle = False

    return DataLoader(dataset, batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, collate_fn=transform, worker_init_fn=init_worker)
