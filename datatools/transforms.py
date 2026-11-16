from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
import math
import random

import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision.ops import box_convert

from models.base import Targets


@dataclass
class Batch:
    """BGR images, pixel boxes as (x1, y1, x2, y2, class, ignore) and the output (width, height)."""
    images: list[np.ndarray]
    boxes: list[np.ndarray]
    size: tuple[int, int]


def clip_boxes(boxes: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    """Clip the boxes to a region and drop those under one pixel.

    Args:
        boxes: boxes of shape (N, 6) as (x1, y1, x2, y2, class, ignore) in pixels.
        region: (left, top, right, bottom) in pixels.

    Returns:
        Clipped boxes that remain.
    """
    left, top, right, bottom = region
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(left, right)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(top, bottom)
    keep = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
    return boxes[keep]


class ToBatch:
    """Convert dataset samples to a batch."""

    def __init__(self, image_size: tuple[int, int]) -> None:
        """Initialize the transform.

        Args:
            image_size: (width, height) of the output images, multiples of the model strides.
        """
        self.image_size = image_size

    def __call__(self, samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Batch:
        """Convert (BGR image, labels, ignored boxes) samples."""
        images, boxes = [], []

        for image, labels, ignores in samples:
            height, width = image.shape[:2]
            rows = np.concatenate([np.pad(labels, ((0, 0), (0, 1))), np.pad(ignores, ((0, 0), (0, 1)), constant_values=1)])
            cx, cy, w, h = rows[:, 1] * width, rows[:, 2] * height, rows[:, 3] * width, rows[:, 4] * height
            images.append(image)
            boxes.append(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, rows[:, 0], rows[:, 5]], axis=1).astype(np.float32))

        return Batch(images, boxes, self.image_size)


class RandomScale:
    """Scale the output size of each batch randomly."""

    def __init__(self, max_scale: float = 1.4, step: int = 32) -> None:
        """Initialize the transform.

        Args:
            max_scale: maximum scale, also used inverted as the minimum.
            step: output sides are multiples of this.
        """
        self.max_scale = max_scale
        self.step = step

    def __call__(self, batch: Batch) -> Batch:
        """Pick the output size of the batch."""
        scale = random.uniform(1, self.max_scale)

        if random.random() < 0.5:
            scale = 1 / scale

        # round half up, at least one step
        batch.size = tuple(max(math.floor(scale * side / self.step + 0.5) * self.step, self.step) for side in batch.size)
        return batch


class RandomCrop:
    """Crop each image randomly and resize it to the output size."""

    def __init__(self, jitter: float = 0.3) -> None:
        """Initialize the transform.

        Args:
            jitter: maximum crop or padding of each side, as a ratio of the image side.
        """
        self.jitter = jitter

    def __call__(self, batch: Batch) -> Batch:
        """Crop and resize every image of the batch."""
        width, height = batch.size

        for i, (image, boxes) in enumerate(zip(batch.images, batch.boxes)):
            image_height, image_width = image.shape[:2]
            max_x, max_y = int(image_width * self.jitter), int(image_height * self.jitter)
            left, right = random.randint(-max_x, max_x), random.randint(-max_x, max_x)
            top, bottom = random.randint(-max_y, max_y), random.randint(-max_y, max_y)
            crop_width, crop_height = image_width - left - right, image_height - top - bottom

            # an empty crop keeps the whole image
            if crop_width <= 0 or crop_height <= 0 or left >= image_width or top >= image_height:
                left, right, top, bottom = 0, 0, 0, 0
                crop_width, crop_height = image_width, image_height

            # negative sides pad with the mean color
            mean = [round(value) for value in cv2.mean(image)[:3]]
            padded = cv2.copyMakeBorder(image, max(-top, 0), max(-bottom, 0), max(-left, 0), max(-right, 0), cv2.BORDER_CONSTANT, value=mean)
            cropped = padded[max(top, 0):max(top, 0) + crop_height, max(left, 0):max(left, 0) + crop_width]
            batch.images[i] = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

            moved = boxes.copy()
            moved[:, :4] = (boxes[:, :4] - [left, top, left, top]) * ([width / crop_width, height / crop_height] * 2)
            batch.boxes[i] = clip_boxes(moved, (0, 0, width, height))

        return batch


class RandomFlip:
    """Flip each image horizontally with a probability."""

    def __init__(self, probability: float = 0.5) -> None:
        """Initialize the transform.

        Args:
            probability: probability to flip each image.
        """
        self.probability = probability

    def __call__(self, batch: Batch) -> Batch:
        """Flip the images of the batch randomly."""
        for i, (image, boxes) in enumerate(zip(batch.images, batch.boxes)):
            if random.random() >= self.probability:
                continue

            width = image.shape[1]
            flipped = boxes.copy()
            flipped[:, 0], flipped[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
            batch.images[i] = cv2.flip(image, 1)
            batch.boxes[i] = flipped

        return batch


class RandomHSV:
    """Shift the hue and scale the saturation and the value of each image randomly."""

    def __init__(self, hue: float = 0.1, saturation: float = 1.5, exposure: float = 1.5) -> None:
        """Initialize the transform.

        Args:
            hue: maximum hue shift, as a ratio of the hue circle.
            saturation: maximum saturation scale, also used inverted as the minimum.
            exposure: maximum value scale, also used inverted as the minimum.
        """
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

    def __call__(self, batch: Batch) -> Batch:
        """Change the colors of every image of the batch."""
        values = np.arange(256)

        for i, image in enumerate(batch.images):
            hue_shift = random.uniform(-self.hue, self.hue)
            scales = [random.uniform(1, maximum) for maximum in (self.saturation, self.exposure)]
            saturation, exposure = [scale if random.random() < 0.5 else 1 / scale for scale in scales]

            # the 8-bit hue is a circle of 180
            lut = np.stack([np.round(values + 179 * hue_shift) % 180, np.round(values * saturation).clip(0, 255), np.round(values * exposure).clip(0, 255)], axis=1)
            hsv = cv2.LUT(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), lut.astype(np.uint8).reshape(256, 1, 3))
            batch.images[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return batch


class Mosaic:
    """Replace the batch by mosaics of 4 images, with a probability."""

    def __init__(self, probability: float = 0.5) -> None:
        """Initialize the transform.

        Args:
            probability: probability to apply the mosaic to each batch.
        """
        self.probability = probability

    def __call__(self, batch: Batch) -> Batch:
        """Make a mosaic for each image of the batch."""
        if random.random() >= self.probability:
            return batch

        width, height = batch.size
        count = len(batch.images)
        images, boxes = [], []

        for _ in range(count):
            sources = random.sample(range(count), 4) if count >= 4 else random.choices(range(count), k=4)
            cut_x = random.randint(int(width * 0.2), int(width * 0.8))
            cut_y = random.randint(int(height * 0.2), int(height * 0.8))
            mosaic = np.empty((height, width, 3), np.uint8)
            pieces = []

            # each quarter takes the opposite corner of its source
            quarters = [(0, 0, cut_x, cut_y), (cut_x, 0, width, cut_y), (0, cut_y, cut_x, height), (cut_x, cut_y, width, height)]

            for (left, top, right, bottom), index in zip(quarters, sources):
                shift_x = cut_x - width if left == 0 else cut_x
                shift_y = cut_y - height if top == 0 else cut_y
                mosaic[top:bottom, left:right] = batch.images[index][top - shift_y:bottom - shift_y, left - shift_x:right - shift_x]
                moved = batch.boxes[index].copy()
                moved[:, :4] += [shift_x, shift_y, shift_x, shift_y]
                pieces.append(clip_boxes(moved, (left, top, right, bottom)))

            images.append(mosaic)
            boxes.append(np.concatenate(pieces))

        return Batch(images, boxes, batch.size)


class Letterbox:
    """Resize each image keeping the aspect ratio and pad it to the output size."""

    def __call__(self, batch: Batch) -> Batch:
        """Letterbox every image of the batch."""
        width, height = batch.size

        for i, (image, boxes) in enumerate(zip(batch.images, batch.boxes)):
            image_height, image_width = image.shape[:2]

            if width * image_height < height * image_width:
                new_width, new_height = width, image_height * width // image_width
            else:
                new_width, new_height = image_width * height // image_height, height

            # the padding is mid gray
            left, top = (width - new_width) // 2, (height - new_height) // 2
            boxed = np.full((height, width, 3), 128, np.uint8)
            boxed[top:top + new_height, left:left + new_width] = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            batch.images[i] = boxed

            moved = boxes.copy()
            moved[:, :4] = boxes[:, :4] * ([new_width / image_width, new_height / image_height] * 2) + [left, top, left, top]
            batch.boxes[i] = moved

        return batch


class Resize:
    """Resize each image keeping the aspect ratio, with sides in multiples of step."""

    def __init__(self, step: int = 32) -> None:
        """Initialize the transform.

        Args:
            step: output sides are multiples of this.
        """
        self.step = step

    def __call__(self, batch: Batch) -> Batch:
        """Resize every image of the batch."""
        width, height = batch.size

        for i, (image, boxes) in enumerate(zip(batch.images, batch.boxes)):
            image_height, image_width = image.shape[:2]
            scale = min(width / image_width, height / image_height)

            # round half up, at least one step
            new_width, new_height = [max(math.floor(scale * side / self.step + 0.5) * self.step, self.step) for side in (image_width, image_height)]
            batch.images[i] = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            moved = boxes.copy()
            moved[:, :4] = boxes[:, :4] * ([new_width / image_width, new_height / image_height] * 2)
            batch.boxes[i] = moved

        return batch


class Normalize:
    """Stack the batch into a normalized RGB tensor and convert the boxes to targets."""

    def __init__(self, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        """Initialize the transform.

        Args:
            mean: mean of each RGB channel, for pixel values in [0, 1].
            std: standard deviation of each RGB channel, for pixel values in [0, 1].
        """
        self.mean = torch.tensor(mean).reshape(1, 3, 1, 1)
        self.std = torch.tensor(std).reshape(1, 3, 1, 1)

    def __call__(self, batch: Batch) -> tuple[Tensor, list[Targets]]:
        """Convert the batch to the model input.

        Args:
            batch: batch whose images all have the same size.

        Returns:
            Images of shape (B, 3, H, W), and the targets of each image with the ignored boxes in extras.
        """
        # bgr to rgb
        images = torch.from_numpy(np.stack(batch.images)).permute(0, 3, 1, 2).flip(1)
        images = (images.float() / 255 - self.mean) / self.std
        height, width = images.shape[-2:]
        scale = torch.tensor([width, height, width, height])
        targets = []

        for boxes in batch.boxes:
            boxes = torch.from_numpy(boxes)
            normalized = box_convert(boxes[:, :4], "xyxy", "cxcywh") / scale
            labels = boxes[:, 4].long()
            ignore = boxes[:, 5] > 0
            targets.append(Targets(normalized[~ignore], labels[~ignore], {"ignore_boxes": normalized[ignore], "ignore_labels": labels[ignore]}))

        return images, targets


class Compose:
    """Run the transforms in order, each on the output of the previous one."""

    def __init__(self, transforms: list[Callable]) -> None:
        """Initialize the chain.

        Args:
            transforms: transforms from the dataset samples to the model input.
        """
        self.transforms = transforms

    def __call__(self, samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[Tensor, list[Targets]]:
        """Convert dataset samples to the model input and the targets."""
        output = samples

        for transform in self.transforms:
            output = transform(output)

        return output


def create_transform(mode: Literal["train", "val", "predict"], image_size: tuple[int, int], mean: tuple[float, float, float], std: tuple[float, float, float]) -> Compose:
    """Create the transform of a mode, used as the collate function.

    Args:
        mode: train with augmentation, val with letterbox, or predict with resize for batches of one image.
        image_size: (width, height) of the output images, multiples of the model strides.
        mean: mean of each RGB channel, for pixel values in [0, 1].
        std: standard deviation of each RGB channel, for pixel values in [0, 1].

    Returns:
        Transform from dataset samples to the model input and the targets.
    """
    blocks = {
        "train": [RandomScale(), RandomCrop(), RandomFlip(), RandomHSV(), Mosaic()],
        "val": [Letterbox()],
        "predict": [Resize()]
    }
    return Compose([ToBatch(image_size), *blocks[mode], Normalize(mean, std)])
