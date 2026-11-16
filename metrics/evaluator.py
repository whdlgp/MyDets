from contextlib import redirect_stdout
import io

import numpy as np
import torch
from torch import distributed as dist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_convert

from models.base import Detections, Targets


class Evaluator:
    """Mean average precision of detections over a dataset, computed by pycocotools."""

    def __init__(self, num_classes: int) -> None:
        """Initialize the evaluator.

        Args:
            num_classes: number of object classes.
        """
        self.num_classes = num_classes
        self.images = []
        self.annotations = []
        self.results = []

    def update(self, detections: list[Detections], targets: list[Targets], image_size: tuple[int, int]) -> None:
        """Add the detections and the ground truth of one batch.

        Args:
            detections: detections of each image.
            targets: ground truth of each image, with the ignored boxes in extras.
            image_size: (width, height) of the input images.
        """
        width, height = image_size
        scale = torch.tensor([width, height, width, height])

        for detection, target in zip(detections, targets):
            image_id = len(self.images) + 1
            self.images.append({"id": image_id, "width": width, "height": height})

            # the ignored boxes are crowd regions
            for boxes, labels, crowd in [(target.boxes, target.labels, 0), (target.extras["ignore_boxes"], target.extras["ignore_labels"], 1)]:
                for box, label in zip(box_convert(boxes.cpu() * scale, "cxcywh", "xywh").tolist(), labels.tolist()):
                    self.annotations.append({"id": len(self.annotations) + 1, "image_id": image_id, "category_id": label, "bbox": box, "area": box[2] * box[3], "iscrowd": crowd})

            boxes = box_convert(detection.boxes.cpu() * scale, "cxcywh", "xywh")
            self.results += [[image_id, *box, score, label] for box, score, label in zip(boxes.tolist(), detection.scores.tolist(), detection.labels.tolist())]

    def compute(self) -> dict[str, float]:
        """Compute the mAP over all added batches.

        In distributed evaluation, the batches of every process are gathered first.

        Returns:
            mAP at IoU 0.5 under "map50", and averaged over IoU 0.5 to 0.95 under "map50_95".
        """
        images, annotations, results = self.images, self.annotations, self.results

        # distributed evaluation gathers every process, with the ids renumbered to stay unique
        if dist.is_available() and dist.is_initialized():
            parts = [None] * dist.get_world_size()
            dist.all_gather_object(parts, (self.images, self.annotations, self.results))
            images, annotations, results = [], [], []

            for part_images, part_annotations, part_results in parts:
                offset = len(images)
                images += [{**image, "id": image["id"] + offset} for image in part_images]
                annotations += [{**annotation, "id": len(annotations) + i + 1, "image_id": annotation["image_id"] + offset} for i, annotation in enumerate(part_annotations)]
                results += [[result[0] + offset, *result[1:]] for result in part_results]

        # pycocotools cannot load empty detections, which score zero
        if not results:
            return {"map50": 0.0, "map50_95": 0.0}

        # silence the progress prints of pycocotools
        with redirect_stdout(io.StringIO()):
            ground_truth = COCO()
            ground_truth.dataset = {"images": images, "annotations": annotations, "categories": [{"id": i} for i in range(self.num_classes)]}
            ground_truth.createIndex()
            evaluation = COCOeval(ground_truth, ground_truth.loadRes(np.array(results)), "bbox")
            evaluation.evaluate()
            evaluation.accumulate()
            evaluation.summarize()

        return {"map50": float(evaluation.stats[1]), "map50_95": float(evaluation.stats[0])}
