from dataclasses import MISSING, dataclass, field, fields


@dataclass
class CommonSettings:
    """Settings of the common section, read by training, prediction and the report.

    Attributes:
        image_size: (width, height) of the model input, multiples of the model strides.
        mean: mean of each RGB channel, for pixel values in [0, 1].
        std: standard deviation of each RGB channel, for pixel values in [0, 1].
    """

    image_size: list[int] = field(default_factory=lambda: [416, 416])
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class TrainSettings:
    """Settings of the train section.

    Attributes:
        dataset_dir: darknet-format dataset with names.txt and the train and val splits.
        transforms: train transforms in order, each as {"type": class name, "args": constructor arguments}.
        val_transforms: val transforms in order, in the same form as transforms.
        batch_size: number of images per batch.
        num_workers: number of data loader processes.
        epochs: number of epochs.
        warmup_epochs: epochs of linear learning rate warmup.
        freeze_epochs: first epochs with the backbone frozen.
        close_mosaic_epochs: mosaic off for the last N + 1 epochs, like the original YOLOX trainer.
        lr: peak learning rate of AdamW.
        weight_decay: weight decay of AdamW, not applied to norms and biases.
        ema: evaluate and save the EMA weights.
        amp: mixed precision, bfloat16 if supported and float16 otherwise.
        seed: random seed.
    """

    dataset_dir: str
    transforms: list[dict]
    val_transforms: list[dict]
    batch_size: int = 16
    num_workers: int = 8
    epochs: int = 100
    warmup_epochs: int = 3
    freeze_epochs: int = 0
    close_mosaic_epochs: int = 15
    lr: float = 0.001
    weight_decay: float = 0.05
    ema: bool = True
    amp: bool = True
    seed: int = 0


@dataclass
class PredictSettings:
    """Settings of the predict section.

    Attributes:
        transforms: prediction transforms in order, in the same form as the train transforms.
        score_threshold: minimum score of the drawn boxes.
    """

    transforms: list[dict]
    score_threshold: float = 0.3


def read_settings[T](cls: type[T], config: dict, section: str) -> T:
    """Read a config section into its settings class, failing on unknown and missing keys.

    Args:
        cls: settings class of the section.
        config: whole config.
        section: name of the section.

    Returns:
        Settings of the section.
    """
    values = config.get(section) or {}
    known = [f.name for f in fields(cls)]
    unknown = [key for key in values if key not in known]
    missing = [f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING and f.name not in values]

    if unknown:
        raise SystemExit(f"unknown keys in the {section} section: {unknown}, expected some of {known}")

    if missing:
        raise SystemExit(f"missing keys in the {section} section: {missing}")

    return cls(**values)
