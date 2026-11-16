from torch import nn

from models.backbone.timm_backbone import TimmBackbone

from models.neck.bifpn import BiFPN
from models.neck.cspnext_pafpn import CSPNeXtPAFPN
from models.neck.hybrid_encoder import HybridEncoder
from models.neck.pan import PAN

from models.head.retinanet_head import RetinaNetHead
from models.head.rtmdet_head import RTMDetSepBNHead
from models.head.tood_head import TOODHead
from models.head.yolox_head import YOLOXHead

from models.base import Detector

BACKBONES = [
    TimmBackbone
]

NECKS = [
    BiFPN,
    CSPNeXtPAFPN,
    HybridEncoder,
    PAN
]

HEADS = [
    RetinaNetHead,
    RTMDetSepBNHead,
    TOODHead,
    YOLOXHead
]

ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU
}


def build_module(registry: list[type], section: str, config: dict, *connecting_args: list[int] | int) -> nn.Module:
    """Build a backbone, neck or head from its config section.

    Args:
        registry: registered classes of the section.
        section: name of the config section, for the error messages.
        config: config section with the type and optional args.
        connecting_args: connecting args from the previous module, in the constructor order.

    Returns:
        Module of the type, built with the connecting args and the converted args.
    """
    # find the class of the type
    name = config["type"]
    classes = {cls.__name__: cls for cls in registry}

    if name not in classes:
        raise ValueError(f"unknown {section} type {name}, expected one of {list(classes)}")

    # convert the args to the constructor arguments
    args = {}

    for key, value in (config.get("args") or {}).items():
        # yaml has no tuple, and the sequence options of the implementations are tuples
        if isinstance(value, list):
            value = tuple(value)

        if key.endswith("activation"):
            if value not in ACTIVATIONS:
                raise ValueError(f"unknown {section} {key} {value}, expected one of {list(ACTIVATIONS)}")

            value = ACTIVATIONS[value]

        args[key] = value

    return classes[name](*connecting_args, **args)


def build_model(config: dict, num_classes: int) -> Detector:
    """Build a detector from the model section of a config.

    Args:
        config: model section with the backbone, neck and head sections, each with a type and optional args.
        num_classes: number of object classes, from the dataset.

    Returns:
        Detector with the backbone, neck and head connected in order.
    """
    backbone = build_module(BACKBONES, "backbone", config["backbone"])
    neck = build_module(NECKS, "neck", config["neck"], backbone.out_channels, backbone.out_strides)
    head = build_module(HEADS, "head", config["head"], neck.out_channels, neck.out_strides, num_classes)
    return Detector(backbone, neck, head)
