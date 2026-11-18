from dataclasses import MISSING, dataclass, fields
from typing import Literal, get_args, get_origin
import inspect
import re

from datatools.transforms import TRANSFORMS
from models.builder import ACTIVATIONS, BACKBONES, HEADS, NECKS

CONNECTING_ARGS = ["in_channels", "in_strides", "num_classes"]


@dataclass
class FieldInfo:
    """One argument or setting shown in the form."""
    name: str
    annotation: str
    default: object
    required: bool
    help: str
    choices: list[str] | None


def parse_docstring(docstring: str | None, section: str) -> dict[str, str]:
    """Read the descriptions of a Google style docstring section such as Args or Attributes.

    Args:
        docstring: docstring to read.
        section: section name without the colon.

    Returns:
        Description of each name, with the continuation lines joined.
    """
    lines = inspect.cleandoc(docstring or "").splitlines()

    if f"{section}:" not in lines:
        return {}

    descriptions, name = {}, None

    for line in lines[lines.index(f"{section}:") + 1:]:
        # the next section, such as Returns, ends this one
        if line and not line.startswith(" "):
            break

        match = re.match(r"^ {4}(\w+): (.*)$", line)

        if match:
            name = match[1]
            descriptions[name] = match[2]
        elif name and line.strip():
            descriptions[name] += " " + line.strip()

    return descriptions


def format_annotation(annotation: object) -> str:
    """Write a type hint in the short form of the code, such as list[int] or float."""
    if annotation is inspect.Parameter.empty:
        return ""

    if isinstance(annotation, type) and not get_args(annotation):
        return annotation.__name__

    return str(annotation).replace("typing.", "").replace("collections.abc.", "")


def describe_value(name: str, annotation: object, default: object) -> tuple[object, list[str] | None]:
    """Convert a default to its config form and find the choices of a name, for activations and literals."""
    if name.endswith("activation"):
        names = {cls: key for key, cls in ACTIVATIONS.items()}
        return names.get(default, default), list(ACTIVATIONS)

    if get_origin(annotation) is Literal:
        return default, [str(a) for a in get_args(annotation)]

    return (list(default) if isinstance(default, tuple) else default), None


def describe_class(cls: type, skipped: list[str]) -> list[FieldInfo]:
    """Describe the constructor arguments of a class from its signature and the Args of its docstring.

    Args:
        cls: class whose constructor takes the arguments.
        skipped: arguments left out, such as the connecting args.

    Returns:
        Argument of the form, in the constructor order.
    """
    helps = parse_docstring(cls.__init__.__doc__, "Args")
    infos = []

    for p in list(inspect.signature(cls.__init__).parameters.values())[1:]:
        if p.name in skipped or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue

        required = p.default is inspect.Parameter.empty
        default, choices = describe_value(p.name, p.annotation, None if required else p.default)
        infos.append(FieldInfo(p.name, format_annotation(p.annotation), default, required, helps.get(p.name, ""), choices))

    return infos


def describe_settings(cls: type) -> list[FieldInfo]:
    """Describe the fields of a settings dataclass from its fields and the Attributes of its docstring.

    Args:
        cls: settings dataclass, such as TrainSettings.

    Returns:
        Field of the form, in the dataclass order.
    """
    helps = parse_docstring(cls.__doc__, "Attributes")
    infos = []

    for f in fields(cls):
        required = f.default is MISSING and f.default_factory is MISSING
        default = None if required else f.default if f.default is not MISSING else f.default_factory()
        infos.append(FieldInfo(f.name, format_annotation(f.type), default, required, helps.get(f.name, ""), None))

    return infos


def list_types() -> dict[str, list[type]]:
    """Return the registered classes of each config block, the backbones, necks, heads and transforms."""
    return {"backbone": BACKBONES, "neck": NECKS, "head": HEADS, "transform": TRANSFORMS}
