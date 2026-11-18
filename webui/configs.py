from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import re

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from shared.output_directory import OutputDirectory
from shared.settings import CommonSettings, PredictSettings, TrainSettings
from webui.schema import CONNECTING_ARGS, FieldInfo, describe_class, describe_settings, list_types

PRESET_DIR = Path("configs")
USER_DIR = Path("configs/user")
SECTIONS = {"common": CommonSettings, "train": TrainSettings, "predict": PredictSettings}
TRANSFORM_LISTS = [("train", "transforms"), ("train", "val_transforms"), ("predict", "transforms")]
MODEL_BLOCKS = ["backbone", "neck", "head"]
TYPES = {"int": int, "float": (int, float), "bool": bool, "str": str}


@dataclass
class ConfigEntry:
    """A config file and whether it can be edited."""
    name: str
    path: Path
    preset: bool
    locked: bool


def list_configs() -> list[ConfigEntry]:
    """List the user configs in configs/user/ and then the presets in configs/, locking the presets and the started ones."""
    user = [ConfigEntry(p.stem, p, False, OutputDirectory.from_config(p).last.exists()) for p in sorted(USER_DIR.glob("*.yaml"))]
    presets = [ConfigEntry(p.stem, p, True, True) for p in sorted(PRESET_DIR.glob("*.yaml"))]
    return user + presets


def find_config(name: str) -> ConfigEntry | None:
    """Return the config of a name, None when there is none."""
    return next((entry for entry in list_configs() if entry.name == name), None)


def check_name(name: str) -> str | None:
    """Return the text key of the problem with a new config name, None when it can be used."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        return "configs.name_invalid"

    # the training of the config would share the output directory of the same name
    if find_config(name) or (Path("runs") / name).exists():
        return "configs.name_taken"

    return None


def create_config(base_path: Path, name: str) -> Path:
    """Copy a config to configs/user/<name>.yaml.

    Args:
        base_path: config to copy, with its comments.
        name: name of the new config, checked with check_name.

    Returns:
        Path of the new config.
    """
    problem = check_name(name)

    if problem:
        raise ValueError(problem)

    USER_DIR.mkdir(parents=True, exist_ok=True)
    path = USER_DIR / f"{name}.yaml"
    path.write_text(base_path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def create_yaml() -> YAML:
    """Create the round-trip yaml reader and writer, which keeps the comments, the order and the flow lists."""
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096
    return yaml


def load_config(path: Path) -> CommentedMap:
    """Read a config with its comments, for editing and saving it again."""
    return create_yaml().load(path.read_text(encoding="utf-8"))


def write_config(config: CommentedMap) -> str:
    """Write a config as yaml text, with its comments."""
    stream = StringIO()
    create_yaml().dump(config, stream)
    return stream.getvalue()


def describe_block(block: dict, registry: list[type]) -> list[FieldInfo]:
    """Describe the arguments of a model or transform block, empty when its type is unknown."""
    classes = {cls.__name__: cls for cls in registry}
    cls = classes.get(block.get("type"))
    return describe_class(cls, CONNECTING_ARGS) if cls else []


def check_values(where: str, values: dict, infos: list[FieldInfo], strict: bool) -> list[str]:
    """Check the keys and the simple types of a mapping against the fields of a class.

    Args:
        where: location for the messages, such as train or model.head.
        values: mapping of the config.
        infos: fields that the mapping may have.
        strict: report the missing required fields too.

    Returns:
        Problems found.
    """
    known = {info.name: info for info in infos}
    problems = [f"{where}: unknown key {key}" for key in values if key not in known]

    if strict:
        problems += [f"{where}: missing key {info.name}" for info in infos if info.required and info.name not in values]

    for key, value in values.items():
        info = known.get(key)

        if info is None:
            continue

        expected = TYPES.get(info.annotation)
        is_list = info.annotation.startswith(("list", "tuple"))

        if info.choices and value not in info.choices:
            problems.append(f"{where}.{key}: {value} is not one of {info.choices}")
        elif expected and (not isinstance(value, expected) or (isinstance(value, bool) and expected is not bool)):
            problems.append(f"{where}.{key}: expected {info.annotation}, got {value!r}")
        elif is_list and not isinstance(value, list):
            problems.append(f"{where}.{key}: expected a list, got {value!r}")

    return problems


def check_config(config: dict) -> list[str]:
    """Check a config against the code: sections, keys, simple types, model types and transform types.

    Args:
        config: config read with load_config.

    Returns:
        Problems found, empty when the config can be saved.
    """
    if not isinstance(config, dict):
        return ["the config is not a mapping"]

    types = list_types()
    problems = [f"missing section {name}" for name in ["model", *SECTIONS] if name not in config]

    for name, cls in SECTIONS.items():
        if isinstance(config.get(name), dict):
            problems += check_values(name, config[name], describe_settings(cls), strict=True)

    for name in MODEL_BLOCKS:
        block = (config.get("model") or {}).get(name)
        classes = [cls.__name__ for cls in types[name]]

        if not isinstance(block, dict) or block.get("type") not in classes:
            problems.append(f"model.{name}: type must be one of {classes}")
            continue

        problems += check_values(f"model.{name}", block.get("args") or {}, describe_block(block, types[name]), strict=True)

    for section, key in TRANSFORM_LISTS:
        classes = [cls.__name__ for cls in types["transform"]]

        for i, block in enumerate((config.get(section) or {}).get(key) or []):
            if not isinstance(block, dict) or block.get("type") not in classes:
                problems.append(f"{section}.{key}[{i}]: type must be one of {classes}")
                continue

            problems += check_values(f"{section}.{key}[{i}]", block.get("args") or {}, describe_block(block, types["transform"]), strict=True)

    return problems


def find_comment_column(value: object) -> int | None:
    """Return the rightmost column of the end-of-line comments in a config, to align the new ones with them."""
    columns = []

    if isinstance(value, CommentedMap):
        columns += [items[2].column for items in value.ca.items.values() if len(items) > 2 and items[2] is not None]
        columns += [find_comment_column(v) or 0 for v in value.values()]

    if isinstance(value, list):
        columns += [find_comment_column(v) or 0 for v in value]

    return max(columns, default=0) or None


def add_comments(values: object, infos: list[FieldInfo], column: int | None) -> None:
    """Add the first sentence of the description as a comment to the keys that have none.

    Args:
        values: mapping of the config, skipped when it is not a commented mapping.
        infos: fields of the mapping with their descriptions.
        column: column of the new comments.
    """
    if not isinstance(values, CommentedMap):
        return

    helps = {info.name: info.help.split(". ")[0].rstrip(".") for info in infos if info.help}

    for key in values:
        has_comment = key in values.ca.items and len(values.ca.items[key]) > 2 and values.ca.items[key][2] is not None

        if key in helps and not has_comment and not isinstance(values[key], dict):
            values.yaml_add_eol_comment(helps[key], key, column=column)


def save_config(entry: ConfigEntry, config: CommentedMap) -> None:
    """Save a user config, keeping its comments and adding the description of the keys without one.

    Args:
        entry: config to overwrite, refused when it is locked.
        config: config checked with check_config.
    """
    if entry.locked:
        raise ValueError(f"{entry.name} is locked")

    types = list_types()
    column = find_comment_column(config)

    for name, cls in SECTIONS.items():
        add_comments(config.get(name), describe_settings(cls), column)

    for name in MODEL_BLOCKS:
        block = config["model"][name]
        add_comments(block.get("args"), describe_block(block, types[name]), column)

    for section, key in TRANSFORM_LISTS:
        for block in config[section].get(key) or []:
            add_comments(block.get("args"), describe_block(block, types["transform"]), column)

    entry.path.write_text(write_config(config), encoding="utf-8")
