from collections.abc import Callable
import inspect

from nicegui import ui
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from shared.settings import CommonSettings, PredictSettings, TrainSettings
from webui.bridge.configs import create_yaml
from webui.bridge.schema import CONNECTING_ARGS, FieldInfo, describe_class, describe_settings, list_types
from webui.tabs.common.texts import t


SECTIONS = [("common", CommonSettings), ("train", TrainSettings), ("predict", PredictSettings)]
TRANSFORM_LISTS = {"train": ["transforms", "val_transforms"], "predict": ["transforms"]}


def describe_type(cls: type) -> str:
    """Return the first line of the docstring of a registered class."""
    return (inspect.getdoc(cls) or "").splitlines()[0] if inspect.getdoc(cls) else ""


def add_section_card(section: str) -> ui.column:
    """Add the card of a config section with its yaml key and its meaning, returning the column for its content."""
    with ui.column().classes("card w-full gap-0"):
        with ui.row().classes("card-head w-full items-baseline gap-3"):
            ui.label(section).classes("section-title mono")
            ui.label(t(f"section.{section}")).classes("caption")

        return ui.column().classes("w-full gap-4 p-5")


def add_config_form(config: CommentedMap, readonly: bool, on_change: Callable[[], None]) -> None:
    """Add the form of every config section, built from the settings classes and the registered classes."""
    types = list_types()

    with add_section_card("model"):
        for block_name in ["backbone", "neck", "head"]:
            add_model_block(config["model"], block_name, types[block_name], readonly, on_change)

    for section, cls in SECTIONS:
        with add_section_card(section):
            infos = [info for info in describe_settings(cls) if not info.annotation.startswith("list[dict")]
            add_fields(config, section, infos, readonly, on_change)

            for key in TRANSFORM_LISTS.get(section, []):
                TransformList(config[section], key, types["transform"], readonly, on_change)


def add_fields(parent: CommentedMap, key: str, infos: list[FieldInfo], readonly: bool, on_change: Callable[[], None], columns: int = 3) -> None:
    """Add an input for each field of a mapping, which writes the value into the config on change."""
    def get_mapping() -> CommentedMap:
        if not isinstance(parent.get(key), dict):
            parent[key] = CommentedMap()

        return parent[key]

    values = parent.get(key) or {}

    with ui.grid(columns=columns).classes("w-full gap-x-4 gap-y-3"):
        for info in infos:
            add_field(get_mapping, info, values.get(info.name, info.default), readonly, on_change)


def add_field(get_mapping: Callable[[], CommentedMap], info: FieldInfo, value: object, readonly: bool, on_change: Callable[[], None]) -> None:
    """Add the input of one field, chosen by its type hint, with its type and description as a tooltip."""
    def write(new_value: object) -> None:
        get_mapping()[info.name] = convert_value(new_value, info)
        on_change()

    with ui.column().classes("gap-1 min-w-0"):
        with ui.row().classes("items-center gap-1 no-wrap"):
            ui.label(info.name).classes("text-xs mono font-semibold truncate label-strong")

            with ui.icon("help_outline", size="15px").classes("faint cursor-help"):
                default = f" = {info.default}" if not info.required else ""
                help_text = f"\n{info.help}" if info.help else ""
                ui.tooltip(f"{info.name}: {info.annotation}{default}{help_text}").classes("whitespace-pre-line")

        if info.choices:
            element = ui.select(info.choices, value=value if value in info.choices else None, on_change=lambda e: write(e.value))
        elif info.annotation == "bool":
            element = ui.checkbox(t("editor.on"), value=bool(value), on_change=lambda e: write(e.value)).props("dense")
        elif info.annotation in ("int", "float"):
            element = ui.number(value=value, step=1 if info.annotation == "int" else None, on_change=lambda e: write(e.value))
        elif info.annotation.startswith(("list", "tuple")):
            element = ui.input(value=", ".join(str(v) for v in value) if value is not None else "", on_change=lambda e: write(e.value))
        else:
            element = ui.input(value="" if value is None else str(value), on_change=lambda e: write(e.value))

        if info.annotation != "bool":
            element.props("dense outlined").classes("w-full")

        if readonly:
            element.disable()


def convert_value(value: object, info: FieldInfo) -> object:
    """Convert the value of an input to the config value of a field."""
    if info.annotation == "int" and value is not None:
        return int(value)

    if info.annotation.startswith(("list", "tuple")):
        # the round-trip reader keeps the [a, b] form in the saved file
        return create_yaml().load(f"[{value or ''}]")

    return value


def add_type_menu(registry: list[type], label: str, icon: str, on_choose: Callable[[str], None], props: str) -> ui.button:
    """Add a button with a menu of the registered classes and the first line of their docstrings."""
    with ui.button(label, icon=icon).props(f"{props} dense no-caps").classes("px-3 shrink-0") as button:
        with ui.menu().props("auto-close").classes("max-h-[60vh]"):
            for cls in registry:
                with ui.menu_item(on_click=lambda name=cls.__name__: on_choose(name)):
                    with ui.column().classes("gap-0 py-1"):
                        ui.label(cls.__name__).classes("font-semibold text-sm")
                        ui.label(describe_type(cls)).classes("text-xs muted")

    return button


def make_default_args(cls: type, skipped: list[str]) -> CommentedMap:
    """Make the arguments of a new block with the defaults of its class, leaving out the required ones for the user to fill."""
    args = CommentedMap()

    for info in describe_class(cls, skipped):
        if info.required:
            continue

        if isinstance(info.default, (list, tuple)):
            args[info.name] = convert_value(", ".join(str(v) for v in info.default), info)
        else:
            args[info.name] = info.default

    return args


def add_model_block(model: CommentedMap, block_name: str, registry: list[type], readonly: bool, on_change: Callable[[], None]) -> None:
    """Add a backbone, neck or head block with its type, the button that changes it and the inputs of its arguments."""
    block = model[block_name]
    classes = {cls.__name__: cls for cls in registry}

    @ui.refreshable
    def model_block() -> None:
        cls = classes.get(block["type"])

        with ui.column().classes("box w-full p-4 gap-3"):
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                with ui.column().classes("gap-0 min-w-0 flex-1"):
                    ui.label(block_name).classes("block-kind")
                    ui.label(str(block["type"])).classes("text-lg font-semibold")
                    ui.label(describe_type(cls) if cls else t("editor.unknown_type")).classes("text-xs muted truncate")

                if not readonly:
                    add_type_menu(registry, t("action.change_type"), "swap_horiz", change_type, "outline")

            if cls is not None:
                add_fields(block, "args", describe_class(cls, CONNECTING_ARGS), readonly, on_change)

    def change_type(new_type: str) -> None:
        # the arguments of the old type mean nothing to the new one
        block["type"] = new_type
        block["args"] = make_default_args(classes[new_type], CONNECTING_ARGS)
        model_block.refresh()
        on_change()

    model_block()


class TransformList:
    """Transform list of a config section as blocks in order, with buttons to add, move and delete transforms."""

    def __init__(self, section: CommentedMap, key: str, registry: list[type], readonly: bool, on_change: Callable[[], None]) -> None:
        """Add the list.

        Args:
            section: config section that has the list.
            key: key of the list in the section, such as transforms.
            registry: transform classes that can be added.
            readonly: show the list without buttons and with disabled inputs.
            on_change: function called after any change.
        """
        self.section = section
        self.key = key
        self.registry = registry
        self.classes = {cls.__name__: cls for cls in registry}
        self.readonly = readonly
        self.on_change = on_change
        self.blocks_view = ui.refreshable(self.add_blocks)
        self.blocks_view()

    def get_blocks(self) -> CommentedSeq:
        """Return the list in the config, made empty when it is missing."""
        if not isinstance(self.section.get(self.key), list):
            self.section[self.key] = CommentedSeq()

        return self.section[self.key]

    def add(self, type_name: str) -> None:
        """Add a transform at the end with the defaults of its class."""
        args = make_default_args(self.classes[type_name], [])
        self.get_blocks().append(CommentedMap({"type": type_name, "args": args}))
        self.changed()

    def move(self, index: int, offset: int) -> None:
        """Move a transform up or down."""
        blocks = self.get_blocks()

        if 0 <= index + offset < len(blocks):
            blocks.insert(index + offset, blocks.pop(index))
            self.changed()

    def delete(self, index: int) -> None:
        """Delete a transform."""
        self.get_blocks().pop(index)
        self.changed()

    def changed(self) -> None:
        """Draw the list again and report the change."""
        self.blocks_view.refresh()
        self.on_change()

    def add_blocks(self) -> None:
        """Add a block per transform and the tile that adds one."""
        blocks = self.section.get(self.key) or []

        with ui.column().classes("w-full gap-2 pt-2").classes("divider-top"):
            with ui.row().classes("items-baseline gap-2"):
                ui.label(self.key).classes("text-sm mono font-semibold")
                ui.label(t("editor.in_order", count=len(blocks))).classes("caption")

            for i, block in enumerate(blocks):
                self.add_block(i, block)

            if not self.readonly:
                with ui.row().classes("add-tile w-full items-center gap-2"):
                    ui.icon("add", size="20px")
                    ui.label(t("action.add_transform"))

                    with ui.menu().props("auto-close fit").classes("max-h-[60vh]"):
                        for cls in self.registry:
                            with ui.menu_item(on_click=lambda name=cls.__name__: self.add(name)):
                                with ui.column().classes("gap-0 py-1"):
                                    ui.label(cls.__name__).classes("font-semibold text-sm")
                                    ui.label(describe_type(cls)).classes("text-xs muted")

    def add_block(self, index: int, block: CommentedMap) -> None:
        """Add one transform with its buttons and the inputs of its arguments."""
        cls = self.classes.get(block["type"])
        description = describe_type(cls) if cls else t("editor.unknown_type")

        with ui.column().classes("box w-full px-4 py-3 gap-2"):
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                with ui.row().classes("items-center gap-3 no-wrap min-w-0"):
                    ui.label(str(index + 1)).classes("w-6 h-6 rounded-full text-xs font-semibold flex items-center justify-center index-dot")
                    ui.label(str(block["type"])).classes("font-semibold")
                    ui.label(description).classes("text-xs muted truncate")

                if not self.readonly:
                    with ui.row().classes("gap-0 no-wrap"):
                        ui.button(icon="arrow_upward", on_click=lambda: self.move(index, -1)).props("flat dense round size=sm").tooltip(t("action.move_up"))
                        ui.button(icon="arrow_downward", on_click=lambda: self.move(index, 1)).props("flat dense round size=sm").tooltip(t("action.move_down"))
                        ui.button(icon="delete_outline", on_click=lambda: self.delete(index), color="negative").props("flat dense round size=sm").tooltip(t("action.delete"))

            infos = describe_class(cls, []) if cls else []

            if infos:
                add_fields(block, "args", infos, self.readonly, self.on_change, columns=4)
