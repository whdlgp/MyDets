from nicegui import ui
from ruamel.yaml.comments import CommentedMap

from webui.bridge.configs import ConfigEntry, check_config, create_yaml, find_config, is_trained, list_configs, load_config, save_config, write_config
from webui.bridge.processes import list_devices
from webui.tabs.common.layout import add_frame, get_theme
from webui.tabs.common.start_training import open_add_dialog
from webui.tabs.common.texts import t
from webui.tabs.common.widgets import add_pill
from webui.tabs.configs.common import open_new_config_dialog
from webui.tabs.configs.subpage.form import add_config_form


@ui.page("/configs/{name}")
def editor_page(name: str) -> None:
    """Build a config with blocks and fields made from the code, or edit it as text."""
    devices = list_devices()
    add_frame(devices, "/configs")
    entry = find_config(name)

    with ui.column().classes("w-full max-w-[1400px] px-8 py-7 gap-4"):
        ui.link(t("action.back_configs"), "/configs").classes("text-sm muted no-underline")

        if entry is None:
            ui.label(t("configs.missing", name=name)).classes("muted")
            return

        ConfigEditor(entry, devices)


class ConfigEditor:
    """Form and text editor of one config, with its preview, save and start buttons."""

    def __init__(self, entry: ConfigEntry, devices: list[tuple[str, str]]) -> None:
        """Add the editor of a config.

        Args:
            entry: config to edit, read only when it is a preset or trained.
            devices: (name, description) of every device, for the start dialog.
        """
        self.entry = entry
        self.devices = devices
        self.config = load_config(entry.path)
        self.readonly = entry.locked
        self.dirty = False
        self.add_header()

        if self.readonly:
            locked = t("editor.locked_preset") if entry.preset else t("editor.locked_trained")

            with ui.row().classes("w-full items-center gap-2 px-4 py-3 rounded-lg banner-warn"):
                ui.icon("lock", size="18px")
                ui.label(locked).classes("text-sm")

        self.errors = ui.column().classes("w-full gap-1")
        text_theme = "basicDark" if get_theme()[0] == "dark" else "basicLight"

        with ui.row().classes("w-full gap-5 items-start no-wrap"):
            self.form = ui.column().classes("flex-1 gap-4 min-w-0")
            self.text = ui.codemirror(write_config(self.config), language="YAML", theme=text_theme).classes("flex-1 min-w-0 card overflow-hidden")
            self.text.set_visibility(False)

            if self.readonly:
                self.text.props("disable")

            with ui.column().classes("card w-[400px] gap-0 sticky top-4") as self.preview_card:
                with ui.column().classes("card-head w-full gap-0"):
                    ui.label(t("editor.preview")).classes("section-title")
                    ui.label(t("editor.preview_hint")).classes("caption")

                self.preview = ui.code(write_config(self.config), language="yaml").classes("w-full text-xs max-h-[75vh] overflow-y-auto").style("border-radius:0 0 12px 12px")

        self.mode.on_value_change(self.switch_mode)

        if not self.readonly:
            self.text.on_value_change(self.mark_text_changed)
            self.save_button.on_click(self.save)

        if not is_trained(entry):
            self.start_button.on_click(self.start)

        self.build_form()

    def add_header(self) -> None:
        """Add the name, the state pills, the path and the buttons of the config."""
        entry = self.entry

        with ui.row().classes("w-full items-start justify-between no-wrap"):
            with ui.column().classes("gap-1"):
                with ui.row().classes("items-center gap-3"):
                    ui.label(entry.name).classes("text-2xl font-bold")

                    if entry.preset:
                        add_pill(t("configs.preset"), "tag-grey")
                    elif self.readonly:
                        add_pill(t("configs.trained"), "tag-orange")
                    else:
                        add_pill(t("configs.editable"), "tag-blue")

                    with ui.row() as self.unsaved:
                        add_pill(t("editor.unsaved"), "tag-orange")

                    self.unsaved.set_visibility(False)

                ui.label(entry.path.as_posix()).classes("text-xs mono muted")

            with ui.row().classes("items-center gap-2 no-wrap"):
                self.mode = ui.toggle({"form": t("editor.form"), "text": t("editor.text")}, value="form").props("no-caps dense unelevated toggle-color=primary")

                if self.readonly:
                    ui.button(t("action.copy_new"), icon="content_copy", on_click=lambda: open_new_config_dialog(list_configs(), entry)).props("outline no-caps")
                else:
                    ui.button(t("action.revert"), on_click=ui.navigate.reload).props("outline no-caps")
                    self.save_button = ui.button(t("action.save"), icon="save").props("outline no-caps")

                # a preset can be trained as it is, a trained config only through a copy
                if not is_trained(entry):
                    self.start_button = ui.button(t("action.start_training"), icon="play_arrow").props("unelevated no-caps")

    def mark_changed(self) -> None:
        """Show the unsaved pill and the new preview after a change in the form."""
        self.mark_text_changed()
        self.preview.set_content(write_config(self.config))

    def mark_text_changed(self) -> None:
        """Show the unsaved pill after a change in the text."""
        self.dirty = True
        self.unsaved.set_visibility(True)

    def build_form(self) -> None:
        """Draw the form of the config again."""
        self.form.clear()

        with self.form:
            add_config_form(self.config, self.readonly, self.mark_changed)

    def switch_mode(self, event: object) -> None:
        """Switch between the form and the text, carrying the changes over."""
        if event.value == "text":
            self.text.set_value(write_config(self.config))
        else:
            parsed = parse_text(self.text.value)

            if parsed is None:
                self.mode.set_value("text")
                return

            self.config = parsed
            self.build_form()
            self.preview.set_content(write_config(self.config))

        self.form.set_visibility(event.value == "form")
        self.preview_card.set_visibility(event.value == "form")
        self.text.set_visibility(event.value == "text")

    def save(self) -> bool:
        """Check and save the config, showing the problems instead when there are any.

        Returns:
            Whether the config was saved.
        """
        if self.mode.value == "text":
            parsed = parse_text(self.text.value)

            if parsed is None:
                return False

            self.config = parsed

        problems = check_config(self.config)
        self.errors.clear()

        if problems:
            with self.errors:
                with ui.column().classes("w-full gap-1 px-4 py-3 rounded-lg banner-bad"):
                    ui.label(t("editor.errors")).classes("text-sm font-semibold")

                    for problem in problems:
                        ui.label(problem).classes("text-sm mono")

            return False

        save_config(self.entry, self.config)
        self.config = load_config(self.entry.path)
        self.text.set_value(write_config(self.config))
        self.preview.set_content(write_config(self.config))
        self.build_form()
        self.dirty = False
        self.unsaved.set_visibility(False)
        ui.notify(t("editor.saved"), type="positive")
        return True

    def start(self) -> None:
        """Open the start dialog, saving the changes first since the training reads the file."""
        if self.dirty and not self.save():
            return

        open_add_dialog(self.devices, self.entry.name)


def parse_text(value: str) -> CommentedMap | None:
    """Parse the yaml text of the text mode, notifying and returning None when it is not valid."""
    try:
        return create_yaml().load(value)
    except Exception as error:
        ui.notify(t("editor.invalid_yaml", error=str(error).splitlines()[0]), type="negative", multi_line=True)
        return None
