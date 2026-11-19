from pathlib import Path

from nicegui import ui

from webui.bridge.configs import ConfigEntry, check_name, create_config
from webui.tabs.common.texts import t


def describe_name_problem(value: str) -> str | None:
    """Return the problem of a new config name in the current language, None when it is fine or empty."""
    problem = check_name(value) if value else None
    return t(problem) if problem else None


def describe_new_paths(value: str) -> str:
    """Return the config file and the output directory that a new config name makes."""
    return f"configs/user/{value}.yaml → runs/{value}/" if value else ""


def open_new_config_dialog(entries: list[ConfigEntry], base: ConfigEntry | None) -> None:
    """Open the dialog that copies a config into configs/user/ under a new name and opens it."""
    with ui.dialog() as dialog, ui.column().classes("card w-[520px] max-w-[92vw] p-0 gap-0"):
        with ui.row().classes("card-head w-full items-center justify-between"):
            ui.label(t("configs.new")).classes("text-lg font-semibold")
            ui.button(icon="close", on_click=dialog.close).props("flat round dense")

        with ui.column().classes("w-full gap-4 p-5"):
            ui.label(t("configs.new_hint")).classes("text-sm muted")
            options = {str(e.path): e.name + (f"  ({t('configs.preset')})" if e.preset else "") for e in entries}
            default = base or next((e for e in entries if e.preset), None)
            base_select = ui.select(options, value=str(default.path) if default else None, label=t("configs.base")).props("outlined dense").classes("w-full")
            name_input = ui.input(t("configs.name"), validation=describe_name_problem).props("outlined dense").classes("w-full")
            ui.label().bind_text_from(name_input, "value", backward=describe_new_paths).classes("text-xs mono muted")

        def create() -> None:
            problem = check_name(name_input.value or "")

            if problem or not base_select.value:
                ui.notify(t(problem or "configs.base"), type="warning")
                return

            create_config(Path(base_select.value), name_input.value)
            dialog.close()
            ui.notify(t("notify.created", name=name_input.value), type="positive")
            ui.navigate.to(f"/configs/{name_input.value}")

        with ui.row().classes("w-full justify-end gap-2 px-5 py-4").classes("divider-top"):
            ui.button(t("action.cancel"), on_click=dialog.close).props("flat no-caps")
            ui.button(t("action.create"), on_click=create).props("unelevated no-caps")

    dialog.open()
