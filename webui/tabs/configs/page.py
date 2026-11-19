from nicegui import ui

from shared.settings import read_config
from webui.bridge.configs import ConfigEntry, is_trained, list_configs, summarize_model
from webui.bridge.processes import list_devices
from webui.tabs.common.layout import add_frame, add_page_title
from webui.tabs.common.texts import t
from webui.tabs.common.widgets import add_pill
from webui.tabs.configs.common import open_new_config_dialog


@ui.page("/configs")
def configs_page() -> None:
    """Show the user configs and the presets, where new configs are made from a copy."""
    add_frame(list_devices(), "/configs")
    entries = list_configs()
    mine = [e for e in entries if not e.preset]

    with ui.column().classes("w-full max-w-[1280px] px-8 py-7 gap-5"):
        with add_page_title(t("configs.title"), t("configs.subtitle")):
            ui.button(t("action.new_config"), icon="add", on_click=lambda: open_new_config_dialog(entries, None)).props("unelevated no-caps")

        with ui.column().classes("card w-full gap-0"):
            with ui.row().classes("card-head w-full items-baseline gap-2"):
                ui.label(t("configs.mine")).classes("section-title")
                ui.label("configs/user/").classes("caption mono")

            if not mine:
                with ui.column().classes("w-full items-center gap-2 py-10"):
                    ui.label(t("configs.empty")).classes("text-sm muted")
                    ui.button(t("action.new_config"), icon="add", on_click=lambda: open_new_config_dialog(entries, None)).props("outline no-caps")

            for entry in mine:
                add_config_row(entry, entries)

        with ui.column().classes("card w-full gap-0"):
            with ui.row().classes("card-head w-full items-baseline gap-2"):
                ui.label(t("configs.presets")).classes("section-title")
                ui.label(f"configs/ · {t('configs.readonly')}").classes("caption mono")

            for entry in [e for e in entries if e.preset]:
                add_config_row(entry, entries)


def add_config_row(entry: ConfigEntry, entries: list[ConfigEntry]) -> None:
    """Add one config with its model, its state and its buttons."""
    config = read_config(entry.path)
    row = ui.row().classes("w-full items-center gap-4 list-row clickable no-wrap")
    row.on("click", lambda: ui.navigate.to(f"/configs/{entry.name}"))

    with row:
        with ui.column().classes("gap-0 flex-1 min-w-0"):
            ui.label(entry.name).classes("font-semibold mono text-sm truncate")
            ui.label(f"{summarize_model(config)} · {config['train']['dataset_dir']} · {config['train']['epochs']} epochs").classes("text-xs muted truncate")

        if entry.preset:
            add_pill(t("configs.preset"), "tag-grey")
        elif is_trained(entry):
            add_pill(t("configs.trained"), "tag-orange")
        else:
            add_pill(t("configs.editable"), "tag-blue")

        with ui.row().classes("gap-1 no-wrap").on("click.stop", lambda: None):
            open_label = t("action.view") if entry.locked else t("action.edit")
            ui.button(open_label, on_click=lambda: ui.navigate.to(f"/configs/{entry.name}")).props("flat dense no-caps").classes("px-2")
            ui.button(t("action.copy"), on_click=lambda: open_new_config_dialog(entries, entry)).props("flat dense no-caps").classes("px-2")
