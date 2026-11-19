from nicegui import ui

from shared.settings import read_config
from webui.bridge.configs import ConfigEntry, find_config, is_trained, list_configs, summarize_model
from webui.bridge.processes import list_busy_devices, start_train
from webui.tabs.common.texts import t
from webui.tabs.common.widgets import add_pill


def open_add_dialog(devices: list[tuple[str, str]], config_name: str | None = None) -> None:
    """Open the dialog that starts a new training from a config on a free device.

    Args:
        devices: (name, description) of every device.
        config_name: config chosen at the start, such as the one open in the editor.
    """
    entries = list_configs()
    busy = list_busy_devices()
    free = [name for name, _ in devices if name not in busy]
    chosen = {"config": config_name, "device": free[0] if free else None}
    choices = {"config": {}, "device": {}}

    def choose(kind: str, name: str) -> None:
        # only the classes change, so that the list keeps its scroll position
        for key, element in choices[kind].items():
            element.classes(add="selected" if key == name else "", remove="" if key == name else "selected")

        chosen[kind] = name
        update_run()

    with ui.dialog() as dialog, ui.column().classes("card w-[680px] max-w-[92vw] p-0 gap-0"):
        with ui.row().classes("card-head w-full items-center justify-between"):
            ui.label(t("add.title")).classes("text-lg font-semibold")
            ui.button(icon="close", on_click=dialog.close).props("flat round dense")

        with ui.column().classes("w-full gap-2 px-5 pt-4 pb-3").classes("divider-bottom"):
            ui.label(t("add.device")).classes("section-title")

            with ui.row().classes("w-full gap-2"):
                for name, description in devices:
                    style = "disabled" if name in busy else "selected" if chosen["device"] == name else ""
                    choice = ui.column().classes(f"choice gap-0 min-w-[150px] {style}")

                    if name not in busy:
                        choice.on("click", lambda name=name: choose("device", name))
                        choices["device"][name] = choice

                    with choice:
                        ui.label(name).classes("mono font-semibold")
                        ui.label(f"{t('devices.in_use')} · {busy[name]}" if name in busy else description).classes("text-xs muted")

        with ui.column().classes("w-full gap-3 px-5 py-4 max-h-[52vh] overflow-y-auto no-wrap"):
            with ui.row().classes("w-full items-baseline justify-between"):
                ui.label(t("add.config")).classes("section-title")
                ui.link(t("add.new_config"), "/configs").classes("text-sm")

            for group, title in [(False, t("configs.mine")), (True, t("configs.presets"))]:
                group_entries = [e for e in entries if e.preset == group]

                if not group_entries:
                    continue

                ui.label(title).classes("caption font-semibold -mb-1")

                for entry in group_entries:
                    choice = add_config_choice(entry, chosen["config"] == entry.name)

                    if choice is not None:
                        choice.on("click", lambda name=entry.name: choose("config", name))
                        choices["config"][entry.name] = choice

        with ui.row().classes("w-full items-center justify-end gap-3 px-5 py-4").classes("divider-top"):
            missing_label = ui.label().classes("text-sm text-warn")
            ui.button(t("action.cancel"), on_click=dialog.close).props("flat no-caps")
            run_button = ui.button(t("action.run"), icon="play_arrow", on_click=lambda: run_training(dialog, chosen)).props("unelevated no-caps")

    def update_run() -> None:
        missing = t("add.choose_device") if chosen["device"] is None else t("add.choose_config") if chosen["config"] is None else ""
        missing_label.set_text(missing)
        run_button.set_enabled(not missing)

    update_run()
    dialog.open()


def add_config_choice(entry: ConfigEntry, selected: bool) -> ui.row | None:
    """Add one config of the add dialog, returning the element to click, None when its training exists."""
    config = read_config(entry.path)
    trained = is_trained(entry)
    choice = ui.row().classes(f"choice w-full items-center justify-between no-wrap {'disabled' if trained else 'selected' if selected else ''}")

    with choice:
        with ui.column().classes("gap-0 min-w-0"):
            ui.label(entry.name).classes("font-semibold mono text-sm truncate")
            ui.label(f"{summarize_model(config)} · {config['train']['dataset_dir']} · {config['train']['epochs']} epochs").classes("text-xs muted truncate")

        if trained:
            add_pill(t("add.trained"), "tag-grey")

    return None if trained else choice


def run_training(dialog: ui.dialog, chosen: dict) -> None:
    """Start the training of the chosen config on the chosen device and open it."""
    entry = find_config(chosen["config"])
    start_train(entry.path, chosen["device"])
    dialog.close()
    ui.notify(t("notify.started", name=entry.name, device=chosen["device"]), type="positive")
    ui.navigate.to(f"/train/{entry.name}")
