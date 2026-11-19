from collections.abc import Callable

from nicegui import ui

from shared.output_directory import OutputDirectory
from webui.bridge.configs import find_config
from webui.bridge.processes import RUNS_DIR, TrainStatus, list_busy_devices, start_train, stop_process
from webui.tabs.common.texts import t


def format_seconds(seconds: float) -> str:
    """Write seconds as h:mm:ss."""
    seconds = round(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def add_device_menu(devices: list[tuple[str, str]], label: str, on_choose: Callable[[str], None], props: str = "unelevated") -> None:
    """Add a button with a menu of the devices, where only the free ones can be chosen.

    Args:
        devices: (name, description) of every device.
        label: text of the button.
        on_choose: function called with the chosen device name.
        props: quasar props of the button.
    """
    busy = list_busy_devices()

    with ui.button(label, icon="play_arrow").props(f"{props} dense no-caps").classes("px-3"):
        with ui.menu().props("auto-close"):
            ui.label(t("devices.choose")).classes("caption px-4 pt-2 pb-1")

            for name, description in devices:
                item = ui.menu_item(on_click=lambda name=name: on_choose(name))

                with item:
                    with ui.row().classes("items-center gap-3 no-wrap"):
                        ui.label(name).classes("mono font-semibold w-16")
                        ui.label(f"{description} · {t('devices.in_use')} ({busy[name]})" if name in busy else description).classes("text-sm muted")

                if name in busy:
                    item.disable()


def resume(name: str, device: str, on_done: Callable[[], None]) -> None:
    """Start a stopped or failed training again from its last.pt on a device."""
    entry = find_config(name)

    if entry is None:
        ui.notify(t("notify.no_config", name=name), type="warning")
        return

    start_train(entry.path, device)
    ui.notify(t("notify.started", name=name, device=device), type="positive")
    on_done()


def add_train_actions(status: TrainStatus, devices: list[tuple[str, str]], on_done: Callable[[], None], flat: bool) -> None:
    """Add the stop button of a running training or the resume menu of a stopped or failed one."""
    if status.state == "training":
        def stop() -> None:
            stop_process(OutputDirectory(RUNS_DIR / status.name).pid)
            ui.notify(t("notify.stopped", name=status.name))
            on_done()

        ui.button(t("action.stop"), icon="stop", on_click=stop, color="negative").props(f"{'flat' if flat else 'outline'} dense no-caps").classes("px-3")
    elif status.state in ("stopped", "failed", "new"):
        # a training stopped before its first epoch has no last.pt, so it starts over
        label = t("action.start") if status.state == "new" else t("action.resume")
        add_device_menu(devices, label, lambda device: resume(status.name, device, on_done), "flat" if flat else "outline")
