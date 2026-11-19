from collections.abc import Callable

from nicegui import run, ui

from webui.bridge.processes import TrainStatus, list_devices, list_statuses
from webui.tabs.common.layout import add_frame, add_page_title
from webui.tabs.common.start_training import open_add_dialog
from webui.tabs.common.texts import t
from webui.tabs.common.widgets import Watcher, add_busy, add_pill
from webui.tabs.trainings.common import add_train_actions, format_seconds


STATES = ["training", "stopped", "done", "failed", "new"]
STATE_COLORS = {"training": "primary", "stopped": "warning", "done": "positive", "failed": "negative", "new": "grey-6"}


@ui.page("/")
async def trainings_page() -> None:
    """Show every training in runs/, with the button that adds a new one."""
    devices = list_devices()
    add_frame(devices, "/")
    training_list = ui.refreshable(add_training_list)

    def update(statuses: list[TrainStatus]) -> None:
        shown["statuses"] = statuses
        training_list.refresh()

    with ui.column().classes("w-full max-w-[1280px] px-8 py-7 gap-5"):
        with add_page_title(t("trainings.title"), t("trainings.subtitle")):
            ui.button(t("action.add"), icon="add", on_click=lambda: open_add_dialog(devices)).props("unelevated no-caps")

        list_area = ui.column().classes("w-full gap-5")

    # the page shows at once, and the statuses fill it after they are read
    with list_area:
        busy = add_busy(t("trainings.loading"))

    await ui.context.client.connected()
    shown = {"state": "all", "statuses": await run.io_bound(list_statuses)}
    busy.delete()

    with list_area:
        training_list(shown, devices, update)
        Watcher(list_statuses, shown["statuses"], update)


def add_training_list(shown: dict, devices: list[tuple[str, str]], on_change: Callable[[list[TrainStatus]], None]) -> None:
    """Add the state filter chips and a row per training, which opens the training on a click.

    Args:
        shown: statuses on the page and the chosen state filter.
        devices: (name, description) of every device.
        on_change: function called with new statuses, after a filter change or a button.
    """
    statuses = shown["statuses"]

    if not statuses:
        with ui.column().classes("card w-full items-center gap-3 py-16"):
            ui.icon("model_training", size="40px").classes("faint")
            ui.label(t("trainings.empty")).classes("text-lg font-semibold")
            ui.label(t("trainings.empty_hint")).classes("text-sm muted")
            ui.button(t("action.add"), icon="add", on_click=lambda: open_add_dialog(devices)).props("unelevated no-caps").classes("mt-2")

        return

    # filter chips
    with ui.row().classes("gap-2"):
        for state in ["all"] + STATES:
            count = len(statuses) if state == "all" else sum(s.state == state for s in statuses)

            if count == 0:
                continue

            chip = ui.row().classes(f"chip {'active' if shown['state'] == state else ''}")
            chip.on("click", lambda state=state: (shown.update(state=state), on_change(statuses)))

            with chip:
                ui.label(t("state.all") if state == "all" else t(f"state.{state}"))
                ui.label(str(count)).classes("opacity-60")

    columns = "100px minmax(180px, 2fr) minmax(160px, 2fr) 70px 150px 100px 80px 110px"

    with ui.column().classes("card w-full gap-0 overflow-x-auto"):
        with ui.grid(columns=columns).classes("w-full gap-x-4 list-head"):
            for key in ["status", "name", "model", "device", "progress", "best_map", "elapsed", "actions"]:
                ui.label(t(f"column.{key}"))

        for status in statuses:
            if shown["state"] not in ("all", status.state):
                continue

            row = ui.grid(columns=columns).classes("w-full gap-x-4 items-center list-row clickable")
            row.on("click", lambda name=status.name: ui.navigate.to(f"/train/{name}"))
            progress = status.epoch / status.epochs if status.epochs else 0
            best_map = f"{status.best_map:.3f}" if status.best_map is not None else "—"

            with row:
                add_pill(t(f"state.{status.state}"), f"state-{status.state}")
                ui.label(status.name).classes("font-semibold text-link truncate")
                ui.label(status.model).classes("text-sm muted truncate")
                ui.label(status.device or "—").classes("mono text-xs")

                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.linear_progress(progress, show_value=False, size="6px").props(f"color={STATE_COLORS[status.state]}").classes("w-20")
                    ui.label(f"{status.epoch} / {status.epochs}").classes("text-xs muted")

                ui.label(best_map).classes("mono")
                ui.label(format_seconds(status.elapsed)).classes("text-sm muted")

                # the buttons must not open the training as the row click does
                with ui.row().classes("gap-1 no-wrap").on("click.stop", lambda: None):
                    add_train_actions(status, devices, lambda: on_change(list_statuses()), flat=True)
