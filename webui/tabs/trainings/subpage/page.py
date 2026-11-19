from collections.abc import Callable
import json

from nicegui import run, ui

from shared.output_directory import OutputDirectory
from shared.settings import read_config
from webui.bridge.configs import find_config, summarize_model
from webui.bridge.processes import RUNS_DIR, TrainStatus, find_process, list_devices, make_report, read_status
from webui.tabs.common.layout import add_frame
from webui.tabs.common.texts import t
from webui.tabs.common.widgets import Watcher, add_pill
from webui.tabs.trainings.common import add_train_actions
from webui.tabs.trainings.subpage.export import add_export
from webui.tabs.trainings.subpage.monitor import Monitor
from webui.tabs.trainings.subpage.prediction import add_predict


@ui.page("/train/{name}")
def training_page(name: str) -> None:
    """Show one training, with the monitor, results, predict and export tabs."""
    devices = list_devices()
    add_frame(devices, "/")
    output = OutputDirectory(RUNS_DIR / name)

    with ui.column().classes("w-full max-w-[1280px] px-8 py-7 gap-4"):
        ui.link(t("action.back_trainings"), "/").classes("text-sm muted no-underline")

        # a training that was just started writes its config.yaml a moment later
        if not output.config.exists():
            starting = find_process(output.pid) is not None
            ui.label(t("trainings.starting", name=name) if starting else t("trainings.missing", name=name)).classes("muted")

            if starting:
                ui.timer(1, ui.navigate.reload, once=True)

            return

        shown = {"status": read_status(output.path)}
        header = ui.refreshable(add_training_header)

        def update(status: TrainStatus) -> None:
            shown["status"] = status
            header.refresh()

        header(output, shown, devices, update)
        Watcher(lambda: read_status(output.path), shown["status"], update)

        with ui.tabs().classes("w-full divider-bottom").props("align=left no-caps dense") as tabs:
            monitor_tab = ui.tab("monitor", t("tab.monitor"))
            results_tab = ui.tab("results", t("tab.results"))
            predict_tab = ui.tab("predict", t("tab.predict"))
            export_tab = ui.tab("export", t("tab.export"))

        with ui.tab_panels(tabs, value=monitor_tab).classes("w-full bg-transparent").props("animated=false"):
            with ui.tab_panel(monitor_tab).classes("p-0 gap-4"):
                Monitor(output)

            with ui.tab_panel(results_tab).classes("p-0 gap-4"):
                results_panel = ui.refreshable(add_results)
                results_panel(output)

            with ui.tab_panel(predict_tab).classes("p-0 gap-4"):
                predict_panel = ui.refreshable(add_predict)
                predict_panel(output, devices)

            with ui.tab_panel(export_tab).classes("p-0 gap-4"):
                export_panel = ui.refreshable(add_export)
                export_panel(output)

        # the results and the weights appear while the page is open, so these tabs are drawn again when opened
        panels = {"results": results_panel, "predict": predict_panel, "export": export_panel}
        tabs.on_value_change(lambda e: panels[e.value].refresh() if e.value in panels else None)


def add_training_header(output: OutputDirectory, shown: dict, devices: list[tuple[str, str]], on_change: Callable[[TrainStatus], None]) -> None:
    """Add the name, state, config and model of a training with its stop or resume button.

    Args:
        output: output directory of the training.
        shown: status on the page.
        devices: (name, description) of every device.
        on_change: function called with the new status after a button.
    """
    name = output.path.name
    status = shown["status"]
    entry = find_config(name)
    config = read_config(output.config)
    device = f" · {status.device}" if status.device else ""
    summary = f"{summarize_model(config)} · {config['train']['dataset_dir']} · batch {config['train']['batch_size']}{device}"

    with ui.row().classes("w-full items-start justify-between no-wrap"):
        with ui.column().classes("gap-1"):
            with ui.row().classes("items-center gap-3"):
                ui.label(name).classes("text-2xl font-bold")
                add_pill(t(f"state.{status.state}"), f"state-{status.state}")

            with ui.row().classes("items-center gap-2 text-sm muted"):
                if entry is not None:
                    ui.link(entry.path.as_posix(), f"/configs/{name}").classes("mono text-xs")
                    ui.label("·")

                ui.label(summary)

        with ui.row().classes("gap-2"):
            add_train_actions(status, devices, lambda: on_change(read_status(output.path)), flat=False)


def add_results(output: OutputDirectory) -> None:
    """Add the epochs of the best weights and the report."""
    with ui.column().classes("card w-full gap-0"):
        with ui.row().classes("card-head w-full"):
            ui.label(t("results.best")).classes("section-title")

        if not output.result.exists():
            ui.label(t("results.no_result")).classes("text-sm muted p-5")
        else:
            result = json.loads(output.result.read_text())
            columns = "160px 100px 1fr 1fr 1fr"

            with ui.grid(columns=columns).classes("w-full gap-x-4 list-head"):
                for key in ["weights_file", "epoch", "val_loss", "map50", "map50_95"]:
                    ui.label(t(f"results.{key}") if key in ("weights_file", "epoch") else t(f"metric.{key}"))

            for key, record in result.items():
                weights = "best_loss.pt" if "loss" in key else "best_map.pt"

                with ui.grid(columns=columns).classes("w-full gap-x-4 list-row"):
                    ui.label(weights).classes("mono font-semibold")
                    ui.label(str(int(record["epoch"]))).classes("mono")

                    for metric in ["val_loss", "map50", "map50_95"]:
                        ui.label(f"{record[metric]:.3f}").classes("mono")

    with ui.column().classes("card w-full gap-0"):
        with ui.row().classes("card-head w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label(t("results.report")).classes("section-title")
                ui.label("report.txt").classes("caption mono")

            report_button = ui.button(t("action.rebuild"), icon="refresh").props("outline dense no-caps").classes("px-3")

        report = output.report.read_text(encoding="utf-8") if output.report.exists() else t("results.no_report")
        report_label = ui.label(report).classes("mono text-xs whitespace-pre overflow-x-auto p-5 w-full")

    async def rebuild_report() -> None:
        report_button.props("loading")
        report_path = await run.io_bound(make_report, output.path)
        report_label.set_text(report_path.read_text(encoding="utf-8"))
        report_button.props(remove="loading")

    report_button.on_click(rebuild_report)
