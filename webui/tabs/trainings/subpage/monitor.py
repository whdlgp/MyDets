from nicegui import ui

from shared.output_directory import OutputDirectory
from webui.bridge.processes import TrainStatus, parse_step, read_console, read_log, read_status
from webui.tabs.common.layout import find_accent
from webui.tabs.common.texts import t
from webui.tabs.common.widgets import CONFIG, Watcher
from webui.tabs.trainings.common import format_seconds


def make_chart_options(color: str) -> dict:
    """Make the options of an empty line chart, with gray axes that read in both modes and the accent color for the first line."""
    return {
        "color": [color, "#EA580C"],
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 0, "right": 0, "icon": "roundRect", "itemHeight": 3, "textStyle": {"color": "#8A909C"}},
        "grid": {"left": 44, "right": 12, "top": 30, "bottom": 28},
        "xAxis": {"type": "category", "data": [], "axisLine": {"lineStyle": {"color": "rgba(138, 144, 156, 0.5)"}}, "axisLabel": {"color": "#8A909C"}},
        "yAxis": {"type": "value", "scale": True, "splitLine": {"lineStyle": {"color": "rgba(138, 144, 156, 0.18)"}}, "axisLabel": {"color": "#8A909C"}},
        "series": []
    }


class Monitor:
    """Progress, last values, loss and mAP charts and console of a training, refreshed from log.csv and console.log."""

    def __init__(self, output: OutputDirectory) -> None:
        """Add the monitor and start refreshing it.

        Args:
            output: output directory of the training.
        """
        self.output = output

        with ui.row().classes("card w-full p-5 items-center gap-8 no-wrap"):
            with ui.column().classes("flex-1 gap-2"):
                with ui.row().classes("w-full justify-between"):
                    self.epoch_label = ui.label().classes("font-semibold")
                    self.step_label = ui.label().classes("text-sm muted")

                self.epoch_bar = ui.linear_progress(0, show_value=False, size="8px")

            self.time_labels = {}

            for key in ["elapsed", "remaining", "per_epoch"]:
                with ui.column().classes("gap-0"):
                    ui.label(t(f"monitor.{key}")).classes("caption")
                    self.time_labels[key] = ui.label("—").classes("mono font-semibold")

        self.value_labels = {}

        with ui.grid(columns=4).classes("w-full gap-4"):
            for key in ["train_loss", "val_loss", "map50", "map50_95"]:
                with ui.column().classes("card p-4 gap-1"):
                    ui.label(t(f"metric.{key}")).classes("caption font-semibold")
                    self.value_labels[key] = (ui.label("—").classes("stat-value"), ui.label().classes("text-xs muted"))

        self.empty_label = ui.label(t("monitor.no_log")).classes("text-sm muted")
        self.charts = {}

        with ui.grid(columns=2).classes("w-full gap-4"):
            for key, title in [("loss", t("chart.loss")), ("map", t("chart.map"))]:
                with ui.column().classes("card p-4 gap-0"):
                    ui.label(title).classes("section-title")
                    self.charts[key] = ui.echart(make_chart_options(find_accent())).classes("w-full h-64")

        with ui.column().classes("card w-full gap-0"):
            with ui.row().classes("card-head w-full items-center justify-between"):
                ui.label(t("tab.console")).classes("section-title")
                self.full_switch = ui.switch(t("console.full", lines=CONFIG["console_lines"])).props("dense").classes("text-sm muted")

            self.console = ui.label().classes("console mono w-full")

        initial = self.read()
        self.show(initial)
        Watcher(self.read, initial, self.show)
        self.full_switch.on_value_change(lambda: self.show(self.read()))

    def read(self) -> tuple[list[dict[str, float]], TrainStatus, list[str]]:
        """Read the log, the status and the console lines."""
        num_lines = CONFIG["console_lines"] if self.full_switch.value else CONFIG["console_tail"]
        lines = read_console(self.output.console, num_lines)
        return read_log(self.output), read_status(self.output.path), lines

    def show(self, values: tuple[list[dict[str, float]], TrainStatus, list[str]]) -> None:
        """Draw the values that read returned."""
        log, status, lines = values
        step = parse_step(lines) if status.state == "training" else None
        self.console.set_text("\n".join(lines) if lines else t("console.empty"))
        self.epoch_label.set_text(t("monitor.epoch", epoch=status.epoch, epochs=status.epochs))
        self.epoch_bar.set_value(status.epoch / status.epochs if status.epochs else 0)
        self.step_label.set_text(t("monitor.step", step=step[0], steps=step[1]) if step else "")
        self.empty_label.set_visibility(not log)

        if not log:
            return

        per_epoch = status.elapsed / len(log)
        remaining = format_seconds((status.epochs - status.epoch) * per_epoch) if status.state == "training" else "—"
        self.time_labels["elapsed"].set_text(format_seconds(status.elapsed))
        self.time_labels["remaining"].set_text(remaining)
        self.time_labels["per_epoch"].set_text(f"{per_epoch:.0f}s")

        for key, (value_label, note_label) in self.value_labels.items():
            value_label.set_text(f"{log[-1][key]:.3f}")

            # the loss cards show the parts of the last epoch, the map cards the best epoch
            if key.endswith("loss"):
                parts = [f"{k.removeprefix(key + '_')} {v:.2f}" for k, v in log[-1].items() if k.startswith(key + "_")]
                note_label.set_text(" · ".join(parts))
            else:
                best = max(log, key=lambda r: r[key])
                note_label.set_text(t("monitor.best", value=f"{best[key]:.3f}", epoch=int(best["epoch"])))

        epochs_axis = [int(r["epoch"]) for r in log]

        for chart_key, keys in [("loss", ["train_loss", "val_loss"]), ("map", ["map50", "map50_95"])]:
            chart = self.charts[chart_key]
            chart.options["xAxis"]["data"] = epochs_axis
            series = [{"type": "line", "name": key, "data": [round(r[key], 4) for r in log], "showSymbol": len(log) == 1, "lineStyle": {"width": 2}} for key in keys]
            chart.options["series"] = series
            chart.update()
