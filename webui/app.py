from collections.abc import Awaitable, Callable
from pathlib import Path
import asyncio
import base64
import inspect
import json
import re
import sys
import tempfile

# the web ui runs from the project root like the scripts, and imports their modules from there
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
print("Starting the MyDets web UI...", flush=True)

from nicegui import app, run, ui
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from predict import CONFIG as PREDICT_CONFIG
from shared.output_directory import OutputDirectory
from shared.settings import CommonSettings, PredictSettings, TrainSettings, read_config
from webui.configs import ConfigEntry, check_config, check_name, create_config, create_yaml, find_config, list_configs, load_config, save_config, write_config
from webui.datasets import DatasetInfo, draw_sample, list_datasets, list_downloads, read_dataset, start_download, stop_download
from webui.processes import RUNS_DIR, TrainStatus, export_onnx, find_process, list_busy_devices, list_devices, list_statuses, make_report, predict_images, read_console, read_log, read_status, run_onnx_tool, start_train, stop_process
from webui.schema import CONNECTING_ARGS, FieldInfo, describe_class, describe_settings, list_types
from webui.texts import TEXTS, get_language, t

CONFIG = {
    "port": 8080,
    "refresh_seconds": 2,
    "console_tail": 20,
    "console_lines": 300,
    "storage_secret": "mydets-webui"
}
STATES = ["training", "stopped", "done", "failed", "new"]
STATE_COLORS = {"training": "primary", "stopped": "warning", "done": "positive", "failed": "negative", "new": "grey-6"}
NAV = [("/", "nav.trainings", "model_training"), ("/configs", "nav.configs", "tune"), ("/datasets", "nav.datasets", "photo_library")]
SECTIONS = [("common", CommonSettings), ("train", TrainSettings), ("predict", PredictSettings)]
TRANSFORM_LISTS = {"train": ["transforms", "val_transforms"], "predict": ["transforms"]}
DATASET_LAYOUT = """data/<name>
├─ names.txt      class names, one per line
├─ train
│  ├─ images
│  ├─ labels      one .txt per image, a line per box: class_id cx cy w h, normalized
│  └─ ignores     optional, boxes to ignore, in the same form as labels
└─ val
   ├─ images
   ├─ labels
   └─ ignores"""
STYLE = """
<style>
body{background:var(--bg);color:var(--text);font-family:Inter,'Noto Sans KR','Malgun Gothic',system-ui,sans-serif}
.nicegui-content{padding:0}
.mono{font-family:'JetBrains Mono',Consolas,'Courier New',monospace}
.muted{color:var(--text-2)}
.faint{color:var(--text-3)}
.label-strong{color:var(--label)}
.text-link{color:var(--accent)}
.text-ok{color:var(--ok)}
.text-warn{color:var(--warn)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px}
.card-head{padding:14px 18px;border-bottom:1px solid var(--border-soft)}
.divider-top{border-top:1px solid var(--border-soft)}
.divider-bottom{border-bottom:1px solid var(--border-soft)}
.section-title{font-size:15px;font-weight:600}
.caption{font-size:12px;color:var(--text-2)}
.drawer{background:var(--surface)!important;border-right:1px solid var(--border)}
.logo{background:var(--accent)}
.navlink{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:8px;color:var(--label);text-decoration:none;font-size:14px;font-weight:500}
.navlink:hover{background:var(--hover);color:var(--text)}
.navlink.active{background:var(--accent-soft);color:var(--accent)}
.pill{justify-self:start;width:fit-content;display:inline-flex;align-items:center;gap:6px;height:22px;padding:0 9px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap;flex-wrap:nowrap}
.pill .dot{width:7px;height:7px;border-radius:50%;background:currentColor}
.state-training,.tag-blue,.dataset-downloading{background:var(--accent-soft);color:var(--accent)}
.state-stopped,.tag-orange,.dataset-incomplete{background:var(--warn-soft);color:var(--warn)}
.state-done,.dataset-ready{background:var(--ok-soft);color:var(--ok)}
.state-failed{background:var(--bad-soft);color:var(--bad)}
.state-new,.tag-grey,.dataset-missing{background:var(--grey-soft);color:var(--label)}
.banner-warn{background:var(--warn-soft);color:var(--warn)}
.banner-bad{background:var(--bad-soft);color:var(--bad)}
.list-head{font-size:12px;font-weight:600;color:var(--text-2);padding:10px 18px;border-bottom:1px solid var(--border)}
.list-row{padding:12px 18px;border-bottom:1px solid var(--border-soft);font-size:14px}
.list-row:last-child{border-bottom:none}
.clickable{cursor:pointer}
.clickable:hover{background:var(--hover)}
.chip{height:30px;padding:0 12px;border-radius:999px;border:1px solid var(--border-strong);background:var(--surface);font-size:13px;font-weight:500;cursor:pointer;display:flex;align-items:center;gap:6px}
.chip.active{background:var(--text);border-color:var(--text);color:var(--bg)}
.choice{border:1px solid var(--border);border-radius:10px;padding:10px 14px;cursor:pointer;background:var(--surface)}
.choice:hover{border-color:var(--accent)}
.choice.selected{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
.choice.disabled{cursor:not-allowed;opacity:.5;background:var(--surface-2)}
.choice.disabled:hover{border-color:var(--border)}
.add-tile{border:1.5px dashed var(--border-strong);border-radius:10px;color:var(--accent);font-weight:600;font-size:14px;padding:12px;cursor:pointer;justify-content:center;background:var(--surface)}
.add-tile:hover{border-color:var(--accent);background:var(--accent-soft)}
.box{border:1px solid var(--border);border-radius:10px;background:var(--surface-2)}
.block-kind{font-size:11px;font-weight:700;letter-spacing:.06em;color:var(--text-3);text-transform:uppercase}
.index-dot{background:var(--grey-soft)}
.img-frame{border:1px solid var(--border)}
.empty-frame{border:1.5px dashed var(--border-strong)}
.swatch{width:22px;height:22px;border-radius:50%;cursor:pointer;border:2px solid transparent}
.swatch.selected{border-color:var(--text)}
.console{background:var(--console);color:var(--console-text);border-radius:0 0 12px 12px;padding:14px 18px;font-size:12.5px;line-height:1.6;white-space:pre;overflow-x:auto}
.stat-value{font-size:26px;font-weight:600;font-family:'JetBrains Mono',Consolas,monospace}
.q-btn{text-transform:none;border-radius:8px;font-weight:500}
.q-btn.bg-primary{color:var(--accent-text)!important}
.q-btn-toggle{background:var(--surface);border:1px solid var(--border)}
.q-field--outlined .q-field__control{border-radius:8px;background:var(--surface)}
.q-field--outlined .q-field__control:before{border-color:var(--border-strong)}
.q-field--dense .q-field__control,.q-field--dense .q-field__marginal{height:34px}
.q-field--disabled .q-field__control{background:var(--surface-2)}
.q-tab{text-transform:none;font-weight:500;padding:0 14px;min-height:40px;color:var(--text-2)}
.q-tab--active{color:var(--text)}
.q-tab__indicator{background:var(--text)}
.q-tab__label{font-size:14px}
.q-linear-progress{border-radius:4px}
.q-linear-progress__track{background:var(--border)!important;opacity:1!important}
.q-tooltip{font-size:13px;max-width:320px;background:var(--text);color:var(--bg)}
.q-menu,.q-dialog .card{background:var(--surface);color:var(--text)}
.q-uploader{background:var(--surface);color:var(--text)}
.nicegui-code{background:var(--surface-2)!important}
.nicegui-markdown pre,.codehilite{background:var(--surface-2)!important}
</style>
"""
THEMES = json.loads((Path(__file__).parent / "themes.json").read_text(encoding="utf-8"))
MODES = {"light": False, "dark": True, "auto": None}


# common parts
class Watcher:
    """Reads a value in a worker thread at every refresh and calls on_change only when it changed.

    Reading files and processes off the event loop keeps clicks responsive, and drawing only on a change keeps buttons from being replaced under the pointer.
    """

    def __init__(self, read: Callable[[], object], initial: object, on_change: Callable[[object], object]) -> None:
        """Start reading.

        Args:
            read: function that returns the value.
            initial: value already drawn.
            on_change: function called with the new value, awaited when it is async.
        """
        self.read = read
        self.last = initial
        self.on_change = on_change
        ui.timer(CONFIG["refresh_seconds"], self.check)

    async def check(self) -> None:
        """Read the value and call on_change when it changed."""
        value = await run.io_bound(self.read)

        if value != self.last:
            self.last = value
            result = self.on_change(value)

            if inspect.isawaitable(result):
                await result


def add_frame(devices: list[tuple[str, str]], current: str) -> None:
    """Add the side menu with the two pages, the device list and the language choice.

    Args:
        devices: (name, description) of every device.
        current: path of the page in the menu that is open.
    """
    mode, accent = get_theme()
    ui.add_head_html(STYLE + make_theme_css(accent))
    ui.dark_mode(MODES[mode])

    with ui.left_drawer(value=True, fixed=True).classes("drawer p-3 flex flex-col gap-1").props("width=232"):
        with ui.row().classes("items-center gap-2 px-2 pt-2 pb-4"):
            ui.element("div").classes("w-6 h-6 rounded-md logo")
            ui.label("MyDets").classes("text-lg font-bold")

        for path, key, icon in NAV:
            with ui.link(target=path).classes(f"navlink {'active' if path == current else ''}"):
                ui.icon(icon, size="20px")
                ui.label(t(key))

        ui.space()

        shown = {"busy": list_busy_devices()}
        device_panel = ui.refreshable(add_device_panel)
        device_panel(devices, shown)
        Watcher(list_busy_devices, shown["busy"], lambda busy: (shown.update(busy=busy), device_panel.refresh()))

        with ui.row().classes("w-full items-center justify-between mt-2 no-wrap"):
            languages = {code: texts["language.name"] for code, texts in TEXTS.items()}
            ui.toggle(languages, value=get_language(), on_change=lambda e: set_language(e.value)).props("dense no-caps unelevated toggle-color=primary")
            add_theme_menu(mode, accent)


def add_device_panel(devices: list[tuple[str, str]], shown: dict) -> None:
    """Add the devices of the side menu, free or in use by a training."""
    busy = shown["busy"]

    with ui.column().classes("card w-full p-3 gap-2"):
        ui.label(t("devices.title")).classes("caption font-semibold")

        for name, description in devices:
            state = t("devices.in_use") if name in busy else t("devices.free")
            color = "text-link" if name in busy else "text-ok"
            detail = f"{description} · {busy[name]}" if name in busy else description

            with ui.column().classes("gap-0 w-full"):
                with ui.row().classes("w-full justify-between items-center no-wrap"):
                    ui.label(name).classes("text-sm font-semibold mono")
                    ui.label(state).classes(f"text-xs font-semibold {color}")

                ui.label(detail).classes("text-xs muted")


def get_theme() -> tuple[str, str]:
    """Return the mode and the accent color that the current browser chose, following the system in blue by default."""
    mode = app.storage.user.get("mode", "auto")
    accent = app.storage.user.get("accent", "blue")
    return mode if mode in MODES else "auto", accent if accent in THEMES["accents"] else "blue"


def make_theme_css(accent: str) -> str:
    """Write the colors of both modes with an accent color as css variables, quasar's brand colors included."""
    blocks = []

    for mode in ["light", "dark"]:
        colors = {**THEMES["modes"][mode], **THEMES["accents"][accent][mode]}
        brand = {"q-primary": colors["accent"], "q-positive": colors["ok"], "q-negative": colors["bad"], "q-warning": colors["warn"]}
        # quasar sets its brand colors inline on the body, so they need !important to change
        variables = "".join(f"--{key}:{value};" for key, value in colors.items()) + "".join(f"--{key}:{value}!important;" for key, value in brand.items())
        blocks.append(f"body.body--{mode}{{{variables}}}")

    return f"<style>{''.join(blocks)}</style>"


def find_accent() -> str:
    """Return the accent color for things drawn outside css, the light one unless dark mode was chosen."""
    mode, accent = get_theme()
    return THEMES["accents"][accent]["dark" if mode == "dark" else "light"]["accent"]


def add_theme_menu(mode: str, accent: str) -> None:
    """Add the button with the menu of the light, dark and system modes and the accent colors."""
    def choose(key: str, value: str) -> None:
        app.storage.user[key] = value
        ui.navigate.reload()

    with ui.button(icon="palette").props("flat round dense").tooltip(t("theme.title")):
        with ui.menu().classes("p-3"):
            ui.label(t("theme.mode")).classes("caption font-semibold pb-1")
            ui.toggle({key: t(f"theme.{key}") for key in MODES}, value=mode, on_change=lambda e: choose("mode", e.value)).props("dense no-caps unelevated toggle-color=primary")
            ui.label(t("theme.accent")).classes("caption font-semibold pt-3 pb-1")

            with ui.row().classes("gap-2"):
                for name, colors in THEMES["accents"].items():
                    swatch = ui.element("div").classes(f"swatch {'selected' if name == accent else ''}").style(f"background:{colors['light']['accent']}").tooltip(name)
                    swatch.on("click", lambda name=name: choose("accent", name))


def set_language(language: str) -> None:
    """Keep the language of this browser and redraw the page in it."""
    app.storage.user["language"] = language
    ui.navigate.reload()


def add_page_title(title: str, subtitle: str) -> ui.row:
    """Add the title of a page, returning the row on its right for the page buttons."""
    with ui.row().classes("w-full items-end justify-between no-wrap"):
        with ui.column().classes("gap-1"):
            ui.label(title).classes("text-2xl font-bold")
            ui.label(subtitle).classes("text-sm muted")

        return ui.row().classes("items-center gap-2 no-wrap")


def add_busy(text: str) -> ui.row:
    """Add a spinner with what is running, returning its row to remove when the job ends."""
    with ui.row().classes("items-center gap-2 text-sm muted py-2") as row:
        ui.spinner(size="sm")
        ui.label(text)

    return row


def add_pill(text: str, style: str) -> None:
    """Add a small rounded label, with a dot for the training states."""
    with ui.row().classes(f"pill {style}"):
        if style.startswith("state-"):
            ui.element("span").classes("dot")

        ui.label(text)


def format_seconds(seconds: float) -> str:
    """Write seconds as h:mm:ss."""
    seconds = round(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def describe_type(cls: type) -> str:
    """Return the first line of the docstring of a registered class."""
    return (inspect.getdoc(cls) or "").splitlines()[0] if inspect.getdoc(cls) else ""


def is_trained(entry: ConfigEntry) -> bool:
    """Return whether a config already has its training in runs/, also when it has just started."""
    output = OutputDirectory(RUNS_DIR / entry.name)
    return output.config.exists() or output.pid.exists()


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


# trainings
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


def summarize_model(config: dict) -> str:
    """Return the backbone, neck and head types of a config in one line."""
    return " · ".join(str(config["model"][block]["type"]) for block in ["backbone", "neck", "head"])


# training detail
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


def parse_step(lines: list[str]) -> tuple[int, int] | None:
    """Find the step and the steps of the last train progress bar in the console lines."""
    for line in reversed(lines):
        match = re.search(r"train:.*?(\d+)/(\d+) \[", line)

        if match:
            return int(match[1]), int(match[2])

    return None


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


def show_image(path: Path) -> None:
    """Show an image of runs/, with its modification time in the url so that a redrawn image replaces the cached one."""
    if path.exists():
        # a plain img keeps the height of the image, where ui.image keeps a fixed ratio
        source = f"/runs/{path.relative_to(RUNS_DIR).as_posix()}?v={path.stat().st_mtime_ns}"
        ui.element("img").props(f'src="{source}"').classes("w-full rounded-lg img-frame")
    else:
        with ui.column().classes("w-full h-48 items-center justify-center rounded-lg empty-frame"):
            ui.label(t("predict.no_images")).classes("text-sm faint")


def add_predict(output: OutputDirectory, devices: list[tuple[str, str]]) -> None:
    """Add the choice of weights and device, the val grid of each weights file and the prediction of one uploaded image."""
    weights = [name for name in PREDICT_CONFIG["weights"] if (output.path / name).exists()]

    if not weights:
        with ui.column().classes("card w-full p-5"):
            ui.label(t("predict.no_weights")).classes("text-sm muted")

        return

    busy = list_busy_devices()
    free = [name for name, _ in devices if name not in busy]

    with ui.row().classes("card w-full p-4 items-end gap-5"):
        with ui.column().classes("gap-1"):
            ui.label(t("predict.weights")).classes("caption font-semibold")

            with ui.row().classes("gap-3"):
                weight_boxes = {name: ui.checkbox(name, value=True).props("dense").classes("mono text-sm") for name in weights}

        num_images = ui.number(t("predict.num_images"), value=PREDICT_CONFIG["num_images"], min=1, step=1, format="%d").props("dense outlined").classes("w-28")
        seed = ui.number(t("predict.seed"), value=PREDICT_CONFIG["seed"], step=1, format="%d").props("dense outlined").classes("w-24")
        device_options = {name: f"{name}  {description}" for name, description in devices if name in free}
        device_select = ui.select(device_options, value=free[0] if free else None, label=t("predict.device")).props("dense outlined").classes("w-80")

    if not free:
        with ui.row().classes("w-full items-center gap-2 px-4 py-3 rounded-lg banner-warn"):
            ui.icon("lock", size="18px")
            ui.label(t("predict.locked")).classes("text-sm")

    def read_choices() -> tuple[list[str], int, int, str] | None:
        chosen = [name for name, box in weight_boxes.items() if box.value]

        if not chosen or device_select.value is None:
            ui.notify(t("predict.locked") if device_select.value is None else t("predict.choose_weights"), type="warning")
            return None

        return chosen, int(num_images.value or 1), int(seed.value or 0), device_select.value

    async def draw(input_path: Path, button: ui.button) -> list[Path]:
        choices = read_choices()

        if choices is None:
            return []

        button.props("loading")
        saved = await run.io_bound(predict_images, output.path, input_path, *choices)
        button.props(remove="loading")
        return saved

    with ui.column().classes("card w-full gap-0"):
        with ui.row().classes("card-head w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label(t("predict.grid")).classes("section-title")
                ui.label(t("predict.grid_hint")).classes("caption")

            with ui.row().classes("gap-2"):
                new_images_button = ui.button(t("action.new_images"), icon="shuffle").props("outline dense no-caps").classes("px-3")
                draw_button = ui.button(t("action.draw"), icon="grid_view").props("unelevated dense no-caps").classes("px-3")

        @ui.refreshable
        def grid_images() -> None:
            with ui.grid(columns=len(weights)).classes("w-full gap-4 p-5"):
                for name in weights:
                    with ui.column().classes("gap-1"):
                        show_image(output.predict / f"grid_{Path(name).stem}.jpg")
                        ui.label(name).classes("text-xs mono muted")

        grid_images()

    async def draw_grid() -> None:
        dataset_dir = Path(read_config(output.config)["train"]["dataset_dir"])
        await draw(dataset_dir / "val" / "images", draw_button)
        grid_images.refresh()

    async def draw_new_images() -> None:
        seed.set_value(int(seed.value or 0) + 1)
        await draw_grid()

    draw_button.on_click(draw_grid)
    new_images_button.on_click(draw_new_images)

    with ui.column().classes("card w-full gap-0"):
        with ui.row().classes("card-head w-full"):
            with ui.column().classes("gap-0"):
                ui.label(t("predict.single")).classes("section-title")
                ui.label(t("predict.single_hint")).classes("caption")

        with ui.row().classes("w-full gap-4 p-5 items-start"):
            upload_dir = Path(tempfile.mkdtemp(prefix="mydets_upload_"))
            single_images = ui.row().classes("flex-1 gap-4")

            async def predict_upload(event: object) -> None:
                input_path = upload_dir / event.file.name
                await event.file.save(input_path)
                single_images.clear()

                with single_images:
                    add_busy(t("predict.running", name=event.file.name))

                upload.disable()
                saved = await draw(input_path, upload)
                upload.enable()
                upload.reset()
                single_images.clear()

                with single_images:
                    for path in saved:
                        with ui.column().classes("gap-1 w-80"):
                            show_image(path)
                            ui.label(path.name).classes("text-xs mono muted")

            upload = ui.upload(label=t("predict.upload"), auto_upload=True, on_upload=predict_upload).props("accept=image/* flat bordered color=grey-8").classes("w-72 order-first")


# onnx export
def add_export(output: OutputDirectory) -> None:
    """Add the onnx export of a best weights file, and the verify and inference of an exported file on an uploaded image."""
    weights = [name for name in PREDICT_CONFIG["weights"] if (output.path / name).exists()]

    if not weights:
        with ui.column().classes("card w-full p-5"):
            ui.label(t("predict.no_weights")).classes("text-sm muted")

        return

    width, height = read_config(output.config)["common"]["image_size"]

    with ui.column().classes("card w-full gap-0") as export_card:
        with ui.row().classes("card-head w-full"):
            with ui.column().classes("gap-0"):
                ui.label(t("export.title")).classes("section-title")
                ui.label(t("export.hint")).classes("caption")

        with ui.row().classes("w-full p-4 items-end gap-5"):
            weights_select = ui.select(weights, value=weights[-1], label=t("export.weights")).props("dense outlined").classes("w-40")
            size_mode = ui.toggle({"fixed": t("export.fixed"), "dynamic": t("export.dynamic")}, value="fixed").props("dense no-caps unelevated toggle-color=primary")
            width_input = ui.number(t("export.width"), value=width, min=32, step=32, format="%d").props("dense outlined").classes("w-24")
            height_input = ui.number(t("export.height"), value=height, min=32, step=32, format="%d").props("dense outlined").classes("w-24")
            opset_input = ui.number("opset", value=19, min=16, step=1, format="%d").props("dense outlined").classes("w-24")
            simplify_box = ui.checkbox(t("export.simplify")).props("dense")
            export_button = ui.button(t("action.export"), icon="file_download").props("unelevated dense no-caps").classes("px-3")

        export_log = ui.label().classes("console mono w-full")
        export_log.set_visibility(False)

    size_mode.on_value_change(lambda e: (width_input.set_enabled(e.value == "fixed"), height_input.set_enabled(e.value == "fixed")))

    with ui.column().classes("card w-full gap-0"):
        with ui.row().classes("card-head w-full"):
            with ui.column().classes("gap-0"):
                ui.label(t("export.files")).classes("section-title")
                ui.label(t("export.files_hint")).classes("caption")

        with ui.row().classes("w-full p-4 items-start gap-5"):
            files = {str(path): path.name for path in sorted(output.export.glob("*.onnx"))}
            file_select = ui.select(files, value=next(iter(files), None), label=t("export.file")).props("dense outlined").classes("w-80")
            tool_toggle = ui.toggle({"verify": t("export.verify"), "inference": t("export.inference")}, value="verify").props("dense no-caps unelevated toggle-color=primary")
            upload_dir = Path(tempfile.mkdtemp(prefix="mydets_upload_"))
            upload = ui.upload(label=t("export.upload"), auto_upload=True).props("accept=image/* flat bordered color=grey-8").classes("w-72")

        result = ui.column().classes("w-full gap-3 px-5 pb-5")

    async def export() -> None:
        image_size = (int(width_input.value), int(height_input.value)) if size_mode.value == "fixed" else None
        export_button.props("loading")
        export_log.set_visibility(False)

        with export_card:
            busy = add_busy(t("export.exporting", name=weights_select.value))

        succeeded, text = await run.io_bound(export_onnx, output.path, weights_select.value, image_size, int(opset_input.value), simplify_box.value)
        busy.delete()
        export_button.props(remove="loading")
        export_log.set_text(text)
        export_log.set_visibility(True)
        ui.notify(t("notify.exported") if succeeded else t("notify.export_failed"), type="positive" if succeeded else "negative")
        exported = {str(path): path.name for path in sorted(output.export.glob("*.onnx"))}
        file_select.set_options(exported, value=file_select.value or next(iter(exported), None))

    async def run_tool(event: object) -> None:
        if file_select.value is None:
            ui.notify(t("export.choose_file"), type="warning")
            upload.reset()
            return

        image_path = upload_dir / event.file.name
        await event.file.save(image_path)
        result.clear()

        with result:
            add_busy(t("export.running", tool=tool_toggle.value, name=event.file.name))

        upload.disable()
        succeeded, text, saved = await run.io_bound(run_onnx_tool, tool_toggle.value, output.path, Path(file_select.value), image_path)
        upload.enable()
        upload.reset()
        result.clear()

        with result:
            if succeeded or saved.exists():
                show_image(saved)

            ui.label(text).classes("mono text-xs whitespace-pre overflow-x-auto w-full")

    export_button.on_click(export)
    upload.on_upload(run_tool)


# configs
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


# config editor
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


# datasets
@ui.page("/datasets")
async def datasets_page() -> None:
    """Show the datasets in data/, with the download button of the ones that a script can make."""
    add_frame(list_devices(), "/datasets")

    # counting the files of a large dataset takes a while, so it is done on opening, on the button and after a download
    async def reload() -> None:
        list_area.clear()

        with list_area:
            add_busy(t("datasets.loading"))

        datasets = await run.io_bound(list_datasets)
        users = await run.io_bound(find_dataset_users)
        list_area.clear()

        with list_area:
            add_dataset_list(datasets, users, reload)

    with ui.column().classes("w-full max-w-[1280px] px-8 py-7 gap-5") as page:
        with add_page_title(t("datasets.title"), t("datasets.subtitle")):
            ui.button(t("action.refresh"), icon="refresh", on_click=reload).props("outline no-caps")

        list_area = ui.column().classes("w-full")

        with ui.column().classes("card w-full gap-0"):
            with ui.row().classes("card-head w-full items-baseline gap-3"):
                ui.label(t("datasets.layout")).classes("section-title")
                ui.label(t("datasets.layout_hint")).classes("caption")

            ui.label(DATASET_LAYOUT).classes("mono text-xs whitespace-pre p-5")

    await ui.context.client.connected()
    await reload()

    # only the running downloads are checked every refresh, and a finished one reads the datasets again
    with page:
        Watcher(list_downloads, await run.io_bound(list_downloads), lambda _: reload())


def add_dataset_list(datasets: list[DatasetInfo], users: dict[str, list[str]], reload: Callable[[], Awaitable[None]]) -> None:
    """Add a row per dataset, which opens the dataset on a click.

    Args:
        datasets: datasets to show.
        users: configs that train on each dataset directory, by its path.
        reload: function that reads the datasets again, after a download button.
    """
    columns = "110px minmax(180px, 2fr) 80px 110px 110px minmax(160px, 2fr) 130px"

    with ui.column().classes("card w-full gap-0 overflow-x-auto"):
        with ui.grid(columns=columns).classes("w-full gap-x-4 list-head"):
            for key in ["status", "name", "classes", "train", "val", "used_by", "actions"]:
                ui.label(t(f"datasets.column.{key}"))

        if not datasets:
            ui.label(t("datasets.empty")).classes("text-sm muted p-5")

        for info in datasets:
            row = ui.grid(columns=columns).classes("w-full gap-x-4 items-center list-row clickable")
            row.on("click", lambda name=info.name: ui.navigate.to(f"/datasets/{name}"))
            used_by = ", ".join(users.get(info.path.as_posix(), [])) or "—"

            with row:
                add_pill(t(f"datasets.state.{info.state}"), f"dataset-{info.state}")

                with ui.column().classes("gap-0 min-w-0"):
                    ui.label(info.name).classes("font-semibold text-link truncate")
                    ui.label(info.path.as_posix()).classes("text-xs mono muted")

                ui.label(str(len(info.classes)) if info.classes else "—").classes("mono")

                for split in info.splits:
                    ui.label(f"{split.images:,}" if split.images else "—").classes("mono")

                ui.label(used_by).classes("text-sm muted truncate")

                with ui.row().classes("gap-1 no-wrap").on("click.stop", lambda: None):
                    add_download_actions(info, reload)


def find_dataset_users() -> dict[str, list[str]]:
    """Return the configs that train on each dataset directory, by its path."""
    users = {}

    for entry in list_configs():
        dataset_dir = Path(read_config(entry.path)["train"]["dataset_dir"]).as_posix()
        users.setdefault(dataset_dir, []).append(entry.name)

    return users


def add_download_actions(info: DatasetInfo, on_done: Callable[[], Awaitable[None]]) -> None:
    """Add the download button of a dataset that a script can make and that is not ready, or the stop button of a running download."""
    if info.state == "downloading":
        async def stop() -> None:
            stop_download(info.name)
            await on_done()

        ui.button(t("action.stop"), icon="stop", on_click=stop, color="negative").props("flat dense no-caps").classes("px-3")
    elif info.script is not None and info.state != "ready":
        async def download() -> None:
            start_download(info.name)
            ui.notify(t("notify.download_started", name=info.name), type="positive")
            await on_done()

        ui.button(t("action.download"), icon="download", on_click=download).props("flat dense no-caps").classes("px-3")


def make_tree(info: DatasetInfo) -> list[tuple[str, str, bool]]:
    """Make the lines of the folder tree of a dataset as (path, note, problem)."""
    lines = [(info.path.as_posix(), "", False)]
    lines.append(("├─ names.txt", t("datasets.classes_count", count=len(info.classes)) if info.classes else t("datasets.missing_file"), not info.classes))

    for i, split in enumerate(info.splits):
        last = i == len(info.splits) - 1
        branch, indent = ("└─", "   ") if last else ("├─", "│  ")
        folders = [("images", split.images), ("labels", split.labels)] + ([("ignores", split.ignores)] if split.ignores else [])
        lines.append((f"{branch} {split.name}", "", False))

        for j, (folder, count) in enumerate(folders):
            sub_branch = "└─" if j == len(folders) - 1 else "├─"
            problem = count == 0 and folder != "ignores"
            note = f"{count:,}" if count else t("datasets.missing_folder")

            if folder == "images" and split.unlabeled:
                note, problem = f"{count:,}  ·  {t('datasets.unlabeled', count=split.unlabeled)}", True

            lines.append((f"{indent}{sub_branch} {folder}", note, problem))

    return lines


@ui.page("/datasets/{name}")
async def dataset_page(name: str) -> None:
    """Show one dataset: its folder tree, its classes and one drawn image of each split."""
    add_frame(list_devices(), "/datasets")
    shown = {"seed": 0}

    # the files are counted on opening, on the button and after a download, as on the datasets page
    async def reload() -> None:
        content_area.clear()

        with content_area:
            add_busy(t("datasets.loading"))

        info = await run.io_bound(read_dataset, name)
        content_area.clear()

        with content_area:
            add_dataset_content(info, shown, reload)

    with ui.column().classes("w-full max-w-[1280px] px-8 py-7 gap-4") as page:
        with ui.row().classes("w-full items-center justify-between"):
            ui.link(t("action.back_datasets"), "/datasets").classes("text-sm muted no-underline")
            ui.button(t("action.refresh"), icon="refresh", on_click=reload).props("outline dense no-caps").classes("px-3")

        content_area = ui.column().classes("w-full gap-4")

    await ui.context.client.connected()
    await reload()

    with page:
        Watcher(lambda: name in list_downloads(), name in await run.io_bound(list_downloads), lambda _: reload())


def add_dataset_content(info: DatasetInfo, shown: dict, reload: Callable[[], Awaitable[None]]) -> None:
    """Add the name, state, folder tree, classes and samples of a dataset.

    Args:
        info: dataset to show.
        shown: seed of the samples, kept between reloads.
        reload: function that reads the dataset again, after a download button.
    """
    script = f"  ·  {info.script.as_posix()}" if info.script else ""

    with ui.row().classes("w-full items-start justify-between no-wrap"):
        with ui.column().classes("gap-1"):
            with ui.row().classes("items-center gap-3"):
                ui.label(info.name).classes("text-2xl font-bold")
                add_pill(t(f"datasets.state.{info.state}"), f"dataset-{info.state}")

            ui.label(info.path.as_posix() + script).classes("text-xs mono muted")

        with ui.row().classes("gap-2"):
            add_download_actions(info, reload)

    if info.state == "downloading" or (info.path / "download.log").exists():
        add_download_console(info)

    if info.state == "missing":
        return

    with ui.row().classes("w-full gap-4 items-stretch no-wrap"):
        with ui.column().classes("card gap-0 flex-1 min-w-0"):
            with ui.row().classes("card-head w-full"):
                ui.label(t("datasets.structure")).classes("section-title")

            with ui.column().classes("gap-0 p-5 mono text-sm"):
                for path, note, problem in make_tree(info):
                    with ui.row().classes("gap-0 no-wrap"):
                        ui.label(path).classes("whitespace-pre w-48")
                        ui.label(note).classes(f"whitespace-pre {'text-warn' if problem else 'muted'}")

        with ui.column().classes("card gap-0 flex-1 min-w-0"):
            with ui.row().classes("card-head w-full"):
                ui.label(t("datasets.classes")).classes("section-title")

            with ui.row().classes("gap-2 p-5"):
                for i, class_name in enumerate(info.classes or []):
                    with ui.row().classes("pill tag-grey"):
                        ui.label(str(i)).classes("opacity-60 mono")
                        ui.label(class_name)

    if info.state != "downloading":
        add_samples(info, shown)


def add_download_console(info: DatasetInfo) -> None:
    """Add the last lines of the download log, refreshed while the download runs."""
    log_path = info.path / "download.log"

    with ui.column().classes("card w-full gap-0"):
        with ui.row().classes("card-head w-full items-baseline gap-3"):
            ui.label(t("datasets.download_log")).classes("section-title")
            ui.label(log_path.as_posix()).classes("caption mono")

        console = ui.label().classes("console mono w-full")

    def show(lines: list[str]) -> None:
        console.set_text("\n".join(lines) if lines else t("console.empty"))

    initial = read_console(log_path, CONFIG["console_tail"])
    show(initial)

    if info.state == "downloading":
        Watcher(lambda: read_console(log_path, CONFIG["console_tail"]), initial, show)


def add_samples(info: DatasetInfo, shown: dict) -> None:
    """Add one random image of each split with its labels drawn, and a button for other images."""
    with ui.column().classes("card w-full gap-0"):
        with ui.row().classes("card-head w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label(t("datasets.samples")).classes("section-title")
                ui.label(t("datasets.samples_hint")).classes("caption")

            other_button = ui.button(t("action.other_images"), icon="shuffle").props("outline dense no-caps").classes("px-3")

        samples = ui.grid(columns=len(info.splits)).classes("w-full gap-4 p-5")

    async def draw_other() -> None:
        shown["seed"] += 1
        await draw_samples(info, shown["seed"], samples, other_button)

    other_button.on_click(draw_other)
    ui.timer(0.1, lambda: draw_samples(info, shown["seed"], samples, other_button), once=True)


async def draw_samples(info: DatasetInfo, seed: int, samples: ui.grid, button: ui.button) -> None:
    """Draw one image of each split with its labels into the samples grid, or the problem of its label file.

    Args:
        info: dataset to draw.
        seed: seed of the random images.
        samples: grid that gets a column per split.
        button: button that shows loading while drawing.
    """
    button.props("loading")
    samples.clear()

    for split in info.splits:
        with samples:
            column = ui.column().classes("gap-1 min-w-0")

        # a broken label file is what this page is for, so it is shown instead of raised
        try:
            sample = await run.io_bound(draw_sample, info, split.name, seed)
        except ValueError as error:
            sample, problem = None, str(error)
        else:
            problem = None if sample else t("datasets.no_images")

        with column:
            ui.label(split.name).classes("caption font-semibold")

            if sample:
                image, caption = sample
                source = f"data:image/jpeg;base64,{base64.b64encode(image).decode()}"
                ui.element("img").props(f'src="{source}"').classes("w-full rounded-lg img-frame")
                ui.label(caption).classes("text-xs mono muted")
            else:
                with ui.column().classes("w-full h-48 items-center justify-center rounded-lg px-4 empty-frame"):
                    ui.label(problem).classes("text-sm mono text-center text-warn")

    button.props(remove="loading")


def ignore_connection_reset(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """Keep the windows event loop from printing a traceback whenever a browser closes a connection."""
    if not isinstance(context.get("exception"), ConnectionResetError):
        loop.default_exception_handler(context)


def main():
    """Serve the web ui to train, monitor and check the models of this project.

    Usage: uv run webui\\app.py
    """
    RUNS_DIR.mkdir(exist_ok=True)
    app.add_media_files("/runs", RUNS_DIR)
    app.on_startup(lambda: asyncio.get_running_loop().set_exception_handler(ignore_connection_reset))
    ui.run(title="MyDets", port=CONFIG["port"], storage_secret=CONFIG["storage_secret"], reload=False, show=True)


if __name__ in {"__main__", "__mp_main__"}:
    main()
