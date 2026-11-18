from pathlib import Path
import asyncio
import base64
import inspect
import json
import os
import re
import sys
import tempfile

# the web ui runs from the project root, so that it finds the configs, runs and modules like the scripts do
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
print("Starting the MyDets web UI...", flush=True)

from nicegui import app, run, ui
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from predict import CONFIG as PREDICT_CONFIG
from shared.output_directory import OutputDirectory
from shared.settings import CommonSettings, PredictSettings, TrainSettings, read_config
from webui.configs import ConfigEntry, check_config, check_name, create_config, create_yaml, find_config, list_configs, load_config, save_config, write_config
from webui.datasets import DatasetInfo, draw_sample, list_datasets, read_dataset, start_download, stop_download
from webui.processes import TrainStatus, find_process, list_busy_devices, list_devices, list_statuses, make_report, predict_images, read_console, read_log, read_status, start_train, stop_process
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
def watch(read, initial: object, on_change) -> None:
    """Read a value in a worker thread at every refresh and call on_change only when it changed.

    Reading files and processes off the event loop keeps clicks responsive, and drawing only on a change keeps buttons from being replaced under the pointer.

    Args:
        read: function that returns the value.
        initial: value already drawn.
        on_change: function called with the new value.
    """
    last = {"value": initial}

    async def check() -> None:
        value = await run.io_bound(read)

        if value != last["value"]:
            last["value"] = value
            on_change(value)

    ui.timer(CONFIG["refresh_seconds"], check)


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

        @ui.refreshable
        def device_panel() -> None:
            busy = shown["busy"]

            with ui.column().classes("card w-full p-3 gap-2"):
                ui.label(t("devices.title")).classes("caption font-semibold")

                for name, description in devices:
                    with ui.column().classes("gap-0 w-full"):
                        with ui.row().classes("w-full justify-between items-center no-wrap"):
                            ui.label(name).classes("text-sm font-semibold mono")
                            ui.label(t("devices.in_use") if name in busy else t("devices.free")).classes(f"text-xs font-semibold {'text-link' if name in busy else 'text-ok'}")

                        ui.label(f"{description} · {busy[name]}" if name in busy else description).classes("text-xs muted")

        device_panel()
        watch(list_busy_devices, shown["busy"], lambda busy: (shown.update(busy=busy), device_panel.refresh()))

        with ui.row().classes("w-full items-center justify-between mt-2 no-wrap"):
            languages = {code: texts["language.name"] for code, texts in TEXTS.items()}
            ui.toggle(languages, value=get_language(), on_change=lambda e: set_language(e.value)).props("dense no-caps unelevated toggle-color=primary")
            add_theme_menu(mode, accent)


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
    output = OutputDirectory(Path("runs") / entry.name)
    return output.config.exists() or output.pid.exists()


def add_device_menu(devices: list[tuple[str, str]], label: str, on_choose, props: str = "unelevated") -> None:
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


def resume(name: str, device: str, on_done) -> None:
    """Start a stopped or failed training again from its last.pt on a device."""
    entry = find_config(name)

    if entry is None:
        ui.notify(t("notify.no_config", name=name), type="warning")
        return

    start_train(entry.path, device)
    ui.notify(t("notify.started", name=name, device=device), type="positive")
    on_done()


def add_train_actions(status: TrainStatus, devices: list[tuple[str, str]], on_done, flat: bool) -> None:
    """Add the stop button of a running training or the resume menu of a stopped or failed one."""
    if status.state == "training":
        def stop() -> None:
            stop_process(OutputDirectory(Path("runs") / status.name).pid)
            ui.notify(t("notify.stopped", name=status.name))
            on_done()

        ui.button(t("action.stop"), icon="stop", on_click=stop, color="negative").props(f"{'flat' if flat else 'outline'} dense no-caps").classes("px-3")
    elif status.state in ("stopped", "failed", "new"):
        # a training stopped before its first epoch has no last.pt, so it starts over
        label = t("action.start") if status.state == "new" else t("action.resume")
        add_device_menu(devices, label, lambda device: resume(status.name, device, on_done), "flat" if flat else "outline")


# trainings
def add_trainings_page(devices: list[tuple[str, str]]) -> None:
    """Add the first page: every training in runs/, with the button that adds a new one."""
    @ui.page("/")
    def trainings_page() -> None:
        add_frame(devices, "/")
        shown = {"state": "all", "statuses": list_statuses()}

        def update(statuses: list[TrainStatus]) -> None:
            shown["statuses"] = statuses
            training_list.refresh()

        with ui.column().classes("w-full max-w-[1280px] px-8 py-7 gap-5"):
            with add_page_title(t("trainings.title"), t("trainings.subtitle")):
                ui.button(t("action.add"), icon="add", on_click=lambda: open_add_dialog(devices)).props("unelevated no-caps")

            @ui.refreshable
            def training_list() -> None:
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
                        chip.on("click", lambda state=state: (shown.update(state=state), training_list.refresh()))

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

                        with row:
                            add_pill(t(f"state.{status.state}"), f"state-{status.state}")
                            ui.label(status.name).classes("font-semibold text-link truncate")
                            ui.label(status.model).classes("text-sm muted truncate")
                            ui.label(status.device or "—").classes("mono text-xs")

                            with ui.row().classes("items-center gap-2 no-wrap"):
                                ui.linear_progress(status.epoch / status.epochs if status.epochs else 0, show_value=False, size="6px").props(f"color={STATE_COLORS[status.state]}").classes("w-20")
                                ui.label(f"{status.epoch} / {status.epochs}").classes("text-xs muted")

                            ui.label(f"{status.best_map:.3f}" if status.best_map is not None else "—").classes("mono")
                            ui.label(format_seconds(status.elapsed)).classes("text-sm muted")

                            # the buttons must not open the training as the row click does
                            with ui.row().classes("gap-1 no-wrap").on("click.stop", lambda: None):
                                add_train_actions(status, devices, lambda: update(list_statuses()), flat=True)

            training_list()
            watch(list_statuses, shown["statuses"], update)


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
def add_training_page(devices: list[tuple[str, str]]) -> None:
    """Add the page of one training, with the monitor, results and predict tabs."""
    @ui.page("/train/{name}")
    def training_page(name: str) -> None:
        add_frame(devices, "/")
        output = OutputDirectory(Path("runs") / name)

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

            def update(status: TrainStatus) -> None:
                shown["status"] = status
                header.refresh()

            @ui.refreshable
            def header() -> None:
                status = shown["status"]
                entry = find_config(name)
                config = read_config(output.config)

                with ui.row().classes("w-full items-start justify-between no-wrap"):
                    with ui.column().classes("gap-1"):
                        with ui.row().classes("items-center gap-3"):
                            ui.label(name).classes("text-2xl font-bold")
                            add_pill(t(f"state.{status.state}"), f"state-{status.state}")

                        with ui.row().classes("items-center gap-2 text-sm muted"):
                            if entry is not None:
                                ui.link(entry.path.as_posix(), f"/configs/{name}").classes("mono text-xs")
                                ui.label("·")

                            ui.label(f"{summarize_model(config)} · {config['train']['dataset_dir']} · batch {config['train']['batch_size']}" + (f" · {status.device}" if status.device else ""))

                    with ui.row().classes("gap-2"):
                        add_train_actions(status, devices, lambda: update(read_status(output.path)), flat=False)

            header()
            watch(lambda: read_status(output.path), shown["status"], update)

            with ui.tabs().classes("w-full divider-bottom").props("align=left no-caps dense") as tabs:
                monitor_tab = ui.tab("monitor", t("tab.monitor"))
                results_tab = ui.tab("results", t("tab.results"))
                predict_tab = ui.tab("predict", t("tab.predict"))

            with ui.tab_panels(tabs, value=monitor_tab).classes("w-full bg-transparent").props("animated=false"):
                with ui.tab_panel(monitor_tab).classes("p-0 gap-4"):
                    add_monitor(output)

                with ui.tab_panel(results_tab).classes("p-0 gap-4"):
                    results_panel = ui.refreshable(add_results)
                    results_panel(output)

                with ui.tab_panel(predict_tab).classes("p-0 gap-4"):
                    predict_panel = ui.refreshable(add_predict)
                    predict_panel(output, devices)

            # the results and the weights appear while the page is open, so these tabs are drawn again when opened
            tabs.on_value_change(lambda e: results_panel.refresh() if e.value == "results" else predict_panel.refresh() if e.value == "predict" else None)


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


def add_monitor(output: OutputDirectory) -> None:
    """Add the progress, the last values, the loss and mAP charts and the console, refreshed from log.csv and console.log."""
    with ui.row().classes("card w-full p-5 items-center gap-8 no-wrap"):
        with ui.column().classes("flex-1 gap-2"):
            with ui.row().classes("w-full justify-between"):
                epoch_label = ui.label().classes("font-semibold")
                step_label = ui.label().classes("text-sm muted")

            epoch_bar = ui.linear_progress(0, show_value=False, size="8px")

        time_labels = {}

        for key in ["elapsed", "remaining", "per_epoch"]:
            with ui.column().classes("gap-0"):
                ui.label(t(f"monitor.{key}")).classes("caption")
                time_labels[key] = ui.label("—").classes("mono font-semibold")

    value_labels = {}

    with ui.grid(columns=4).classes("w-full gap-4"):
        for key in ["train_loss", "val_loss", "map50", "map50_95"]:
            with ui.column().classes("card p-4 gap-1"):
                ui.label(t(f"metric.{key}")).classes("caption font-semibold")
                value_labels[key] = (ui.label("—").classes("stat-value"), ui.label().classes("text-xs muted"))

    empty_label = ui.label(t("monitor.no_log")).classes("text-sm muted")

    with ui.grid(columns=2).classes("w-full gap-4"):
        charts = {}

        for key, title in [("loss", t("chart.loss")), ("map", t("chart.map"))]:
            with ui.column().classes("card p-4 gap-0"):
                ui.label(title).classes("section-title")
                charts[key] = ui.echart(make_chart_options(find_accent())).classes("w-full h-64")

    with ui.column().classes("card w-full gap-0"):
        with ui.row().classes("card-head w-full items-center justify-between"):
            ui.label(t("tab.console")).classes("section-title")
            full_switch = ui.switch(t("console.full", lines=CONFIG["console_lines"])).props("dense").classes("text-sm muted")

        console = ui.label().classes("console mono w-full")

    def read() -> tuple[list[dict[str, float]], TrainStatus, list[str]]:
        lines = read_console(output.console, CONFIG["console_lines"] if full_switch.value else CONFIG["console_tail"])
        return read_log(output), read_status(output.path), lines

    def show(values: tuple[list[dict[str, float]], TrainStatus, list[str]]) -> None:
        log, status, lines = values
        step = parse_step(lines) if status.state == "training" else None
        console.set_text("\n".join(lines) if lines else t("console.empty"))
        epoch_label.set_text(t("monitor.epoch", epoch=status.epoch, epochs=status.epochs))
        epoch_bar.set_value(status.epoch / status.epochs if status.epochs else 0)
        step_label.set_text(t("monitor.step", step=step[0], steps=step[1]) if step else "")
        empty_label.set_visibility(not log)

        if not log:
            return

        per_epoch = status.elapsed / len(log)
        time_labels["elapsed"].set_text(format_seconds(status.elapsed))
        time_labels["remaining"].set_text(format_seconds((status.epochs - status.epoch) * per_epoch) if status.state == "training" else "—")
        time_labels["per_epoch"].set_text(f"{per_epoch:.0f}s")

        for key, (value_label, note_label) in value_labels.items():
            best = min(log, key=lambda r: r[key]) if key.endswith("loss") else max(log, key=lambda r: r[key])
            value_label.set_text(f"{log[-1][key]:.3f}")

            # the loss cards show the parts of the last epoch, the map cards the best epoch
            parts = [f"{k.removeprefix(key + '_')} {v:.2f}" for k, v in log[-1].items() if k.startswith(key + "_")]
            note_label.set_text(" · ".join(parts) if key.endswith("loss") else t("monitor.best", value=f"{best[key]:.3f}", epoch=int(best["epoch"])))

        epochs_axis = [int(r["epoch"]) for r in log]

        for chart_key, keys in [("loss", ["train_loss", "val_loss"]), ("map", ["map50", "map50_95"])]:
            charts[chart_key].options["xAxis"]["data"] = epochs_axis
            charts[chart_key].options["series"] = [{"type": "line", "name": key, "data": [round(r[key], 4) for r in log], "showSymbol": len(log) == 1, "lineStyle": {"width": 2}} for key in keys]
            charts[chart_key].update()

    initial = read()
    show(initial)
    watch(read, initial, show)
    full_switch.on_value_change(lambda: show(read()))


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

        report_label = ui.label(output.report.read_text(encoding="utf-8") if output.report.exists() else t("results.no_report")).classes("mono text-xs whitespace-pre overflow-x-auto p-5 w-full")

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
        ui.element("img").props(f'src="/runs/{path.relative_to("runs").as_posix()}?v={path.stat().st_mtime_ns}"').classes("w-full rounded-lg img-frame")
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
        device_select = ui.select({name: f"{name}  {description}" for name, description in devices if name in free}, value=free[0] if free else None, label=t("predict.device")).props("dense outlined").classes("w-80")

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

            async def predict_upload(event) -> None:
                input_path = upload_dir / event.file.name
                await event.file.save(input_path)
                saved = await draw(input_path, upload)
                upload.reset()
                single_images.clear()

                with single_images:
                    for path in saved:
                        with ui.column().classes("gap-1 w-80"):
                            show_image(path)
                            ui.label(path.name).classes("text-xs mono muted")

            upload = ui.upload(label=t("predict.upload"), auto_upload=True, on_upload=predict_upload).props("accept=image/* flat bordered color=grey-8").classes("w-72 order-first")


# configs
def add_configs_page(devices: list[tuple[str, str]]) -> None:
    """Add the page of the user configs and the presets, where new configs are made from a copy."""
    @ui.page("/configs")
    def configs_page() -> None:
        add_frame(devices, "/configs")
        entries = list_configs()

        with ui.column().classes("w-full max-w-[1280px] px-8 py-7 gap-5"):
            with add_page_title(t("configs.title"), t("configs.subtitle")):
                ui.button(t("action.new_config"), icon="add", on_click=lambda: open_new_config_dialog(entries, None)).props("unelevated no-caps")

            mine = [e for e in entries if not e.preset]

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
            ui.button(t("action.view") if entry.locked else t("action.edit"), on_click=lambda: ui.navigate.to(f"/configs/{entry.name}")).props("flat dense no-caps").classes("px-2")
            ui.button(t("action.copy"), on_click=lambda: open_new_config_dialog(entries, entry)).props("flat dense no-caps").classes("px-2")


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
            name_input = ui.input(t("configs.name"), validation=lambda value: t(check_name(value)) if value and check_name(value) else None).props("outlined dense").classes("w-full")
            ui.label().bind_text_from(name_input, "value", backward=lambda value: f"configs/user/{value}.yaml → runs/{value}/" if value else "").classes("text-xs mono muted")

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
def add_editor_page(devices: list[tuple[str, str]]) -> None:
    """Add the page that builds a config with blocks and fields made from the code, or edits it as text."""
    @ui.page("/configs/{name}")
    def editor_page(name: str) -> None:
        add_frame(devices, "/configs")
        entry = find_config(name)

        with ui.column().classes("w-full max-w-[1400px] px-8 py-7 gap-4"):
            ui.link(t("action.back_configs"), "/configs").classes("text-sm muted no-underline")

            if entry is None:
                ui.label(t("configs.missing", name=name)).classes("muted")
                return

            config = load_config(entry.path)
            readonly = entry.locked
            state = {"dirty": False}

            with ui.row().classes("w-full items-start justify-between no-wrap"):
                with ui.column().classes("gap-1"):
                    with ui.row().classes("items-center gap-3"):
                        ui.label(name).classes("text-2xl font-bold")

                        if entry.preset:
                            add_pill(t("configs.preset"), "tag-grey")
                        elif readonly:
                            add_pill(t("configs.trained"), "tag-orange")
                        else:
                            add_pill(t("configs.editable"), "tag-blue")

                        unsaved = ui.row()

                        with unsaved:
                            add_pill(t("editor.unsaved"), "tag-orange")

                        unsaved.set_visibility(False)

                    ui.label(entry.path.as_posix()).classes("text-xs mono muted")

                with ui.row().classes("items-center gap-2 no-wrap"):
                    mode = ui.toggle({"form": t("editor.form"), "text": t("editor.text")}, value="form").props("no-caps dense unelevated toggle-color=primary")

                    if readonly:
                        ui.button(t("action.copy_new"), icon="content_copy", on_click=lambda: open_new_config_dialog(list_configs(), entry)).props("outline no-caps")
                    else:
                        ui.button(t("action.revert"), on_click=ui.navigate.reload).props("outline no-caps")
                        save_button = ui.button(t("action.save"), icon="save").props("outline no-caps")

                    # a preset can be trained as it is, a trained config only through a copy
                    if not is_trained(entry):
                        start_button = ui.button(t("action.start_training"), icon="play_arrow").props("unelevated no-caps")

            if readonly:
                with ui.row().classes("w-full items-center gap-2 px-4 py-3 rounded-lg banner-warn"):
                    ui.icon("lock", size="18px")
                    ui.label(t("editor.locked_preset") if entry.preset else t("editor.locked_trained")).classes("text-sm")

            errors = ui.column().classes("w-full gap-1")

            with ui.row().classes("w-full gap-5 items-start no-wrap"):
                form = ui.column().classes("flex-1 gap-4 min-w-0")
                text = ui.codemirror(write_config(config), language="YAML", theme="basicDark" if get_theme()[0] == "dark" else "basicLight").classes("flex-1 min-w-0 card overflow-hidden")
                text.set_visibility(False)

                if readonly:
                    text.props("disable")

                with ui.column().classes("card w-[400px] gap-0 sticky top-4") as preview_card:
                    with ui.column().classes("card-head w-full gap-0"):
                        ui.label(t("editor.preview")).classes("section-title")
                        ui.label(t("editor.preview_hint")).classes("caption")

                    preview = ui.code(write_config(config), language="yaml").classes("w-full text-xs max-h-[75vh] overflow-y-auto").style("border-radius:0 0 12px 12px")

            def mark_changed() -> None:
                state["dirty"] = True
                unsaved.set_visibility(True)
                preview.set_content(write_config(config))

            def build_form() -> None:
                form.clear()

                with form:
                    add_config_form(config, readonly, mark_changed)

            def switch_mode(event) -> None:
                nonlocal config

                if event.value == "text":
                    text.set_value(write_config(config))
                else:
                    parsed = parse_text(text.value)

                    if parsed is None:
                        mode.set_value("text")
                        return

                    config = parsed
                    build_form()
                    preview.set_content(write_config(config))

                form.set_visibility(event.value == "form")
                preview_card.set_visibility(event.value == "form")
                text.set_visibility(event.value == "text")

            def save() -> bool:
                nonlocal config

                if mode.value == "text":
                    parsed = parse_text(text.value)

                    if parsed is None:
                        return False

                    config = parsed

                problems = check_config(config)
                errors.clear()

                if problems:
                    with errors:
                        with ui.column().classes("w-full gap-1 px-4 py-3 rounded-lg banner-bad"):
                            ui.label(t("editor.errors")).classes("text-sm font-semibold")

                            for problem in problems:
                                ui.label(problem).classes("text-sm mono")

                    return False

                save_config(entry, config)
                config = load_config(entry.path)
                text.set_value(write_config(config))
                preview.set_content(write_config(config))
                build_form()
                state["dirty"] = False
                unsaved.set_visibility(False)
                ui.notify(t("editor.saved"), type="positive")
                return True

            def start() -> None:
                # the training reads the file, so the changes are saved first
                if state["dirty"] and not save():
                    return

                open_add_dialog(devices, name)

            mode.on_value_change(switch_mode)

            if not readonly:
                text.on_value_change(lambda: (state.update(dirty=True), unsaved.set_visibility(True)))
                save_button.on_click(save)

            if not is_trained(entry):
                start_button.on_click(start)

            build_form()


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


def add_config_form(config: CommentedMap, readonly: bool, on_change) -> None:
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
                add_transform_list(config[section], key, types["transform"], readonly, on_change)


def add_fields(parent: CommentedMap, key: str, infos: list[FieldInfo], readonly: bool, on_change, columns: int = 3) -> None:
    """Add an input for each field of a mapping, which writes the value into the config on change."""
    def get_mapping() -> CommentedMap:
        if not isinstance(parent.get(key), dict):
            parent[key] = CommentedMap()

        return parent[key]

    values = parent.get(key) or {}

    with ui.grid(columns=columns).classes("w-full gap-x-4 gap-y-3"):
        for info in infos:
            add_field(get_mapping, info, values.get(info.name, info.default), readonly, on_change)


def add_field(get_mapping, info: FieldInfo, value: object, readonly: bool, on_change) -> None:
    """Add the input of one field, chosen by its type hint, with its type and description as a tooltip."""
    def write(new_value: object) -> None:
        get_mapping()[info.name] = convert_value(new_value, info)
        on_change()

    with ui.column().classes("gap-1 min-w-0"):
        with ui.row().classes("items-center gap-1 no-wrap"):
            ui.label(info.name).classes("text-xs mono font-semibold truncate label-strong")

            with ui.icon("help_outline", size="15px").classes("faint cursor-help"):
                ui.tooltip(f"{info.name}: {info.annotation}" + (f" = {info.default}" if not info.required else "") + (f"\n{info.help}" if info.help else "")).classes("whitespace-pre-line")

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


def add_type_menu(registry: list[type], label: str, icon: str, on_choose, props: str) -> ui.button:
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


def add_model_block(model: CommentedMap, block_name: str, registry: list[type], readonly: bool, on_change) -> None:
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


def add_transform_list(section: CommentedMap, key: str, registry: list[type], readonly: bool, on_change) -> None:
    """Add a transform list as blocks in order, with buttons to add, move and delete transforms."""
    classes = {cls.__name__: cls for cls in registry}

    def get_blocks() -> CommentedSeq:
        if not isinstance(section.get(key), list):
            section[key] = CommentedSeq()

        return section[key]

    def add_block(type_name: str) -> None:
        get_blocks().append(CommentedMap({"type": type_name, "args": make_default_args(classes[type_name], [])}))
        transform_list.refresh()
        on_change()

    def move_block(index: int, offset: int) -> None:
        blocks = get_blocks()

        if 0 <= index + offset < len(blocks):
            blocks.insert(index + offset, blocks.pop(index))
            transform_list.refresh()
            on_change()

    def delete_block(index: int) -> None:
        get_blocks().pop(index)
        transform_list.refresh()
        on_change()

    @ui.refreshable
    def transform_list() -> None:
        blocks = section.get(key) or []

        with ui.column().classes("w-full gap-2 pt-2").classes("divider-top"):
            with ui.row().classes("items-baseline gap-2"):
                ui.label(key).classes("text-sm mono font-semibold")
                ui.label(t("editor.in_order", count=len(blocks))).classes("caption")

            for i, block in enumerate(blocks):
                cls = classes.get(block["type"])

                with ui.column().classes("box w-full px-4 py-3 gap-2"):
                    with ui.row().classes("w-full items-center justify-between no-wrap"):
                        with ui.row().classes("items-center gap-3 no-wrap min-w-0"):
                            ui.label(str(i + 1)).classes("w-6 h-6 rounded-full text-xs font-semibold flex items-center justify-center index-dot")
                            ui.label(str(block["type"])).classes("font-semibold")
                            ui.label(describe_type(cls) if cls else t("editor.unknown_type")).classes("text-xs muted truncate")

                        if not readonly:
                            with ui.row().classes("gap-0 no-wrap"):
                                ui.button(icon="arrow_upward", on_click=lambda i=i: move_block(i, -1)).props("flat dense round size=sm").tooltip(t("action.move_up"))
                                ui.button(icon="arrow_downward", on_click=lambda i=i: move_block(i, 1)).props("flat dense round size=sm").tooltip(t("action.move_down"))
                                ui.button(icon="delete_outline", on_click=lambda i=i: delete_block(i), color="negative").props("flat dense round size=sm").tooltip(t("action.delete"))

                    infos = describe_class(cls, []) if cls else []

                    if infos:
                        add_fields(block, "args", infos, readonly, on_change, columns=4)

            if not readonly:
                with ui.row().classes("add-tile w-full items-center gap-2"):
                    ui.icon("add", size="20px")
                    ui.label(t("action.add_transform"))

                    with ui.menu().props("auto-close fit").classes("max-h-[60vh]"):
                        for cls in registry:
                            with ui.menu_item(on_click=lambda name=cls.__name__: add_block(name)):
                                with ui.column().classes("gap-0 py-1"):
                                    ui.label(cls.__name__).classes("font-semibold text-sm")
                                    ui.label(describe_type(cls)).classes("text-xs muted")

    transform_list()


# datasets
def add_datasets_page(devices: list[tuple[str, str]]) -> None:
    """Add the page of the datasets in data/, with the download button of the ones that a script can make."""
    @ui.page("/datasets")
    def datasets_page() -> None:
        add_frame(devices, "/datasets")
        shown = {"datasets": list_datasets()}
        users = find_dataset_users()

        def update(datasets: list[DatasetInfo]) -> None:
            shown["datasets"] = datasets
            dataset_list.refresh()

        with ui.column().classes("w-full max-w-[1280px] px-8 py-7 gap-5"):
            add_page_title(t("datasets.title"), t("datasets.subtitle"))

            @ui.refreshable
            def dataset_list() -> None:
                columns = "110px minmax(180px, 2fr) 80px 110px 110px minmax(160px, 2fr) 130px"

                with ui.column().classes("card w-full gap-0 overflow-x-auto"):
                    with ui.grid(columns=columns).classes("w-full gap-x-4 list-head"):
                        for key in ["status", "name", "classes", "train", "val", "used_by", "actions"]:
                            ui.label(t(f"datasets.column.{key}"))

                    if not shown["datasets"]:
                        ui.label(t("datasets.empty")).classes("text-sm muted p-5")

                    for info in shown["datasets"]:
                        row = ui.grid(columns=columns).classes("w-full gap-x-4 items-center list-row clickable")
                        row.on("click", lambda name=info.name: ui.navigate.to(f"/datasets/{name}"))

                        with row:
                            add_pill(t(f"datasets.state.{info.state}"), f"dataset-{info.state}")

                            with ui.column().classes("gap-0 min-w-0"):
                                ui.label(info.name).classes("font-semibold text-link truncate")
                                ui.label(info.path.as_posix()).classes("text-xs mono muted")

                            ui.label(str(len(info.classes)) if info.classes else "—").classes("mono")

                            for split in info.splits:
                                ui.label(f"{split.images:,}" if split.images else "—").classes("mono")

                            ui.label(", ".join(users.get(info.path.as_posix(), [])) or "—").classes("text-sm muted truncate")

                            with ui.row().classes("gap-1 no-wrap").on("click.stop", lambda: None):
                                add_download_actions(info, lambda: update(list_datasets()))

            dataset_list()
            watch(list_datasets, shown["datasets"], update)

            with ui.column().classes("card w-full gap-0"):
                with ui.row().classes("card-head w-full items-baseline gap-3"):
                    ui.label(t("datasets.layout")).classes("section-title")
                    ui.label(t("datasets.layout_hint")).classes("caption")

                ui.label(DATASET_LAYOUT).classes("mono text-xs whitespace-pre p-5")


def find_dataset_users() -> dict[str, list[str]]:
    """Return the configs that train on each dataset directory, by its path."""
    users = {}

    for entry in list_configs():
        dataset_dir = Path(read_config(entry.path)["train"]["dataset_dir"]).as_posix()
        users.setdefault(dataset_dir, []).append(entry.name)

    return users


def add_download_actions(info: DatasetInfo, on_done) -> None:
    """Add the download button of a dataset that a script can make and that is not ready, or the stop button of a running download."""
    if info.state == "downloading":
        def stop() -> None:
            stop_download(info.name)
            on_done()

        ui.button(t("action.stop"), icon="stop", on_click=stop, color="negative").props("flat dense no-caps").classes("px-3")
    elif info.script is not None and info.state != "ready":
        def download() -> None:
            start_download(info.name)
            ui.notify(t("notify.download_started", name=info.name), type="positive")
            on_done()

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


def add_dataset_page(devices: list[tuple[str, str]]) -> None:
    """Add the page of one dataset: its folder tree, its classes and one drawn image of each split."""
    @ui.page("/datasets/{name}")
    def dataset_page(name: str) -> None:
        add_frame(devices, "/datasets")
        shown = {"info": read_dataset(name), "seed": 0}

        def update(info: DatasetInfo) -> None:
            shown["info"] = info
            content.refresh()

        with ui.column().classes("w-full max-w-[1280px] px-8 py-7 gap-4"):
            ui.link(t("action.back_datasets"), "/datasets").classes("text-sm muted no-underline")

            @ui.refreshable
            def content() -> None:
                info = shown["info"]

                with ui.row().classes("w-full items-start justify-between no-wrap"):
                    with ui.column().classes("gap-1"):
                        with ui.row().classes("items-center gap-3"):
                            ui.label(name).classes("text-2xl font-bold")
                            add_pill(t(f"datasets.state.{info.state}"), f"dataset-{info.state}")

                        ui.label(info.path.as_posix() + (f"  ·  {info.script.as_posix()}" if info.script else "")).classes("text-xs mono muted")

                    with ui.row().classes("gap-2"):
                        add_download_actions(info, lambda: update(read_dataset(name)))

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

            content()
            watch(lambda: read_dataset(name), shown["info"], update)


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
        watch(lambda: read_console(log_path, CONFIG["console_tail"]), initial, show)


def add_samples(info: DatasetInfo, shown: dict) -> None:
    """Add one random image of each split with its labels drawn, and a button for other images."""
    with ui.column().classes("card w-full gap-0"):
        with ui.row().classes("card-head w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label(t("datasets.samples")).classes("section-title")
                ui.label(t("datasets.samples_hint")).classes("caption")

            other_button = ui.button(t("action.other_images"), icon="shuffle").props("outline dense no-caps").classes("px-3")

        samples = ui.grid(columns=len(info.splits)).classes("w-full gap-4 p-5")

    async def draw() -> None:
        other_button.props("loading")
        samples.clear()

        for split in info.splits:
            with samples:
                column = ui.column().classes("gap-1 min-w-0")

            # a broken label file is what this page is for, so it is shown instead of raised
            try:
                sample = await run.io_bound(draw_sample, info, split.name, shown["seed"])
            except ValueError as error:
                sample, problem = None, str(error)
            else:
                problem = None if sample else t("datasets.no_images")

            with column:
                ui.label(split.name).classes("caption font-semibold")

                if sample:
                    ui.element("img").props(f'src="data:image/jpeg;base64,{base64.b64encode(sample[0]).decode()}"').classes("w-full rounded-lg img-frame")
                    ui.label(sample[1]).classes("text-xs mono muted")
                else:
                    with ui.column().classes("w-full h-48 items-center justify-center rounded-lg px-4 empty-frame"):
                        ui.label(problem).classes("text-sm mono text-center text-warn")

        other_button.props(remove="loading")

    async def draw_other() -> None:
        shown["seed"] += 1
        await draw()

    other_button.on_click(draw_other)
    ui.timer(0.1, draw, once=True)


def ignore_connection_reset() -> None:
    """Keep the windows event loop from printing a traceback whenever a browser closes a connection."""
    loop = asyncio.get_running_loop()

    def handle(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        if not isinstance(context.get("exception"), ConnectionResetError):
            loop.default_exception_handler(context)

    loop.set_exception_handler(handle)


def main():
    """Serve the web ui to train, monitor and check the models of this project.

    Usage: uv run webui\\app.py
    """
    # the device list is asked once, in a short process
    devices = list_devices()
    Path("runs").mkdir(exist_ok=True)
    app.add_media_files("/runs", Path("runs"))
    app.on_startup(ignore_connection_reset)

    for add_page in [add_trainings_page, add_training_page, add_configs_page, add_editor_page, add_datasets_page, add_dataset_page]:
        add_page(devices)

    ui.run(title="MyDets", port=CONFIG["port"], storage_secret=CONFIG["storage_secret"], reload=False, show=True)


if __name__ in {"__main__", "__mp_main__"}:
    main()
