from pathlib import Path
import json

from nicegui import app, ui

from webui.bridge.processes import list_busy_devices
from webui.tabs.common.texts import TEXTS, get_language, t
from webui.tabs.common.widgets import Watcher


NAV = [("/", "nav.trainings", "model_training"), ("/configs", "nav.configs", "tune"), ("/datasets", "nav.datasets", "photo_library")]
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
