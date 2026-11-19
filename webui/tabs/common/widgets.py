from collections.abc import Callable
from pathlib import Path
import inspect

from nicegui import run, ui

from webui.bridge.processes import RUNS_DIR
from webui.tabs.common.texts import t


CONFIG = {
    "refresh_seconds": 2,
    "console_tail": 20,
    "console_lines": 300
}


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


def show_image(path: Path) -> None:
    """Show an image of runs/, with its modification time in the url so that a redrawn image replaces the cached one."""
    if path.exists():
        # a plain img keeps the height of the image, where ui.image keeps a fixed ratio
        source = f"/runs/{path.relative_to(RUNS_DIR).as_posix()}?v={path.stat().st_mtime_ns}"
        ui.element("img").props(f'src="{source}"').classes("w-full rounded-lg img-frame")
    else:
        with ui.column().classes("w-full h-48 items-center justify-center rounded-lg empty-frame"):
            ui.label(t("predict.no_images")).classes("text-sm faint")
