from pathlib import Path
import asyncio
import sys

# the web ui runs from the project root like the scripts, and imports their modules from there
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
print("Starting the MyDets web UI...", flush=True)

from nicegui import app, ui

from webui.bridge.processes import RUNS_DIR

# importing the pages registers their routes
import webui.tabs.trainings.page
import webui.tabs.trainings.subpage.page
import webui.tabs.configs.page
import webui.tabs.configs.subpage.page
import webui.tabs.datasets.page
import webui.tabs.datasets.subpage.page

CONFIG = {
    "port": 8080,
    "storage_secret": "mydets-webui"
}


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
