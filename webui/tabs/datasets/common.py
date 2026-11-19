from collections.abc import Awaitable, Callable

from nicegui import ui

from webui.bridge.datasets import DatasetInfo, start_download, stop_download
from webui.tabs.common.texts import t


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
