from collections.abc import Awaitable, Callable
from pathlib import Path

from nicegui import run, ui

from shared.settings import read_config
from webui.bridge.configs import list_configs
from webui.bridge.datasets import DatasetInfo, list_datasets, list_downloads
from webui.bridge.processes import list_devices
from webui.tabs.common.layout import add_frame, add_page_title
from webui.tabs.common.texts import t
from webui.tabs.common.widgets import Watcher, add_busy, add_pill
from webui.tabs.datasets.common import add_download_actions


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
