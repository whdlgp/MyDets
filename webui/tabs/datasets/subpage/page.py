from collections.abc import Awaitable, Callable

from nicegui import run, ui

from webui.bridge.datasets import DatasetInfo, list_downloads, read_dataset
from webui.bridge.processes import list_devices, read_console
from webui.tabs.common.layout import add_frame
from webui.tabs.common.texts import t
from webui.tabs.common.widgets import CONFIG, Watcher, add_busy, add_pill
from webui.tabs.datasets.common import add_download_actions
from webui.tabs.datasets.subpage.samples import add_samples


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
