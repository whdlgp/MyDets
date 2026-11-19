import base64

from nicegui import run, ui

from webui.bridge.datasets import DatasetInfo, draw_sample
from webui.tabs.common.texts import t


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
