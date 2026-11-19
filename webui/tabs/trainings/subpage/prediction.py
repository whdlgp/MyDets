from pathlib import Path
import tempfile

from nicegui import run, ui

from predict import CONFIG as PREDICT_CONFIG
from shared.output_directory import OutputDirectory
from shared.settings import read_config
from webui.bridge.processes import list_busy_devices, predict_images
from webui.tabs.common.texts import t
from webui.tabs.common.widgets import add_busy, show_image


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
