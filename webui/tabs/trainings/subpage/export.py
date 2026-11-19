from pathlib import Path
import tempfile

from nicegui import run, ui

from predict import CONFIG as PREDICT_CONFIG
from shared.output_directory import OutputDirectory
from shared.settings import read_config
from webui.bridge.processes import export_onnx, run_onnx_tool
from webui.tabs.common.texts import t
from webui.tabs.common.widgets import add_busy, show_image


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
