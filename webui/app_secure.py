from pathlib import Path
import asyncio
import secrets
import sys

# the web ui runs from the project root like the scripts, and imports their modules from there
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nicegui import app, ui

from webui.app import CONFIG, RUNS_DIR, ignore_connection_reset
from webui.security.login import TokenGate


def main():
    """Serve the web ui behind a login page that asks for the access token printed at the start.

    Usage: uv run webui\\app_secure.py
    """
    token = secrets.token_urlsafe(32)
    RUNS_DIR.mkdir(exist_ok=True)
    app.add_media_files("/runs", RUNS_DIR)
    app.on_startup(lambda: asyncio.get_running_loop().set_exception_handler(ignore_connection_reset))
    app.add_middleware(TokenGate, token=token)
    print(f"\nCopy this access token and paste it on the login page, once per browser:\n\n{token}\n", flush=True)
    ui.run(title="MyDets", port=CONFIG["port"], storage_secret=CONFIG["storage_secret"], reload=False, show=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
