from pathlib import Path
import json

from nicegui import app

LOCALES_DIR = Path(__file__).parent / "locales"
TEXTS = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(LOCALES_DIR.glob("*.json"))}


def get_language() -> str:
    """Return the language that the current browser chose, English by default."""
    return app.storage.user.get("language", "en")


def t(key: str, **values: object) -> str:
    """Return the text of a key in the current language, falling back to English and then to the key.

    Args:
        key: key in the locale files, such as nav.train.
        values: values of the {placeholders} in the text.

    Returns:
        Text with the placeholders filled.
    """
    text = TEXTS.get(get_language(), {}).get(key) or TEXTS["en"].get(key, key)
    return text.format(**values)
