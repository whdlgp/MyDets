# Web UI

A browser interface for training, monitoring and checking the models of this project. Run it from the project root with `uv run webui\app.py`.

## Structure

```
webui/
├─ app.py                      Entry point: starts the server and imports the pages
├─ app_secure.py               Secure entry point: starts the server with token authentication
├─ bridge/                     Connects the web UI to the rest of the project (configs/, data/, runs/ and the scripts)
│  ├─ configs.py               Training config files: list, check, save, summarize
│  ├─ datasets.py              Datasets in data/: read, draw samples, download
│  ├─ processes.py             Runs the scripts as processes and reads the results they leave in runs/
│  └─ schema.py                Describes the model and settings classes for the config form
├─ security/                   Token-based authentication
│  └─ login.py                 Login page and authentication middleware
└─ tabs/                       Everything the user sees
   ├─ common/                  Shared by every tab
   │  ├─ layout.py             Side menu, theme, language, page title
   │  ├─ widgets.py            Small UI parts (pill: status badge, busy: spinner, image), the refresh timer and its settings
   │  ├─ start_training.py     Dialog that starts a training from a config
   │  ├─ texts.py              Translated texts
   │  ├─ locales/              One json file per language
   │  └─ themes.json           Theme colors
   ├─ trainings/
   │  ├─ page.py               '/'                 Training list
   │  ├─ common.py             Stop and resume buttons, time format (used by both the list and the subpage)
   │  └─ subpage/
   │     ├─ page.py            '/train/{name}'     One training: header and results tab
   │     ├─ monitor.py         Monitor tab: progress, loss and mAP charts, console
   │     ├─ prediction.py      Predict tab
   │     └─ export.py          Export tab: ONNX export, verify, inference
   ├─ configs/
   │  ├─ page.py               '/configs'          Config list
   │  ├─ common.py             New config dialog (used by both the list and the subpage)
   │  └─ subpage/
   │     ├─ page.py            '/configs/{name}'   One config: editor with form and text modes
   │     └─ form.py            The form: fields, sections, model blocks, transform lists
   └─ datasets/
      ├─ page.py               '/datasets'         Dataset list
      ├─ common.py             Download and stop buttons (used by both the list and the subpage)
      └─ subpage/
         ├─ page.py            '/datasets/{name}'  One dataset: folder tree, classes, download console
         └─ samples.py         Random sample images with their labels drawn
```

## How it is organized

- The web UI has two parts. `bridge/` is where it meets the rest of the project, and `tabs/` is everything the user sees, which calls the bridge. The bridge never imports from `tabs/` and creates no UI elements, so it can be used without any UI code.
- A tab has a **page** (its list) and a **subpage** (what opens when you click a list item). In every folder, `page.py` is the screen itself and the other files are its parts.
- A file's location shows who uses it:
  - `tabs/common/` is used by more than one tab.
  - `<tab>/common.py` is used by both the page and the subpage of that tab.
  - The other files in `subpage/` are used by that subpage only.
- Layers, from top to bottom: pages, parts of a subpage, `<tab>/common.py`, `tabs/common/`, `bridge/`, `shared/`. A file imports only from the layers below it, and never from another tab. Files inside `bridge/` may import each other, and so may files inside `tabs/common/`.
- `app.py` registers a page by importing its module: importing runs the `@ui.page` decorator, which registers the route.
- `app_secure.py` uses the same web UI as `app.py`, with token-based authentication.

## Adding code

| What | Where |
| --- | --- |
| A new tab | `tabs/<name>/page.py`, and `tabs/<name>/subpage/page.py` if it opens an item |
| A new part of a subpage | `tabs/<tab>/subpage/<name>.py` |
| Used by both the page and the subpage of a tab | `tabs/<tab>/common.py` |
| Used by several tabs | `tabs/common/` |
| Reading or writing project files, running scripts, reading project classes | `bridge/` |
| Authentication | `security/login.py` |

## Running the Web UI

* `app.py`
  * Local use
  * No authentication

* `app_secure.py`
  * Remote access or external hosting
  * Token-based authentication
  * Must be accessed through an HTTPS proxy provided by the hosting platform

## Notes

- `CONFIG` in `app.py` and in `tabs/common/widgets.py` holds UI settings (port, refresh interval, console lines). It is not the training config in `configs/`.
- If a text key is missing in a language, it falls back to English, and then to the key itself.
