# Pyodide 0.29.5 (minimal) + numpy 2.2.5

What the live sandbox (`town.html?live=1`, `town/live-worker.js`) needs to run CPython 3.13
with numpy in a Web Worker, served next to the page so it works without a CDN:

| file | from |
|---|---|
| `pyodide.js`, `pyodide.asm.js`, `pyodide.asm.wasm`, `python_stdlib.zip`, `pyodide-lock.json` | npm package `pyodide@0.29.5` (unmodified) |
| `numpy-2.2.5-cp313-cp313-pyemscripten_2025_0_wasm32.whl` | the Pyodide 0.29.5 build of numpy; its sha256 (`800c98ed…7f05c`) matches `pyodide-lock.json`, and Pyodide checks it on load |

When this folder is missing the worker falls back to `https://cdn.jsdelivr.net/pyodide/v0.29.5/full/`.
To update, replace all files with the same version's files (the lock file pins the numpy wheel).

Licenses: Pyodide MPL-2.0 (https://github.com/pyodide/pyodide), CPython PSF, numpy BSD-3-Clause.
