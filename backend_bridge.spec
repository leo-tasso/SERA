# PyInstaller spec — freezes the SERA backend bridge (ui/backend_bridge.py)
# into a standalone one-folder executable so released desktop builds run
# without a system Python install.
#
# Build (from the repo root, after `pip install -e ".[build]"`):
#   pyinstaller --noconfirm --clean --distpath ui/pydist backend_bridge.spec
#
# Output: ui/pydist/backend_bridge/  (the executable + its _internal payload).
# Electron ships that folder as <resources>/payload/backend (see ui/package.json
# "extraResources"); main.js spawns it instead of `python` in packaged builds.
#
# The project data, trained twin and GeoJSON are NOT frozen in here — they are
# shipped separately as Electron extraResources and located at runtime via the
# SERA_PROJECT_ROOT env var that main.js sets. This keeps the executable to just
# the interpreter + libraries (numpy/pandas/scikit-learn/scipy/...).

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# scikit-learn and scipy load a lot of submodules lazily (and the pickled twin
# references sklearn estimator classes), so pull them in wholesale.
hidden_imports = (
    collect_submodules("sklearn")
    + collect_submodules("scipy")
    + collect_submodules("sera")
)

datas = collect_data_files("sklearn") + collect_data_files("scipy")

a = Analysis(
    ["ui/backend_bridge.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    # Trim things the bridge never imports to keep the bundle smaller.
    excludes=["matplotlib", "tkinter", "pytest", "IPython", "jupyter", "notebook"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="backend_bridge",
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="backend_bridge",
)
