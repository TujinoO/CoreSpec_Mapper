# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import os


ROOT = Path(SPECPATH).parent
SOURCE = ROOT / "src"
RESOURCE_ROOT = SOURCE / "corespec_mapper" / "resources"
os.environ["QT_API"] = "pyside6"


a = Analysis(
    [str(ROOT / "packaging" / "corespec_mapper_v5_launcher.py")],
    pathex=[str(SOURCE)],
    binaries=[],
    datas=[
        (str(RESOURCE_ROOT), "corespec_mapper/resources"),
    ],
    hiddenimports=[
        "torch",
        "PIL.Image",
        "corespec_mapper.v5_service",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "PyQt5",
        "PyQt6",
        "boto3",
        "botocore",
        "dask",
        "distributed",
        "docutils",
        "jupyter",
        "llvmlite",
        "matplotlib",
        "numba",
        "notebook",
        "openpyxl",
        "pandas",
        "pyarrow",
        "pytest",
        "qtpy",
        "scipy",
        "sphinx",
        "sqlalchemy",
        "tables",
        "tkinter",
        "torchaudio",
        "torch.onnx",
        "torch.utils.tensorboard",
        "torchvision",
        "xarray",
    ],
    noarchive=False,
    optimize=1,
)

# The foreground model is CPU-compatible.  Do not ship the 4 GB CUDA toolchain
# from the development environment; retain torch_cpu and the common runtime DLLs.
GPU_BINARY_MARKERS = (
    "c10_cuda",
    "cublas",
    "cudnn",
    "cufft",
    "cupti",
    "curand",
    "cusolver",
    "cusparse",
    "nvjitlink",
    "nvperf",
    "nvrtc",
    "torch_cuda",
)
a.binaries = [
    entry
    for entry in a.binaries
    if not any(marker in (entry[0] + " " + entry[1]).casefold() for marker in GPU_BINARY_MARKERS)
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CoreSpecMapperV5_3",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="x86_64",
    codesign_identity=None,
    entitlements_file=None,
    icon=str(RESOURCE_ROOT / "corespec_logo.ico"),
    version=str(ROOT / "packaging" / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="CoreSpecMapperV5_3",
)
