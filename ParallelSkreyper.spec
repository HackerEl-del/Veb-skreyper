# -*- mode: python ; coding: utf-8 -*-
# PyInstaller sozlamasi: ParallelSkreyper.exe (desktop ilova, MVP)
# Yig'ish: build_exe.bat  (yoki: python -m PyInstaller --noconfirm --clean ParallelSkreyper.spec)
# Natija:  dist\ParallelSkreyper\ParallelSkreyper.exe  (papka butunligicha tarqatiladi)
#
# GPU qismini (numba-cuda + NVIDIA kutubxonalari) qo'shish: BUNDLE_CUDA=1 (standart).
# GPU siz, kichikroq exe: build_exe.bat nogpu
import os
import site
import sys
from PyInstaller.utils.hooks import collect_all

BUNDLE_CUDA = os.environ.get("BUNDLE_CUDA", "1") == "1"
datas, binaries, hiddenimports = [], [], ["parallel_skreyper"]


def add_pkg(name):
    """Paketni (kod, ma'lumot, DLL) to'liq qo'shish; bo'lmasa o'tkazib yuborish."""
    try:
        d, b, h = collect_all(name)
        datas.extend(d)
        binaries.extend(b)
        hiddenimports.extend(h)
        print(f"[spec] {name}: {len(d)} fayl, {len(b)} DLL")
    except Exception as e:                       # noqa: BLE001
        print(f"[spec] {name} o'tkazib yuborildi: {e}")


def add_tree(path, dest):
    """Papkani diskdagi ko'rinishida (manba fayllar bilan) qo'shish."""
    n = 0
    for root, _, files in os.walk(path):
        if "__pycache__" in root:
            continue
        for f in files:
            datas.append((os.path.join(root, f), os.path.join(dest, os.path.relpath(root, path))))
            n += 1
    print(f"[spec] {dest}: {n} fayl")


for pkg in ("numba", "llvmlite", "bs4", "mpi4py"):
    add_pkg(pkg)

runtime_hooks = []
if BUNDLE_CUDA:
    sp_dirs = [p for p in site.getsitepackages() + [site.getusersitepackages()] if os.path.isdir(p)]
    for sp in sp_dirs:
        for name in ("numba_cuda", "nvidia", "cuda"):
            p = os.path.join(sp, name)
            if os.path.isdir(p):
                # numba-cuda o'z modullarini numba.cuda nomi bilan diskdan yuklaydi,
                # NVIDIA kutubxonalari (NVVM, nvrtc) ham papka tuzilishi bilan kerak
                add_tree(p, name)
        for f in os.listdir(sp):
            low = f.lower()
            if low.endswith(".pth") and "cuda" in low:
                datas.append((os.path.join(sp, f), "_pth"))      # rthook bajaradi
                print(f"[spec] .pth: {f}")
            elif low.endswith(".py") and "cuda" in low:
                # .pth chaqiradigan modul (masalan _numba_cuda_redirector.py) - ildizga
                datas.append((os.path.join(sp, f), "."))
                hiddenimports.append(f[:-3])
                print(f"[spec] modul: {f}")
    # Paket metama'lumotlari (*.dist-info): cuda-core / numba-cuda o'z versiyasini
    # importlib.metadata.version("cuda-bindings") orqali tekshiradi - exe da ham kerak
    from importlib.metadata import distributions
    from PyInstaller.utils.hooks import copy_metadata
    seen = set()
    for dist in distributions():
        name = (dist.metadata["Name"] or "").strip()
        low = name.lower().replace("_", "-")
        if low and low not in seen and low.startswith(("cuda", "nvidia", "numba", "llvmlite")):
            seen.add(low)
            try:
                datas.extend(copy_metadata(name))
                print(f"[spec] metadata: {name}")
            except Exception as e:                   # noqa: BLE001
                print(f"[spec] metadata {name} o'tkazib yuborildi: {e}")
    runtime_hooks.append("rthook_cuda.py")
    hiddenimports += ["numba_cuda"]

a = Analysis(
    ["skreyper_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=runtime_hooks,
    excludes=["pytest", "IPython", "jupyter", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ParallelSkreyper",
    console=False,               # oyna ilova; buyruq qatori rejimi (--task5, MPI) ham ishlaydi
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, upx=False, name="ParallelSkreyper")
