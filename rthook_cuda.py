# PyInstaller runtime hook: exe ichida numba-cuda va NVIDIA kutubxonalarini topish.
# Oddiy Python'da site-packages dagi *.pth fayllar ishga tushadi - exe da bunday
# mexanizm yo'q, shuning uchun ularni shu yerda qo'lda bajaramiz.
import os
import sys

base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
pth_dir = os.path.join(base, "_pth")
if os.path.isdir(pth_dir):
    for name in sorted(os.listdir(pth_dir)):
        try:
            with open(os.path.join(pth_dir, name), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith(("import ", "import\t")):
                        try:
                            exec(line)                       # .pth dagi kabi
                        except Exception:
                            pass
                    else:
                        p = os.path.join(base, line)
                        if os.path.isdir(p) and p not in sys.path:
                            sys.path.append(p)
        except OSError:
            pass
# NVIDIA DLL lari (nvidia/*/bin) - Windows da DLL qidiruv yo'liga qo'shish
nv = os.path.join(base, "nvidia")
if os.path.isdir(nv) and hasattr(os, "add_dll_directory"):
    for root, dirs, files in os.walk(nv):
        if any(f.lower().endswith(".dll") for f in files):
            try:
                os.add_dll_directory(root)
            except OSError:
                pass
            os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
