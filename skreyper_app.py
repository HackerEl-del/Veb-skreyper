#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 PARALLEL VEB-SKREYPER VA MATN TAHLILI TIZIMI - DESKTOP ILOVA (MVP)
 "Parallel dasturlash" (PAD2504) - 5-topshiriq, yakuniy bosqich

 Bajardi : Boltayev Elbek, Soyibjonov Otabek
-------------------------------------------------------------------------------
 Ilova semestr davomida yozilgan parallel_skreyper.py dvigatelidan foydalanadi:
   * sahifalarni yuklash  - ThreadPoolExecutor (2-topshiriq, I/O)
   * tahlil dvigatellari  - Python ketma-ket / ProcessPool (2-topshiriq),
                            Numba prange (3-topshiriq), GPU CUDA va gibrid
                            CPU+GPU (4-topshiriq)
   * "Taqqoslash" yorlig'i - barcha dvigatellarni bitta korpusda jonli
                            solishtirish (5-topshiriq namoyishi uchun)
 Ishga tushirish:  python skreyper_app.py
 Buyruq qatori argumentlari berilsa (masalan --task5 yoki MPI ishchisi
 --mpi-worker), ilova GUI ochmasdan parallel_skreyper.main() ni chaqiradi -
 shuning uchun yig'ilgan ParallelSkreyper.exe ham MPI ishchisi bo'la oladi.
===============================================================================
"""
import csv
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import parallel_skreyper as ps

APP_TITLE = "Parallel veb-skreyper"
AUTHORS = "Boltayev Elbek, Soyibjonov Otabek"


# ============================ DVIGATEL (GUI siz qism) ========================
class Corpus:
    """Yuklangan sahifalar: Python dvigatellari uchun matn ro'yxati, Numba va GPU
    uchun bitta uzluksiz bayt massiv (buf, offs) - kerak bo'lganda yasaladi."""

    @ps.muallif(ps.ELBEK, 5)
    def __init__(self, htmls):
        self.htmls = htmls
        self._packed = None

    @property
    def packed(self):
        if self._packed is None:
            self._packed = ps.pack_corpus(self.htmls)
        return self._packed

    @property
    def mb(self):
        return sum(len(h) for h in self.htmls) / 2 ** 20


@ps.muallif(ps.OTABEK, 5)
def _numba_words(res, corpus):
    """Numba/GPU natijasini {so'z: soni} lug'atiga aylantirish."""
    return ps.decode_words(res, corpus.packed[0])


# Har bir dvigatel: (nomi, talab, funksiya(corpus, ishchilar) -> natija, natija -> lug'at)
ENGINES = {
    "py_seq": ("Ketma-ket (Python, bs4)", None,
               lambda c, w: ps.analyze_sequential(c.htmls), lambda r, c: dict(r)),
    "py_proc": ("Jarayonlar (ProcessPool)", None,
                lambda c, w: ps.analyze_parallel(c.htmls, w, "process"), lambda r, c: dict(r)),
    "nb_seq": ("Numba ketma-ket", "numba",
               lambda c, w: ps.count_words_seq(*c.packed), _numba_words),
    "nb_par": ("Numba prange (OpenMP uslubi)", "numba",
               lambda c, w: ps.count_words_parallel(*c.packed, w, 0, cyclic=True), _numba_words),
    "gpu": ("GPU (CUDA)", "gpu",
            lambda c, w: ps.gpu_count_words(*c.packed, 256, 4), _numba_words),
    "hetero": ("Gibrid CPU + GPU (90% GPU)", "gpu",
               lambda c, w: ps.hetero_static(c.packed[0], c.packed[0], c.packed[1], 0.9, w,
                                             {"tpb": 256, "streams": 4}), _numba_words),
}


@ps.muallif(ps.ELBEK, 5)
def run_engine(key, corpus, workers, sync=True):
    """Bitta dvigatelni ishga tushiradi; (natija lug'ati, sof tahlil vaqti s)."""
    _, _, fn, conv = ENGINES[key]
    t0 = time.perf_counter()
    r = fn(corpus, workers)
    if sync and ENGINES[key][1] == "gpu":
        ps.cuda.synchronize()
    dt = time.perf_counter() - t0
    return conv(r, corpus), dt


# ================================= GUI ======================================
def _import_tk():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    return tk, ttk, filedialog, messagebox


class App:
    """Asosiy oyna. Og'ir ishlar (yuklash, tahlil, taqqoslash) alohida ishchi
    oqimda bajariladi; ishchi natijalarni queue.Queue orqali yuboradi, GUI esa
    har 100 ms da navbatni o'qiydi - Tk faqat asosiy oqimdan boshqariladi."""

    @ps.muallif(ps.BOTH, 5)
    def __init__(self):
        tk, ttk, self.filedialog, self.messagebox = _import_tk()
        self.tk, self.ttk = tk, ttk
        self.root = tk.Tk()
        self.root.title(f"{APP_TITLE} - MVP")
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)
        self.q = queue.Queue()
        self.stop_flag = threading.Event()
        self.busy = False
        self.corpus = None
        self.last_words = None
        self.avail = {"numba": False, "gpu": False}
        self.gpu_name = None
        self.gpu_error = None
        self._build_ui()
        self.root.after(100, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        threading.Thread(target=self._warmup, daemon=True).start()

    # ------------------------------ interfeys ------------------------------
    @ps.muallif(ps.OTABEK, 5)
    def _build_ui(self):
        tk, ttk = self.tk, self.ttk
        style = ttk.Style()
        if "clam" in style.theme_names() and sys.platform != "win32":
            style.theme_use("clam")
        style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=6)
        style.configure("Val.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Head.TLabel", font=("Segoe UI", 15, "bold"))

        top = ttk.Frame(self.root, padding=(12, 8, 12, 0))
        top.pack(fill="x")
        ttk.Label(top, text=APP_TITLE, style="Head.TLabel").pack(side="left")
        ttk.Label(top, text="  yuklash: oqimlar · tahlil: Numba / GPU / jarayonlar",
                  foreground="#555").pack(side="left")

        self.status = tk.StringVar(value="Numba: tayyorlanmoqda...   GPU: tekshirilmoqda...")
        ttk.Label(self.root, textvariable=self.status, relief="sunken", anchor="w",
                  padding=(8, 2)).pack(fill="x", side="bottom")      # avval - doim ko'rinadi
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=8)
        self.tab_run = ttk.Frame(nb, padding=8)
        self.tab_cmp = ttk.Frame(nb, padding=8)
        self.tab_sys = ttk.Frame(nb, padding=8)
        nb.add(self.tab_run, text="  Tahlil  ")
        nb.add(self.tab_cmp, text="  Taqqoslash  ")
        nb.add(self.tab_sys, text="  Tizim haqida  ")
        self._build_run_tab()
        self._build_cmp_tab()
        self._build_sys_tab()

    def _build_run_tab(self):
        tk, ttk = self.tk, self.ttk
        left = ttk.Frame(self.tab_run)
        left.pack(side="left", fill="y", padx=(0, 10))
        right = ttk.Frame(self.tab_run)
        right.pack(side="left", fill="both", expand=True)

        src = ttk.LabelFrame(left, text="Manba", padding=8)
        src.pack(fill="x")
        self.v_src = tk.StringVar(value="demo")
        ttk.Radiobutton(src, text="Demo: lokal sinov serveri", variable=self.v_src,
                        value="demo", command=self._src_changed).grid(row=0, column=0,
                                                                      columnspan=2, sticky="w")
        self.v_pages = tk.IntVar(value=200)
        self.v_kb = tk.IntVar(value=100)
        self.v_delay = tk.IntVar(value=100)
        self.demo_widgets = []
        for i, (lab, var, lo, hi, inc) in enumerate((
                ("Sahifalar soni", self.v_pages, 10, 2000, 10),
                ("Sahifa hajmi, KB", self.v_kb, 5, 1000, 5),
                ("Tarmoq kechikishi, ms", self.v_delay, 0, 2000, 50)), start=1):
            ttk.Label(src, text=lab).grid(row=i, column=0, sticky="w", padx=(18, 4), pady=1)
            sp = ttk.Spinbox(src, from_=lo, to=hi, increment=inc, textvariable=var, width=8)
            sp.grid(row=i, column=1, sticky="w", pady=1)
            self.demo_widgets.append(sp)
        ttk.Radiobutton(src, text="Real saytlar (URL ro'yxati)", variable=self.v_src,
                        value="urls", command=self._src_changed).grid(row=4, column=0,
                                                                      columnspan=2, sticky="w",
                                                                      pady=(8, 0))
        self.txt_urls = tk.Text(src, width=36, height=6, font=("Consolas", 9))
        self.txt_urls.grid(row=5, column=0, columnspan=2, pady=2)
        self.txt_urls.insert("1.0", "https://uz.wikipedia.org/wiki/Python\n")
        self.btn_load = ttk.Button(src, text="Fayldan yuklash...", command=self._load_urls)
        self.btn_load.grid(row=6, column=0, columnspan=2, sticky="w")

        par = ttk.LabelFrame(left, text="Parallel sozlamalar", padding=8)
        par.pack(fill="x", pady=8)
        self.v_fthreads = tk.IntVar(value=16)
        self.v_workers = tk.IntVar(value=min(8, ps.CPU_COUNT))
        ttk.Label(par, text="Yuklash oqimlari").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(par, from_=1, to=64, textvariable=self.v_fthreads, width=6).grid(row=0, column=1)
        ttk.Label(par, text="Tahlil dvigateli").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.v_engine = tk.StringVar()
        self.cb_engine = ttk.Combobox(par, textvariable=self.v_engine, state="readonly", width=30)
        self.cb_engine.grid(row=2, column=0, columnspan=2, pady=2)
        ttk.Label(par, text="Tahlil ishchilari (oqim/jarayon)").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(par, from_=1, to=max(1, ps.CPU_COUNT), textvariable=self.v_workers,
                    width=6).grid(row=3, column=1)
        self._refresh_engines()

        btns = ttk.Frame(left)
        btns.pack(fill="x")
        self.btn_start = ttk.Button(btns, text="▶  Boshlash", style="Big.TButton",
                                    command=self._start)
        self.btn_start.pack(fill="x")
        self.btn_stop = ttk.Button(btns, text="■  To'xtatish", command=self._stop,
                                   state="disabled")
        self.btn_stop.pack(fill="x", pady=4)
        self.btn_csv = ttk.Button(btns, text="Natijani CSV ga saqlash", command=self._save_csv,
                                  state="disabled")
        self.btn_csv.pack(fill="x")

        prog = ttk.Frame(right)
        prog.pack(fill="x")
        self.v_stage = tk.StringVar(value="Tayyor. Manbani tanlab, «Boshlash» ni bosing.")
        ttk.Label(prog, textvariable=self.v_stage).pack(anchor="w")
        self.pb = ttk.Progressbar(prog, mode="determinate")
        self.pb.pack(fill="x", pady=4)

        stats = ttk.LabelFrame(right, text="Natija", padding=8)
        stats.pack(fill="x")
        self.stat_vars = {}
        names = [("pages", "Sahifalar"), ("mb", "Hajm"), ("t_fetch", "Yuklash vaqti"),
                 ("t_an", "Tahlil vaqti"), ("t_total", "Jami vaqt"), ("rate", "O'tkazuvchanlik"),
                 ("words", "Jami so'zlar"), ("uniq", "Xil so'zlar")]
        for i, (k, lab) in enumerate(names):
            r, c = divmod(i, 4)
            ttk.Label(stats, text=lab, foreground="#555").grid(row=2 * r, column=c, sticky="w",
                                                                padx=10)
            v = tk.StringVar(value="—")
            ttk.Label(stats, textvariable=v, style="Val.TLabel").grid(row=2 * r + 1, column=c,
                                                                      sticky="w", padx=10)
            self.stat_vars[k] = v
        for c in range(4):
            stats.columnconfigure(c, weight=1)

        tv_frame = ttk.Frame(right)
        tv_frame.pack(fill="both", expand=True, pady=(8, 0))
        cols = ("n", "word", "count", "share")
        self.tv = ttk.Treeview(tv_frame, columns=cols, show="headings", height=12)
        for c, t, w, a in (("n", "#", 50, "e"), ("word", "So'z", 260, "w"),
                           ("count", "Soni", 110, "e"), ("share", "Ulush, %", 110, "e")):
            self.tv.heading(c, text=t)
            self.tv.column(c, width=w, anchor=a)
        sb = ttk.Scrollbar(tv_frame, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        self.tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

        self.log = tk.Text(right, height=5, font=("Consolas", 9), state="disabled",
                           background="#f6f6f6")
        self.log.pack(fill="x", pady=(8, 0))
        self._src_changed()

    def _build_cmp_tab(self):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.tab_cmp)
        top.pack(fill="x")
        ttk.Label(top, text="Oxirgi yuklangan korpusda tahlil dvigatellarini solishtirish "
                            "(korpus bo'lmasa demo korpus yaratiladi):").pack(anchor="w")
        self.cmp_vars = {}
        chk = ttk.Frame(top)
        chk.pack(fill="x", pady=4)
        self.cmp_checks = {}
        for i, (k, (title, need, _, _)) in enumerate(ENGINES.items()):
            v = tk.BooleanVar(value=k != "py_seq")
            cb = ttk.Checkbutton(chk, text=title, variable=v)
            cb.grid(row=i // 3, column=i % 3, sticky="w", padx=6)
            self.cmp_vars[k] = v
            self.cmp_checks[k] = cb
        row = ttk.Frame(top)
        row.pack(fill="x")
        ttk.Label(row, text="Takrorlar").pack(side="left")
        self.v_reps = tk.IntVar(value=3)
        ttk.Spinbox(row, from_=1, to=10, textvariable=self.v_reps, width=5).pack(side="left",
                                                                                 padx=4)
        self.btn_cmp = ttk.Button(row, text="▶  Taqqoslash", style="Big.TButton",
                                  command=self._start_compare)
        self.btn_cmp.pack(side="left", padx=10)
        self.v_cmp_stage = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.v_cmp_stage).pack(side="left")

        body = ttk.Frame(self.tab_cmp)
        body.pack(fill="both", expand=True, pady=6)
        cols = ("eng", "mean", "min", "s_py", "s_nb", "ok")
        self.tv_cmp = ttk.Treeview(body, columns=cols, show="headings", height=8)
        for c, t, w, a in (("eng", "Dvigatel", 250, "w"), ("mean", "O'rtacha, ms", 110, "e"),
                           ("min", "Min, ms", 90, "e"), ("s_py", "S (Python ga)", 110, "e"),
                           ("s_nb", "S (Numba ketma-ketga)", 150, "e"),
                           ("ok", "Natija mos", 90, "center")):
            self.tv_cmp.heading(c, text=t)
            self.tv_cmp.column(c, width=w, anchor=a)
        self.tv_cmp.pack(fill="x")
        self.fig_frame = ttk.Frame(body)
        self.fig_frame.pack(fill="both", expand=True, pady=6)
        self.canvas = None

    def _build_sys_tab(self):
        tk = self.tk
        self.txt_sys = tk.Text(self.tab_sys, font=("Consolas", 10), wrap="word")
        self.txt_sys.pack(fill="both", expand=True)
        self._show_sys()

    @ps.muallif(ps.OTABEK, 5)
    def _show_sys(self):
        """Tizim haqida: protsessor, xotira, GPU, kutubxonalar, mualliflar."""
        env = ps.collect_env()
        env.pop("GPU", None)
        lines = [f"{APP_TITLE} - MVP", f"Mualliflar: {AUTHORS}", ""]
        lines += [f"{k:34s} {v}" for k, v in env.items()]
        if ps.HAVE_NUMBA:
            lines.append(f"{'numba':34s} {ps.numba.__version__} "
                         f"(oqimlar: {ps.numba.config.NUMBA_NUM_THREADS})")
        if self.avail["gpu"]:
            lines += [f"{k:34s} {v}" for k, v in ps.gpu_env().items()]
        else:
            lines.append(f"{'GPU (CUDA)':34s} {self.gpu_error or 'tekshirilmoqda...'}")
            if self.gpu_error:
                lines += ["", "GPU diagnostikasi:", ps.gpu_diagnostics()]
        lines += ["", "Dvigatellar:"]
        for k, (title, need, _, _) in ENGINES.items():
            ok = need is None or self.avail.get(need)
            lines.append(f"  [{'+' if ok else '-'}] {title}")
        self.txt_sys.configure(state="normal")
        self.txt_sys.delete("1.0", "end")
        self.txt_sys.insert("1.0", "\n".join(lines))
        self.txt_sys.configure(state="disabled")

    # ------------------------------ yordamchi ------------------------------
    def _src_changed(self):
        demo = self.v_src.get() == "demo"
        for w in self.demo_widgets:
            w.configure(state="normal" if demo else "disabled")
        self.txt_urls.configure(state="disabled" if demo else "normal")
        self.btn_load.configure(state="disabled" if demo else "normal")

    def _refresh_engines(self):
        vals = [title for k, (title, need, _, _) in ENGINES.items()
                if need is None or self.avail.get(need)]
        self.cb_engine.configure(values=vals)
        if self.v_engine.get() not in vals:
            pref = [ENGINES[k][0] for k in ("gpu", "nb_par", "py_proc")
                    if ENGINES[k][0] in vals]
            self.v_engine.set(pref[0] if pref else vals[0])
        for k, cb in getattr(self, "cmp_checks", {}).items():
            need = ENGINES[k][1]
            ok = need is None or self.avail.get(need)
            cb.configure(state="normal" if ok else "disabled")
            if not ok:
                self.cmp_vars[k].set(False)

    def _engine_key(self):
        return next(k for k, v in ENGINES.items() if v[0] == self.v_engine.get())

    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", time.strftime("%H:%M:%S ") + msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _load_urls(self):
        path = self.filedialog.askopenfilename(filetypes=[("Matn", "*.txt"), ("Hammasi", "*.*")])
        if path:
            with open(path, encoding="utf-8") as f:
                self.txt_urls.delete("1.0", "end")
                self.txt_urls.insert("1.0", f.read())

    def _set_busy(self, busy):
        self.busy = busy
        self.btn_start.configure(state="disabled" if busy else "normal")
        self.btn_cmp.configure(state="disabled" if busy else "normal")
        self.btn_stop.configure(state="normal" if busy else "disabled")

    # ------------------------------ ishchi oqimlar -------------------------
    @ps.muallif(ps.OTABEK, 5)
    def _warmup(self):
        """Fon oqimi: Numba va CUDA yadrolarini oldindan kompilyatsiya qiladi
        (JIT 2-3 s) - foydalanuvchi birinchi bosishda kutib qolmasligi uchun."""
        msgs = []
        if ps.HAVE_NUMBA:
            try:
                t0 = time.perf_counter()
                b, o = ps.pack_corpus([ps.generate_page(0, 4)])
                ps.count_words_seq(b, o)
                ps.count_words_parallel(b, o, 2, 0, cyclic=True)
                self.avail["numba"] = True
                msgs.append(f"Numba: tayyor (JIT {time.perf_counter() - t0:.1f} s)")
            except Exception as e:                        # noqa: BLE001
                msgs.append(f"Numba: xato ({e.__class__.__name__})")
        else:
            msgs.append("Numba: o'rnatilmagan")
        self.q.put(("avail", None))
        try:
            if ps.gpu_ok():
                t0 = time.perf_counter()
                b, o = ps.pack_corpus([ps.generate_page(0, 4)])
                ps.gpu_count_words(b, o)
                ps.cuda.synchronize()
                self.avail["gpu"] = True
                self.gpu_name = ps.gpu_env().get("GPU", "GPU")
                msgs.append(f"GPU: {self.gpu_name} (JIT {time.perf_counter() - t0:.1f} s)")
            else:
                self.gpu_error = ("numba.cuda yuklanmadi: " + str(ps.CUDA_IMPORT_ERROR)
                                  if ps.cuda is None else "cuda.is_available() = False")
                msgs.append("GPU: topilmadi")
        except Exception as e:                            # noqa: BLE001
            self.gpu_error = f"{e.__class__.__name__}: {e}"
            msgs.append(f"GPU: ishlamadi ({e.__class__.__name__})")
        self.q.put(("avail", "   ".join(msgs)))

    @ps.muallif(ps.ELBEK, 5)
    def _fetch(self, urls, threads):
        """Sahifalarni oqimlar bilan yuklash; har bir tayyor sahifa haqida
        navbat orqali GUI ga xabar; To'xtatish bosilsa qolganlari bekor qilinadi."""
        pages, failed = {}, 0
        ex = ThreadPoolExecutor(max_workers=threads)
        try:
            futs = [ex.submit(ps.fetch_page, u, **ps.FETCH_CFG) for u in urls]
            for i, f in enumerate(as_completed(futs), 1):
                if self.stop_flag.is_set():
                    break
                url, html, err = f.result()
                if html is None:
                    failed += 1
                else:
                    pages[url] = html
                self.q.put(("progress", i, len(urls)))
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        return [pages[u] for u in urls if u in pages], failed

    @ps.muallif(ps.ELBEK, 5)
    def _job_run(self, cfg):
        """Asosiy ish: yuklash -> tahlil -> natija (alohida oqimda)."""
        server = None
        try:
            if cfg["src"] == "demo":
                self.q.put(("stage", f"Demo server ishga tushmoqda ({cfg['pages']} sahifa)..."))
                server = ps.LocalTestServer(cfg["pages"], cfg["kb"], cfg["delay"] / 1000.0)
                server.__enter__()
                urls = server.urls(cfg["pages"])
                ps.FETCH_CFG["per_domain"] = 64
            else:
                urls = cfg["urls"]
                ps.FETCH_CFG["per_domain"] = 4              # real saytlarga yumshoqroq
            self.q.put(("stage", f"Yuklanmoqda: {len(urls)} sahifa, {cfg['fthreads']} oqim..."))
            t0 = time.perf_counter()
            htmls, failed = self._fetch(urls, cfg["fthreads"])
            t_fetch = time.perf_counter() - t0
            if self.stop_flag.is_set():
                self.q.put(("stopped", None))
                return
            if not htmls:
                raise RuntimeError("Birorta ham sahifa yuklanmadi")
            corpus = Corpus(htmls)
            key = cfg["engine"]
            self.q.put(("stage", f"Tahlil: {ENGINES[key][0]}, {cfg['workers']} ishchi..."))
            self.q.put(("indeterminate", None))
            if key != "py_seq" and key != "py_proc":
                corpus.packed                                # baytlarga o'tkazish (tahlildan oldin)
            words, t_an = run_engine(key, corpus, cfg["workers"])
            self.q.put(("done", {"corpus": corpus, "words": words, "t_fetch": t_fetch,
                                 "t_an": t_an, "failed": failed, "engine": ENGINES[key][0]}))
        except Exception as e:                            # noqa: BLE001
            self.q.put(("error", f"{e.__class__.__name__}: {e}"))
        finally:
            if server is not None:
                server.__exit__(None, None, None)

    @ps.muallif(ps.OTABEK, 5)
    def _job_compare(self, keys, reps, workers):
        """Taqqoslash: har bir dvigatel avval bir marta isitiladi (JIT, kesh),
        keyin reps marta o'lchanadi; natijalar bir-biri bilan solishtiriladi."""
        try:
            if self.corpus is None:
                self.q.put(("cmp_stage", "Demo korpus yaratilmoqda (200 sahifa)..."))
                self.corpus = Corpus([ps.generate_page(i, 100) for i in range(200)])
            c = self.corpus
            c.packed
            rows, ref = [], None
            for k in keys:
                if self.stop_flag.is_set():
                    break
                self.q.put(("cmp_stage", f"{ENGINES[k][0]}: isitish..."))
                words, _ = run_engine(k, c, workers)
                ref = ref or words
                ts = []
                for i in range(reps):
                    self.q.put(("cmp_stage", f"{ENGINES[k][0]}: {i + 1}/{reps}"))
                    w, dt = run_engine(k, c, workers)
                    ts.append(dt)
                rows.append({"key": k, "title": ENGINES[k][0], "mean": sum(ts) / len(ts),
                             "min": min(ts), "ok": words == ref})
                self.q.put(("cmp_rows", list(rows)))
            self.q.put(("cmp_done", {"rows": rows, "mb": c.mb, "pages": len(c.htmls)}))
        except Exception as e:                            # noqa: BLE001
            self.q.put(("error", f"{e.__class__.__name__}: {e}"))

    # ------------------------------ tugmalar -------------------------------
    def _start(self):
        if self.busy:
            return
        try:
            cfg = {"src": self.v_src.get(), "pages": int(self.v_pages.get()),
                   "kb": int(self.v_kb.get()), "delay": int(self.v_delay.get()),
                   "fthreads": max(1, int(self.v_fthreads.get())),
                   "workers": max(1, int(self.v_workers.get())), "engine": self._engine_key()}
        except (ValueError, self.tk.TclError):
            self.messagebox.showerror(APP_TITLE, "Sozlamalarda son kiritilishi kerak")
            return
        if cfg["src"] == "urls":
            cfg["urls"] = [u.strip() for u in self.txt_urls.get("1.0", "end").splitlines()
                           if u.strip().startswith("http")]
            if not cfg["urls"]:
                self.messagebox.showwarning(APP_TITLE, "Kamida bitta http(s) manzil kiriting")
                return
        self.stop_flag.clear()
        self._set_busy(True)
        self.pb.configure(mode="determinate", value=0)
        self.tv.delete(*self.tv.get_children())
        for v in self.stat_vars.values():
            v.set("—")
        self._log(f"Boshlandi: {ENGINES[cfg['engine']][0]}, manba: {cfg['src']}")
        self.t_start = time.perf_counter()
        threading.Thread(target=self._job_run, args=(cfg,), daemon=True).start()

    def _stop(self):
        self.stop_flag.set()
        self.v_stage.set("To'xtatilmoqda...")

    def _start_compare(self):
        if self.busy:
            return
        keys = [k for k, v in self.cmp_vars.items() if v.get()]
        if not keys:
            self.messagebox.showwarning(APP_TITLE, "Kamida bitta dvigatelni tanlang")
            return
        self.stop_flag.clear()
        self._set_busy(True)
        self.tv_cmp.delete(*self.tv_cmp.get_children())
        threading.Thread(target=self._job_compare,
                         args=(keys, max(1, int(self.v_reps.get())),
                               max(1, int(self.v_workers.get()))), daemon=True).start()

    @ps.muallif(ps.ELBEK, 5)
    def _save_csv(self):
        """Barcha so'zlar chastotasini CSV ga yozish."""
        if not self.last_words:
            return
        path = self.filedialog.asksaveasfilename(defaultextension=".csv",
                                                 initialfile="sozlar_chastotasi.csv",
                                                 filetypes=[("CSV", "*.csv")])
        if not path:
            return
        total = sum(self.last_words.values())
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["sozlar", "soni", "ulush_foiz"])
            for word, c in sorted(self.last_words.items(), key=lambda x: -x[1]):
                w.writerow([word, c, f"{100 * c / total:.4f}"])
        self._log(f"Saqlandi: {path}")

    # ------------------------------ navbat ---------------------------------
    @ps.muallif(ps.BOTH, 5)
    def _poll(self):
        """Ishchi oqimlardan kelgan xabarlarni GUI da ko'rsatish (asosiy oqim)."""
        try:
            while True:
                self._dispatch(self.q.get_nowait())
        except queue.Empty:
            pass
        except Exception as e:                            # noqa: BLE001 - GUI yiqilmasin
            self._log(f"GUI xatosi: {e}")
        self.root.after(100, self._poll)

    def _dispatch(self, item):
        kind = item[0]
        if kind == "progress":
            _, i, n = item
            self.pb.configure(mode="determinate", maximum=n, value=i)
            self.v_stage.set(f"Yuklanmoqda: {i} / {n} sahifa")
        else:
            getattr(self, "_on_" + kind)(item[1])

    def _on_avail(self, msg):
        self._refresh_engines()
        if msg:
            self.status.set(msg)
            self._log(msg)
            self._show_sys()

    def _on_stage(self, msg):
        self.v_stage.set(msg)

    def _on_indeterminate(self, _):
        self.pb.configure(mode="indeterminate")
        self.pb.start(12)

    def _on_stopped(self, _):
        self.pb.stop()
        self.v_stage.set("To'xtatildi.")
        self._log("Foydalanuvchi to'xtatdi")
        self._set_busy(False)

    def _on_error(self, msg):
        self.pb.stop()
        self.v_stage.set("Xato: " + msg)
        self._log("XATO: " + msg)
        self._set_busy(False)
        self.messagebox.showerror(APP_TITLE, msg)

    def _on_done(self, r):
        self.pb.stop()
        self.pb.configure(mode="determinate", value=self.pb["maximum"])
        c, words = r["corpus"], r["words"]
        self.corpus, self.last_words = c, words
        total = sum(words.values())
        t_total = time.perf_counter() - self.t_start
        sv = self.stat_vars
        sv["pages"].set(f"{len(c.htmls)}" + (f" (xato: {r['failed']})" if r["failed"] else ""))
        sv["mb"].set(f"{c.mb:.1f} MB")
        sv["t_fetch"].set(f"{r['t_fetch']:.2f} s")
        sv["t_an"].set(f"{1000 * r['t_an']:.1f} ms")
        sv["t_total"].set(f"{t_total:.2f} s")
        sv["rate"].set(f"{len(c.htmls) / t_total:.1f} sahifa/s")
        sv["words"].set(f"{total:,}".replace(",", " "))
        sv["uniq"].set(f"{len(words):,}".replace(",", " "))
        for i, (w, n) in enumerate(sorted(words.items(), key=lambda x: -x[1])[:200], 1):
            self.tv.insert("", "end", values=(i, w, f"{n:,}".replace(",", " "),
                                              f"{100 * n / total:.3f}"))
        self.v_stage.set(f"Tayyor: {r['engine']} - tahlil {1000 * r['t_an']:.1f} ms, "
                         f"jami {t_total:.2f} s")
        self._log(f"Tayyor: {len(c.htmls)} sahifa, {c.mb:.1f} MB, yuklash {r['t_fetch']:.2f} s, "
                  f"tahlil {1000 * r['t_an']:.1f} ms ({r['engine']})")
        self.btn_csv.configure(state="normal")
        self._set_busy(False)

    def _on_cmp_stage(self, msg):
        self.v_cmp_stage.set(msg)

    def _on_cmp_rows(self, rows):
        self.tv_cmp.delete(*self.tv_cmp.get_children())
        t_py = next((r["mean"] for r in rows if r["key"] == "py_seq"), None)
        t_nb = next((r["mean"] for r in rows if r["key"] == "nb_seq"), None)
        for r in rows:
            self.tv_cmp.insert("", "end", values=(
                r["title"], f"{1000 * r['mean']:.1f}", f"{1000 * r['min']:.1f}",
                f"{t_py / r['mean']:.1f}x" if t_py else "—",
                f"{t_nb / r['mean']:.2f}x" if t_nb else "—", "ha" if r["ok"] else "YO'Q"))

    @ps.muallif(ps.OTABEK, 5)
    def _on_cmp_done(self, d):
        """Taqqoslash grafigi (matplotlib, Tk ichida)."""
        self._set_busy(False)
        rows = d["rows"]
        self.v_cmp_stage.set(f"Tayyor: {d['pages']} sahifa, {d['mb']:.1f} MB")
        if not rows:
            return
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        fig = Figure(figsize=(8, 3.4), dpi=100)
        ax = fig.add_subplot(111)
        rr = sorted(rows, key=lambda r: -r["mean"])
        bars = ax.barh([r["title"] for r in rr], [1000 * r["mean"] for r in rr], color="#2f6db5")
        for b, r in zip(bars, rr):
            ax.annotate(f"{1000 * r['mean']:.1f} ms", (b.get_width(), b.get_y() + b.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlim(right=ax.get_xlim()[1] * 4)
        ax.set_xlabel("Tahlil vaqti, ms (logarifmik shkala)")
        ax.set_title(f"{d['pages']} sahifa ({d['mb']:.1f} MB) - kamroq yaxshi", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=.3, which="both")
        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._log("Taqqoslash tayyor: " + ", ".join(f"{r['title']} {1000 * r['mean']:.1f} ms"
                                                    for r in rows))

    def _on_close(self):
        self.stop_flag.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


@ps.muallif(ps.BOTH, 5)
def main():
    """Kirish nuqtasi: argument bo'lsa - buyruq qatori rejimi, aks holda GUI."""
    mp.freeze_support()                                    # exe va Windows (spawn) uchun
    if sys.stdout is None:                                 # --windowed exe
        sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")
    if len(sys.argv) > 1:
        sys.exit(ps.main(sys.argv[1:]))
    App().run()


if __name__ == "__main__":
    main()
