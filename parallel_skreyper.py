#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 PARALLEL VEB-SKREYPER VA MATN TAHLILI TIZIMI
 "Parallel dasturlash" (PAD2504) - 2-topshiriq, B yo'nalishi (loyiha)
 Mavzular: A3 - oqimlar va jarayonlar; A4 - PPL uslubi (parallel_invoke,
           parallel_reduce)

 Bajardi : Boltayev Elbek, Soyibjonov Otabek
 Tekshirdi: Musayev Musabek Ural o'g'li
-------------------------------------------------------------------------------
 MUALLIFLAR HISSASI (har bir funksiya ustida @muallif(...) belgisi bor):
   Boltayev Elbek    - tarmoq qismi (I/O): lokal sinov serveri, sahifalarni
                       yuklash, ThreadPool/ProcessPool bilan yuklash,
                       sinxronizatsiya (Lock, Semaphore, Queue), konveyerning
                       ishlab chiqaruvchi qismi, parallel_invoke.
   Soyibjonov Otabek - tahlil qismi (CPU): HTML tozalash va so'z sanash,
                       ProcessPool/ThreadPool bilan tahlil, parallel_reduce,
                       profiling, to'g'rilikni tekshirish, jadval/grafik/hisobot,
                       konveyerning iste'molchi qismi.
-------------------------------------------------------------------------------
 TOPSHIRIQ BANDLARI -> KODDAGI JOYI
   1. Profiling ......................... exp_profiling()        (cProfile)
   2. Bazaviy o'lchov (3 xil yuklama) ... exp_baseline()
   3. Oqimlar va jarayonlar 1,2,4,8 ..... exp_fetch_workers(), exp_analyze_workers()
   4. CPU-bound / IO-bound + GIL ........ exp_profiling() (CPU/wall nisbati) + 3-band
   5. Sinxronizatsiya ................... PoliteGate, FetchStats, race_condition_demo(),
                                          konveyer (queue.Queue + sentinel)
   6. parallel_invoke (3 vazifa) ........ parallel_invoke(), exp_invoke()
   7. parallel_reduce (2,4,8,16 bo'lak) . parallel_reduce(), exp_reduce()
   8. To'g'rilikni tekshirish ........... check(), run_self_tests()  (--test)
   9. Xulosa: jadval + grafik ........... write_report(), make_charts()
   + Loyiha slaydidagi 4 versiya (ketma-ket / oqimlar / jarayonlar / gibrid
     konveyer) ......................... exp_versions()

 ISHGA TUSHIRISH
   pip install requests beautifulsoup4 matplotlib
   python parallel_skreyper.py              # to'liq tajriba (~2-4 daqiqa)
   python parallel_skreyper.py --quick      # tez sinov
   python parallel_skreyper.py --test       # faqat avtomatik testlar
   python parallel_skreyper.py --delay 1.0  # slayddagi modelga yaqin (1 s tarmoq)
   python parallel_skreyper.py --urls urls.txt   # real saytlar (har qatorda URL)
 Natijalar: natijalar/ papkasida (natijalar.md, natijalar.json, *.csv, *.png)

 3-TOPSHIRIQ (A5 - Numba/OpenMP, A6 - MPI)
   pip install numpy numba mpi4py
   Windows: Microsoft MPI (msmpisetup.exe) o'rnatiladi -> mpiexec paydo bo'ladi
   python parallel_skreyper.py --test3       # tez testlar (MPI p=4 ham)
   python parallel_skreyper.py --task3       # barcha o'lchovlar -> natijalar3/
   mpiexec -n 4 python parallel_skreyper.py --mpi-worker   # MPI ni qo'lda ishlatish
 Natijalar: natijalar3/ (natijalar3.md, natijalar3.json, barcha_natijalar3.csv,
 4 ta grafik). korpus_*.npz fayllari katta - ZIP arxivga qo'shilmaydi.

 4-TOPSHIRIQ (A7 - CUDA, A8 - geterogen CPU + GPU)
   NVIDIA GPU + drayver kerak:  pip install "numba-cuda[cu12]"
   GPU bo'lmasa - Google Colab (Runtime -> T4 GPU), batafsil: COLAB_HELP
   python parallel_skreyper.py --test4       # GPU natijasi == CPU natijasi
   python parallel_skreyper.py --task4       # barcha o'lchovlar -> natijalar4/

 5-TOPSHIRIQ (A9 - yakuniy qiyosiy tahlil) va MVP
   python parallel_skreyper.py --task5       # yig'ma tahlil -> natijalar5/ (~10 daqiqa)
   python skreyper_app.py                    # desktop ilova (GUI)
   build_exe.bat                             # ParallelSkreyper.exe ni yig'ish

 Izoh: o'lchovlar takrorlanuvchan bo'lishi uchun standart holatda lokal sinov
 serveri (alohida jarayonda, sun'iy kechikish bilan) ishlatiladi - natija
 internet holatiga bog'liq bo'lmaydi. Keyingi bosqichda (desktop ilova) GUI
 shu fayldagi run_all() funksiyasini chaqiradi.
===============================================================================
"""

import argparse
import cProfile
import csv
import functools
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import platform
import pstats
import queue
import random
import re
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import requests                      # HTTP so'rovlar (bo'lmasa urllib ishlatiladi)
except ImportError:                      # pragma: no cover
    requests = None
try:
    from bs4 import BeautifulSoup        # HTML tozalash (bo'lmasa HTMLParser)
except ImportError:                      # pragma: no cover
    BeautifulSoup = None

# Lokal server so'rovlari proksi orqali ketmasligi uchun (bola jarayonlar ham meros oladi)
_np = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
if "127.0.0.1" not in _np:
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = (_np + ",127.0.0.1,localhost").strip(",")

USER_AGENT = "TATU-ParallelScraper/1.0 (o'quv loyihasi)"
SENTINEL = None                          # konveyerda "ma'lumot tugadi" belgisi
CPU_COUNT = os.cpu_count() or 1
LOG = print                              # GUI ulanganda boshqa funksiyaga almashtiriladi


# =============================================================================
#  UMUMIY: mualliflik reyestri (hisobotdagi "hissa" jadvali shundan olinadi)
# =============================================================================
AUTHORS = {"Boltayev Elbek": [], "Soyibjonov Otabek": [], "Ikkalasi": []}   # 2-topshiriq
AUTHORS3 = {"Boltayev Elbek": [], "Soyibjonov Otabek": [], "Ikkalasi": []}  # 3-topshiriq
AUTHORS4 = {"Boltayev Elbek": [], "Soyibjonov Otabek": [], "Ikkalasi": []}  # 4-topshiriq
AUTHORS5 = {"Boltayev Elbek": [], "Soyibjonov Otabek": [], "Ikkalasi": []}  # 5-topshiriq + MVP


def muallif(name, task=2):
    """Funksiya muallifini reyestrga yozadi (task - qaysi topshiriq uchun);
    funksiyaning o'zini o'zgartirmaydi (pickle va ProcessPool bilan muammo yo'q)."""
    def deco(fn):
        {2: AUTHORS, 3: AUTHORS3, 4: AUTHORS4, 5: AUTHORS5}[task][name].append(fn.__qualname__)
        return fn
    return deco


ELBEK, OTABEK, BOTH = "Boltayev Elbek", "Soyibjonov Otabek", "Ikkalasi"


# #############################################################################
#                    QISM A - BOLTAYEV ELBEK: TARMOQ (I/O)
# #############################################################################

# ---------------------------- Lokal sinov serveri ----------------------------
_SYL = ["ka", "lo", "mi", "ra", "to", "su", "ne", "bi", "do", "ya", "za", "qu",
        "sha", "cho", "ri", "vo", "gu", "pe", "xo", "ti", "na", "ul", "ar", "ib"]
_BASE_WORDS = ["dastur", "parallel", "oqim", "jarayon", "talaba", "kitob", "maktab",
               "universitet", "kompyuter", "tarmoq", "sahifa", "matn", "tahlil",
               "hisob", "natija", "vaqt", "server", "ma'lumot", "navbat", "qulf",
               "yadro", "protsessor", "xotira", "tezlik", "algoritm", "funksiya",
               "bilan", "uchun", "lekin", "yoki", "ham", "esa", "juda", "bir", "har"]


@functools.lru_cache(maxsize=None)                  # lug'at bir marta yasaladi
@muallif(ELBEK)
def build_vocabulary(n_synthetic=3000):
    """Sinov sahifalari uchun lug'at: haqiqiy so'zlar + bo'g'inlardan yasalgan
    sun'iy so'zlar. Zipf taqsimotiga o'xshash og'irliklar qaytariladi."""
    rng = random.Random(42)
    words = list(_BASE_WORDS)
    seen = set(words)
    while len(words) < len(_BASE_WORDS) + n_synthetic:
        w = "".join(rng.choice(_SYL) for _ in range(rng.randint(2, 4)))
        if w not in seen:
            seen.add(w)
            words.append(w)
    weights = [1.0 / (i + 1) for i in range(len(words))]
    return tuple(words), tuple(itertools.accumulate(weights))   # jamlangan og'irliklar


@muallif(ELBEK)
def generate_page(page_id, size_kb=40):
    """Berilgan id uchun DETERMINISTIK HTML sahifa yasaydi (har safar bir xil):
    sarlavha, paragraflar, havolalar va tahlilda tashlab yuborilishi kerak
    bo'lgan <script>/<style> bloklari."""
    words, cum_weights = build_vocabulary()
    rng = random.Random(1_000_003 * page_id + 12345)
    title = " ".join(rng.choices(words[:200], k=4))
    head = (f"<!DOCTYPE html><html lang='uz'><head><meta charset='utf-8'>"
            f"<title>Sahifa {page_id}: {title}</title>"
            f"<style>body{{font-family:sans-serif}} .stilsoz{{color:red}}</style>"
            f"<script>var yashirin = 'skriptsoz skriptsoz';</script></head>"
            f"<body><h1>{title}</h1>\n")
    parts, size, target = [head], len(head), size_kb * 1024
    while size < target:
        para = " ".join(rng.choices(words, cum_weights=cum_weights, k=rng.randint(30, 80)))
        link = rng.randrange(200)
        chunk = (f"<p>{para} <a href='/page/{link}'>{rng.choice(words[:100])}</a>.</p>\n")
        parts.append(chunk)
        size += len(chunk)
    parts.append("<script>console.log('skriptsoz');</script></body></html>")
    return "".join(parts)


ROBOTS_TXT = "User-agent: *\nDisallow: /private/\n"


@muallif(ELBEK)
def make_handler(pages, delay):
    """HTTP so'rov ishlovchisi sinfini yasaydi. Har bir sahifa so'rovi `delay`
    soniya kutadi - bu real tarmoq kechikishini taqlid qiladi."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/robots.txt":
                return self._send(200, ROBOTS_TXT.encode(), "text/plain")
            m = re.fullmatch(r"/page/(\d+)", self.path)
            if not m or int(m.group(1)) not in pages:
                return self._send(404, b"not found", "text/plain")
            time.sleep(delay)                      # sun'iy tarmoq kechikishi
            self._send(200, pages[int(m.group(1))], "text/html; charset=utf-8")

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):              # konsolni ifloslantirmaslik
            pass
    return Handler


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256
    allow_reuse_address = True


@muallif(ELBEK)
def _server_process_main(port_q, n_pages, size_kb, delay, sizes=None):
    """Server ALOHIDA JARAYONDA ishlaydi: shunda server oqimlari o'lchanayotgan
    mijoz jarayonining GIL iga ta'sir qilmaydi. sizes - har bir sahifa hajmi (KB)."""
    pages = {i: generate_page(i, sizes[i] if sizes else size_kb).encode("utf-8")
             for i in range(n_pages)}
    srv = _QuietServer(("127.0.0.1", 0), make_handler(pages, delay))
    port_q.put(srv.server_address[1])
    srv.serve_forever()


class LocalTestServer:
    """Kontekst menejer: `with LocalTestServer(...) as srv:` - serverni ishga
    tushiradi va oxirida to'xtatadi."""

    @muallif(ELBEK)
    def __init__(self, n_pages, size_kb=40, delay=0.05, sizes=None):
        self.n_pages, self.size_kb, self.delay = n_pages, size_kb, delay
        self.sizes = sizes                        # 3-topshiriq: har xil hajmli sahifalar
        self.proc, self.port = None, None

    def __enter__(self):
        q = mp.Queue()
        self.proc = mp.Process(target=_server_process_main,
                               args=(q, self.n_pages, self.size_kb, self.delay,
                                     self.sizes),
                               daemon=True)
        self.proc.start()
        self.port = q.get(timeout=600)
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            self.proc.join(5)

    def urls(self, n, start=0):
        """[start, start+n) oralig'idagi sahifalar URL ro'yxati."""
        return [f"http://127.0.0.1:{self.port}/page/{i}" for i in range(start, start + n)]


# ------------------------- Sinxronizatsiya vositalari ------------------------
class PoliteGate:
    """Saytlarga hurmat + sinxronizatsiya (5-band):
      * robots.txt keshi  - umumiy lug'at, threading.Lock bilan himoyalangan;
      * domen semafori    - bitta domenga bir vaqtda ko'pi bilan `per_domain`
                            ta so'rov (BoundedSemaphore).
    Deadlock oldini olish: Lock ichida hech qachon tarmoq so'rovi qilinmaydi va
    boshqa qulf olinmaydi (bitta qulf, qisqa kritik seksiya, `with` bloki)."""

    @muallif(ELBEK)
    def __init__(self, per_domain=16):
        self.per_domain = per_domain
        self._lock = threading.Lock()
        self._robots = {}
        self._sems = {}

    @muallif(ELBEK)
    def allowed(self, url):
        """URL ni robots.txt bo'yicha yuklashga ruxsat bormi? (keshlanadi)"""
        p = urllib.parse.urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        with self._lock:
            rp = self._robots.get(base)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            try:                                   # tarmoq so'rovi QULFDAN TASHQARIDA
                rp.parse(_http_get(base + "/robots.txt", 5).splitlines())
            except Exception:
                rp.parse([])                       # robots.txt yo'q -> hammasiga ruxsat
            with self._lock:                       # ikki oqim baravar yuklasa, birinchisi qoladi
                rp = self._robots.setdefault(base, rp)
        return rp.can_fetch(USER_AGENT, url)

    @muallif(ELBEK)
    def slot(self, url):
        """Domen uchun semafor (with bilan ishlatiladi). Lug'atga yangi semafor
        qo'shish ham poyga holatiga olib kelmasligi uchun Lock ostida."""
        domain = urllib.parse.urlparse(url).netloc
        with self._lock:
            sem = self._sems.get(domain)
            if sem is None:
                sem = self._sems[domain] = threading.BoundedSemaphore(self.per_domain)
        return sem


class FetchStats:
    """Oqimlar orasida UMUMIY hisoblagich (muvaffaqiyatli/xato/bayt).
    Har bir yangilash Lock ostida - aks holda `+=` poyga holatida yo'qoladi."""

    @muallif(ELBEK)
    def __init__(self):
        self._lock = threading.Lock()
        self.ok = self.failed = self.bytes = 0

    @muallif(ELBEK)
    def record(self, html, err):
        """Bitta yuklash natijasini qayd etadi (thread-safe)."""
        with self._lock:
            if err is None:
                self.ok += 1
                self.bytes += len(html)
            else:
                self.failed += 1

    def snapshot(self):
        with self._lock:
            return {"ok": self.ok, "failed": self.failed, "bytes": self.bytes}


STATS = FetchStats()
_GATE = None
_GATE_LOCK = threading.Lock()
_TLS = threading.local()                  # har bir oqimning o'z HTTP sessiyasi


@muallif(ELBEK)
def _get_gate(per_domain):
    """Jarayon bo'yicha yagona PoliteGate (lazy, Lock bilan yaratiladi)."""
    global _GATE
    with _GATE_LOCK:
        if _GATE is None or _GATE.per_domain != per_domain:
            _GATE = PoliteGate(per_domain)
        return _GATE


@muallif(ELBEK)
def _http_get(url, timeout):
    """Bitta GET so'rov. requests.Session oqimlar orasida bo'lishilmaydi -
    har bir oqim threading.local() orqali o'z sessiyasiga ega."""
    if requests is not None:
        s = getattr(_TLS, "session", None)
        if s is None:
            s = _TLS.session = requests.Session()
        r = s.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        if not r.encoding:
            r.encoding = "utf-8"
        return r.text
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(resp.headers.get_content_charset() or "utf-8", "replace")


@muallif(ELBEK)
def fetch_page(url, timeout=10, retries=2, per_domain=16):
    """Bitta sahifani yuklaydi: robots.txt tekshiruvi, domen semafori, timeout va
    eksponensial kutish bilan qayta urinish. Natija: (url, html | None, xato | None).
    Xato istisno sifatida emas, qiymat sifatida qaytadi - bitta sekin/buzuq sayt
    butun dasturni to'xtatmaydi."""
    gate = _get_gate(per_domain)
    if not gate.allowed(url):
        return url, None, "robots.txt ruxsat bermadi"
    last = None
    for attempt in range(retries + 1):
        try:
            with gate.slot(url):
                return url, _http_get(url, timeout), None
        except Exception as e:                     # noqa: BLE001
            last = e
            time.sleep(0.2 * (2 ** attempt))
    return url, None, repr(last)


@muallif(ELBEK)
def _fetch_tracked(url, timeout=10, retries=2, per_domain=16):
    """fetch_page + umumiy STATS hisoblagichini oqim ichida yangilash."""
    res = fetch_page(url, timeout, retries, per_domain)
    STATS.record(res[1], res[2])
    return res


FETCH_CFG = {"timeout": 10, "retries": 2, "per_domain": 16}


@muallif(ELBEK)
def fetch_sequential(urls):
    """KETMA-KET yuklash (bazaviy versiya)."""
    out = []
    for u in urls:
        res = fetch_page(u, **FETCH_CFG)
        STATS.record(res[1], res[2])
        out.append(res)
    return out


@muallif(ELBEK)
def fetch_parallel(urls, workers, backend="thread"):
    """PARALLEL yuklash: backend='thread' -> ThreadPoolExecutor,
    'process' -> ProcessPoolExecutor. Natija tartibi URL tartibida (map)."""
    if backend == "thread":
        fn = functools.partial(_fetch_tracked, **FETCH_CFG)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(fn, urls))
    fn = functools.partial(fetch_page, **FETCH_CFG)   # partial pickle qilinadi
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(fn, urls))
    for r in res:                                       # jarayonlarda STATS alohida,
        STATS.record(r[1], r[2])                        # shuning uchun asosiy jarayonda
    return res


@muallif(ELBEK)
def race_condition_demo(n_threads=8, n_incr=3000):
    """Poyga holatini ko'rsatish: Lock'siz hisoblagich yangilanishlarni
    yo'qotadi, Lock bilan esa natija har doim to'g'ri.
    time.sleep(0) - "o'qish" va "yozish" orasida oqim almashinuvini majburlaydi."""
    class Box:
        value = 0
    lock = threading.Lock()

    def unsafe(box):
        for _ in range(n_incr):
            v = box.value
            time.sleep(0)
            box.value = v + 1

    def safe(box):
        for _ in range(n_incr):
            with lock:
                v = box.value
                time.sleep(0)
                box.value = v + 1

    result = {"expected": n_threads * n_incr}
    for name, fn in (("lock_siz", unsafe), ("lock_bilan", safe)):
        box = Box()
        ts = [threading.Thread(target=fn, args=(box,)) for _ in range(n_threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        result[name] = box.value
    return result


# --------------------- Konveyer: ishlab chiqaruvchi (Elbek) -------------------
@muallif(ELBEK)
def pipeline_producer(urls, q, k):
    """Konveyerning 1-bosqichi: k ta oqim sahifalarni yuklaydi va tayyor
    bo'lganini darhol navbatga qo'yadi (as_completed). Navbat to'lsa q.put()
    kutadi (backpressure) - xotira cheklanadi. `finally` dagi SENTINEL xato
    bo'lsa ham albatta yuboriladi - iste'molchi abadiy kutib qolmaydi (deadlock yo'q)."""
    try:
        fn = functools.partial(_fetch_tracked, **FETCH_CFG)
        with ThreadPoolExecutor(max_workers=k) as ex:
            for fut in as_completed([ex.submit(fn, u) for u in urls]):
                url, html, err = fut.result()
                if html is not None:
                    q.put((url, html))
    finally:
        q.put(SENTINEL)


# ----------------------- parallel_invoke (A4) va vazifalar -------------------
@muallif(ELBEK)
def parallel_invoke(*tasks, backend="thread"):
    """PPL dagi parallel_invoke analogi: mustaqil vazifalarni bir vaqtda ishga
    tushiradi va hammasi tugashini kutadi. tasks: (funksiya, argumentlar) juftlari."""
    Ex = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    with Ex(max_workers=len(tasks)) as ex:
        futs = [ex.submit(fn, *args) for fn, args in tasks]
        return [f.result() for f in futs]          # xato bo'lsa shu yerda chiqadi


@muallif(ELBEK)
def task_download(urls):
    """1-vazifa (I/O): yangi sahifalar to'plamini yuklash."""
    return [(u, h) for u, h, e in fetch_sequential(urls) if h is not None]


@muallif(ELBEK)
def task_page_stats(htmls):
    """3-vazifa (CPU): sahifalar statistikasi va havolalar grafi -
    sarlavhalar, havolalar soni, eng ko'p havola qilingan sahifalar."""
    in_deg, total_links, total_chars, titles = Counter(), 0, 0, []
    for h in htmls:
        m = re.search(r"<title>(.*?)</title>", h, re.S)
        titles.append(m.group(1) if m else "")
        links = re.findall(r"href=['\"]([^'\"]+)['\"]", h)
        total_links += len(links)
        in_deg.update(links)
        total_chars += len(html_to_text(h))
    return {"pages": len(htmls), "total_links": total_links,
            "avg_text_chars": round(total_chars / max(1, len(htmls)), 1),
            "top_linked": in_deg.most_common(5),
            "titles_sha1": hashlib.sha1("\n".join(titles).encode()).hexdigest()}


# #############################################################################
#                 QISM B - SOYIBJONOV OTABEK: TAHLIL (CPU)
# #############################################################################
TOKEN_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|')+[^\W\d_]")   # kamida 3 harf, ' ichida
STOP_WORDS = {"bilan", "uchun", "lekin", "yoki", "ham", "esa", "juda", "bir", "har",
              "the", "and", "for", "with", "this", "that"}


class _TextExtractor(HTMLParser):
    """bs4 o'rnatilmagan bo'lsa ishlatiladigan zaxira HTML tozalagich."""

    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


@muallif(OTABEK)
def html_to_text(html):
    """HTML dan toza matn: <script>, <style> olib tashlanadi (CPU-bound qism)."""
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
        return soup.get_text(" ")
    p = _TextExtractor()
    p.feed(html)
    return " ".join(p.parts)


@muallif(OTABEK)
def analyze_html(html):
    """MAP qadami: bitta sahifa -> so'zlar chastotasi (Counter).
    Umumiy holatga yozmaydi, natijani QAYTARADI - shuning uchun qulf kerak emas."""
    words = TOKEN_RE.findall(html_to_text(html).lower())
    return Counter(w for w in words if w not in STOP_WORDS)


@muallif(OTABEK)
def analyze_sequential(htmls):
    """KETMA-KET tahlil (bazaviy versiya)."""
    total = Counter()
    for h in htmls:
        total.update(analyze_html(h))
    return total


@muallif(OTABEK)
def analyze_parallel(htmls, workers, backend="process"):
    """PARALLEL tahlil: map - har bir sahifa alohida worker'da, reduce -
    asosiy jarayonda Counter'larni birlashtirish."""
    total = Counter()
    if backend == "thread":
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for c in ex.map(analyze_html, htmls):
                total.update(c)
    else:
        chunk = max(1, len(htmls) // (workers * 4))    # pickle xarajatini kamaytirish
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for c in ex.map(analyze_html, htmls, chunksize=chunk):
                total.update(c)
    return total


@muallif(OTABEK)
def task_build_index(htmls):
    """2-vazifa (CPU): teskari indeks qurish - so'z -> uni o'z ichiga olgan
    sahifalar raqamlari (qidiruv tizimi uchun)."""
    index = defaultdict(set)
    for pid, h in enumerate(htmls):
        for w in analyze_html(h):
            index[w].add(pid)
    return {w: sorted(s) for w, s in sorted(index.items())}


# ----------------------------- parallel_reduce (A4) --------------------------
@muallif(OTABEK)
def split_chunks(seq, k):
    """Ro'yxatni k ta deyarli teng, uzluksiz bo'lakka bo'ladi."""
    k = max(1, min(k, len(seq)))
    size, rem = divmod(len(seq), k)
    out, start = [], 0
    for i in range(k):
        end = start + size + (1 if i < rem else 0)
        out.append(seq[start:end])
        start = end
    return out


@muallif(OTABEK)
def count_chunk(htmls):
    """Bitta bo'lak uchun qisman natija (bo'lakdagi barcha sahifalar Counter'i)."""
    c = Counter()
    for h in htmls:
        c.update(analyze_html(h))
    return c


@muallif(OTABEK)
def merge_counters(a, b):
    """Birlashtiruvchi (assotsiativ) amal: ikki qisman natijani qo'shadi."""
    a.update(b)
    return a


@muallif(OTABEK)
def parallel_reduce(data, chunks, map_chunk=count_chunk, combine=merge_counters,
                    identity=Counter, backend="process"):
    """PPL dagi parallel_reduce analogi: ma'lumot `chunks` bo'lakka bo'linadi,
    har bo'lak parallel hisoblanadi, qisman natijalar `combine` bilan
    birlashtiriladi. Reduce qadami ketma-ket - Amdal qonunidagi f ulush."""
    parts = split_chunks(data, chunks)
    workers = min(len(parts), CPU_COUNT) if backend == "process" else len(parts)
    Ex = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    with Ex(max_workers=workers) as ex:
        partials = list(ex.map(map_chunk, parts))
    result = identity()
    for p in partials:
        result = combine(result, p)
    return result


# ------------------------ Konveyer: iste'molchi (Otabek) ---------------------
@muallif(OTABEK)
def pipeline_consumer(q, p):
    """Konveyerning 2-bosqichi: navbatdan sahifani olib ProcessPool'ga beradi,
    SENTINEL kelganda to'xtaydi va natijalarni birlashtiradi (reduce).
    q.get(timeout=...) - ishlab chiqaruvchi qotib qolsa ham abadiy kutmaydi."""
    total = Counter()
    with ProcessPoolExecutor(max_workers=p) as pp:
        futs = []
        while (item := q.get(timeout=120)) is not SENTINEL:
            futs.append(pp.submit(analyze_html, item[1]))
        for f in as_completed(futs):
            total.update(f.result())
    return total


@muallif(BOTH)
def run_pipeline(urls, k=8, p=None, qsize=64):
    """GIBRID KONVEYER: yuklash (oqimlar) va tahlil (jarayonlar) bir vaqtda.
    Ishlab chiqaruvchi alohida threading.Thread da, iste'molchi asosiy oqimda."""
    p = p or min(8, CPU_COUNT)
    q = queue.Queue(maxsize=qsize)                   # ichida o'z qulfi bor (thread-safe)
    t = threading.Thread(target=pipeline_producer, args=(urls, q, k), daemon=True)
    t.start()
    total = pipeline_consumer(q, p)
    t.join()
    return total


# ---------------------------- O'lchash yordamchilari -------------------------
@muallif(OTABEK)
def measure(fn, repeats=3, warmup=1):
    """Vaqtni o'lchash metodikasi: `warmup` ta birinchi yurish (kesh to'ldirish,
    importlar) hisobga olinmaydi, keyin `repeats` marta o'lchanib o'rtacha olinadi."""
    result = None
    for _ in range(warmup):
        result = fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return {"mean": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "times": times, "result": result}


@muallif(OTABEK)
def cpu_wall(fn):
    """Bosqichning devor vaqti va CPU vaqti. CPU/devor ~ 1 -> CPU-bound,
    CPU/devor << 1 -> IO-bound (vaqt kutishga ketgan)."""
    w0, c0 = time.perf_counter(), time.process_time()
    r = fn()
    return time.perf_counter() - w0, time.process_time() - c0, r


@muallif(OTABEK)
def counter_digest(c):
    """Counter'ning SHA-256 izi (tartibdan qat'i nazar bir xil)."""
    return hashlib.sha256(json.dumps(sorted(c.items()), ensure_ascii=False)
                          .encode()).hexdigest()[:16]


@muallif(OTABEK)
def pages_digest(results):
    """Yuklangan sahifalar to'plamining izi: (url, sha1(html)) URL bo'yicha saralangan."""
    items = sorted((u, hashlib.sha1(h.encode()).hexdigest()) for u, h, *_ in results
                   if h is not None)
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16]


@muallif(OTABEK)
def check(ctx, name, ok, detail=""):
    """To'g'rilik tekshiruvini qayd etadi (8-band) va konsolga chiqaradi."""
    ctx["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    LOG(f"    [{'MOS' if ok else 'XATO'}] {name} {detail}")


# #############################################################################
#                              TAJRIBALAR
# #############################################################################
@muallif(OTABEK)
def exp_profiling(ctx):
    """1-band (profiling) va 4-band (masala turi): ketma-ket konveyer cProfile
    bilan o'lchanadi; har bir bosqich uchun CPU/devor vaqti nisbati olinadi."""
    LOG("\n[1] PROFILING (cProfile) ...")
    urls = ctx["urls"][:min(20, len(ctx["urls"]))]
    fetch_sequential(urls[:2])                      # robots.txt keshi, importlar
    pr = cProfile.Profile()
    pr.enable()
    pages = fetch_sequential(urls)
    analyze_sequential([h for _, h, _ in pages if h])
    pr.disable()
    st = pstats.Stats(pr)
    total = st.total_tt
    rows = []
    for (fname, line, func), (cc, nc, tt, ct, _) in st.stats.items():
        label = func if fname == "~" else f"{os.path.basename(fname)}:{line}({func})"
        rows.append({"func": label, "ncalls": nc, "tottime": tt, "cumtime": ct,
                     "share": 100 * tt / total if total else 0})
    top = sorted(rows, key=lambda r: r["tottime"], reverse=True)[:12]
    stage = {}
    for r in rows:
        for key in ("fetch_sequential", "analyze_sequential"):
            if r["func"].endswith(f"({key})"):
                stage[key] = r["cumtime"]
    # 4-band: CPU/devor nisbati (profilersiz, toza o'lchov)
    fw, fc, pages = cpu_wall(lambda: fetch_sequential(urls))
    htmls = [h for _, h, _ in pages if h]
    aw, ac, _ = cpu_wall(lambda: analyze_sequential(htmls))
    ctx["res"]["profile"] = {
        "pages": len(urls), "total": total, "top": top, "stage": stage,
        "cpu_wall": {"fetch": {"wall": fw, "cpu": fc}, "analyze": {"wall": aw, "cpu": ac}}}
    fs = stage.get("fetch_sequential", 0)
    as_ = stage.get("analyze_sequential", 0)
    LOG(f"    yuklash: {fs:.2f} s ({100 * fs / total:.1f}%), "
        f"tahlil: {as_:.2f} s ({100 * as_ / total:.1f}%), jami {total:.2f} s")
    LOG(f"    CPU/devor: yuklash {fc / fw:.2f}, tahlil {ac / aw:.2f}")


@muallif(OTABEK)
def exp_baseline(ctx):
    """2-band: ketma-ket versiya (yuklash + tahlil) uch xil yuklamada."""
    LOG("\n[2] BAZAVIY O'LCHOV (ketma-ket) ...")
    out = []
    for n in ctx["loads"]:
        urls = ctx["urls"][:n]

        def run():
            pages = fetch_sequential(urls)
            return analyze_sequential([h for _, h, _ in pages if h])
        m = measure(run, ctx["repeats"])
        out.append({"pages": n, "mean": m["mean"], "std": m["std"],
                    "throughput": n / m["mean"], "words": sum(m["result"].values())})
        LOG(f"    n={n:4d}: {m['mean']:.3f} +- {m['std']:.3f} s, {n / m['mean']:.1f} sahifa/s")
    ctx["res"]["baseline"] = out


@muallif(ELBEK)
def exp_fetch_workers(ctx):
    """3-band (yuklash): ThreadPool va ProcessPool, worker = 1, 2, 4, 8."""
    LOG("\n[3a] YUKLASH: oqimlar va jarayonlar ...")
    urls = ctx["main_urls"]
    seq = measure(lambda: fetch_sequential(urls), ctx["repeats"])
    ref = pages_digest(seq["result"])
    rows = [{"backend": "ketma-ket", "workers": 1, "mean": seq["mean"], "std": seq["std"],
             "speedup": 1.0, "eff": 1.0}]
    LOG(f"    ketma-ket: {seq['mean']:.3f} s")
    for backend in ("thread", "process"):
        for w in ctx["workers"]:
            m = measure(lambda: fetch_parallel(urls, w, backend), ctx["repeats"])
            s = seq["mean"] / m["mean"]
            rows.append({"backend": backend, "workers": w, "mean": m["mean"],
                         "std": m["std"], "speedup": s, "eff": s / w})
            LOG(f"    {backend:7s} w={w}: {m['mean']:.3f} s  S={s:.2f}  E={s / w:.2f}")
            if ctx["strict_fetch"]:
                check(ctx, f"yuklash {backend} w={w}", pages_digest(m["result"]) == ref)
    ctx["res"]["fetch_workers"] = rows


@muallif(OTABEK)
def exp_analyze_workers(ctx):
    """3-band (tahlil): ThreadPool va ProcessPool, worker = 1, 2, 4, 8.
    Oqimlarda tezlashish yo'qligi - GIL ta'sirining isboti (4-band)."""
    LOG("\n[3b] TAHLIL: oqimlar va jarayonlar ...")
    htmls = ctx["corpus"]
    seq = measure(lambda: analyze_sequential(htmls), ctx["repeats"])
    ref = seq["result"]
    rows = [{"backend": "ketma-ket", "workers": 1, "mean": seq["mean"], "std": seq["std"],
             "speedup": 1.0, "eff": 1.0}]
    LOG(f"    ketma-ket: {seq['mean']:.3f} s")
    for backend in ("thread", "process"):
        for w in ctx["workers"]:
            m = measure(lambda: analyze_parallel(htmls, w, backend), ctx["repeats"])
            s = seq["mean"] / m["mean"]
            rows.append({"backend": backend, "workers": w, "mean": m["mean"],
                         "std": m["std"], "speedup": s, "eff": s / w})
            LOG(f"    {backend:7s} w={w}: {m['mean']:.3f} s  S={s:.2f}  E={s / w:.2f}")
            check(ctx, f"tahlil {backend} w={w}", m["result"] == ref,
                  counter_digest(m["result"]))
    ctx["res"]["analyze_workers"] = rows


@muallif(ELBEK)
def exp_sync(ctx):
    """5-band: poyga holati namoyishi va umumiy hisoblagich izchilligi."""
    LOG("\n[5] SINXRONIZATSIYA ...")
    race = race_condition_demo()
    LOG(f"    kutilgan={race['expected']}, Lock'siz={race['lock_siz']}, "
        f"Lock bilan={race['lock_bilan']}")
    check(ctx, "Lock bilan hisoblagich to'g'ri", race["lock_bilan"] == race["expected"])
    before = STATS.snapshot()
    res = fetch_parallel(ctx["main_urls"], 8, "thread")
    after = STATS.snapshot()
    got = after["ok"] - before["ok"]
    expected = sum(1 for r in res if r[1] is not None)
    check(ctx, "8 oqimli umumiy STATS hisoblagichi", got == expected,
          f"({got} == {expected})")
    ctx["res"]["sync"] = {"race": race, "stats_delta": got, "stats_expected": expected}


@muallif(ELBEK)
def exp_invoke(ctx):
    """6-band: uchta mustaqil vazifa - yuklash (I/O), indeks qurish (CPU),
    statistika (CPU) - ketma-ket va parallel_invoke uslubida."""
    LOG("\n[6] PARALLEL_INVOKE (3 ta mustaqil vazifa) ...")
    batch, htmls = ctx["batch_urls"], ctx["corpus"]
    tasks = [(task_download, (batch,)), (task_build_index, (htmls,)),
             (task_page_stats, (htmls,))]
    times = {}
    for name in ("task_download", "task_build_index", "task_page_stats"):
        fn, args = next(t for t in tasks if t[0].__name__ == name)
        times[name] = cpu_wall(lambda: fn(*args))[0]
    seq = measure(lambda: [fn(*a) for fn, a in tasks], ctx["repeats"])
    rows = [{"mode": "ketma-ket", "mean": seq["mean"], "std": seq["std"], "speedup": 1.0}]
    LOG(f"    ketma-ket: {seq['mean']:.3f} s  (vazifalar: "
        + ", ".join(f"{k}={v:.2f}s" for k, v in times.items()) + ")")
    ref = seq["result"]
    for backend in ("thread", "process"):
        m = measure(lambda: parallel_invoke(*tasks, backend=backend), ctx["repeats"])
        s = seq["mean"] / m["mean"]
        rows.append({"mode": f"parallel_invoke ({backend})", "mean": m["mean"],
                     "std": m["std"], "speedup": s})
        LOG(f"    {backend:7s}: {m['mean']:.3f} s  S={s:.2f}")
        check(ctx, f"parallel_invoke {backend}", m["result"] == ref)
    ctx["res"]["invoke"] = {"rows": rows, "task_times": times}


@muallif(OTABEK)
def exp_reduce(ctx):
    """7-band: so'zlar chastotasini bo'laklab hisoblash, bo'lak = 2, 4, 8, 16."""
    LOG("\n[7] PARALLEL_REDUCE (bo'laklar soni) ...")
    htmls = ctx["corpus"]
    seq = measure(lambda: count_chunk(htmls), ctx["repeats"])
    ref = seq["result"]
    rows = [{"backend": "ketma-ket", "chunks": 1, "mean": seq["mean"], "std": seq["std"],
             "speedup": 1.0}]
    LOG(f"    ketma-ket: {seq['mean']:.3f} s")
    for backend in ("process", "thread"):
        for k in ctx["chunks"]:
            m = measure(lambda: parallel_reduce(htmls, k, backend=backend), ctx["repeats"])
            s = seq["mean"] / m["mean"]
            rows.append({"backend": backend, "chunks": k, "mean": m["mean"],
                         "std": m["std"], "speedup": s})
            LOG(f"    {backend:7s} bo'lak={k:2d}: {m['mean']:.3f} s  S={s:.2f}")
            check(ctx, f"parallel_reduce {backend} k={k}", m["result"] == ref)
    ctx["res"]["reduce"] = rows


@muallif(BOTH)
def exp_versions(ctx):
    """Loyiha slaydidagi 4 versiya bitta URL ro'yxatida (yuklash + tahlil)."""
    LOG("\n[+] LOYIHANING 4 VERSIYASI (yuklash + tahlil) ...")
    urls, k, p = ctx["main_urls"], 8, min(8, CPU_COUNT)

    def only(pages):
        return [h for _, h, _ in pages if h]
    versions = [
        ("1. Ketma-ket", lambda: analyze_sequential(only(fetch_sequential(urls)))),
        (f"2. Faqat oqimlar (k={k})",
         lambda: analyze_parallel(only(fetch_parallel(urls, k, "thread")), k, "thread")),
        (f"3. Faqat jarayonlar (p={p})",
         lambda: analyze_parallel(only(fetch_parallel(urls, p, "process")), p, "process")),
        (f"4. Gibrid konveyer ({k}+{p})", lambda: run_pipeline(urls, k, p)),
    ]
    rows, ref, t1 = [], None, None
    for name, fn in versions:
        m = measure(fn, ctx["repeats"])
        if ref is None:
            ref, t1 = m["result"], m["mean"]
        s = t1 / m["mean"]
        rows.append({"version": name, "mean": m["mean"], "std": m["std"], "speedup": s,
                     "throughput": len(urls) / m["mean"]})
        LOG(f"    {name:32s}: {m['mean']:.3f} s  S={s:.2f}  {len(urls) / m['mean']:.1f} sahifa/s")
        check(ctx, f"versiya: {name}", m["result"] == ref, counter_digest(m["result"]))
    ctx["res"]["versions"] = rows


# #############################################################################
#                    HISOBOT, JADVAL, GRAFIK (SOYIBJONOV OTABEK)
# #############################################################################
@muallif(OTABEK)
def collect_env():
    """Sinov muhiti: protsessor, yadrolar, xotira, OT, Python va kutubxonalar."""
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "win32":                 # aniq model nomi reyestrdan
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except OSError:
        pass
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
    except OSError:
        pass
    ram = "noma'lum"
    try:
        import psutil
        ram = f"{psutil.virtual_memory().total / 2 ** 30:.1f} GB"
    except Exception:
        try:
            ram = f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30:.1f} GB"
        except (AttributeError, ValueError, OSError):
            pass
    if ram == "noma'lum" and sys.platform == "win32":    # psutil'siz Windows
        try:
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = _MS()
            st.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            ram = f"{st.ullTotalPhys / 2 ** 30:.1f} GB"
        except Exception:
            pass

    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "o'rnatilmagan"
    return {"OT": f"{platform.system()} {platform.release()} ({platform.version()})",
            "Protsessor": cpu, "Mantiqiy yadrolar": CPU_COUNT, "Operativ xotira": ram,
            "GPU": "ishlatilmaydi",
            "Python": f"{platform.python_implementation()} {platform.python_version()}",
            "requests": ver("requests"), "beautifulsoup4": ver("bs4"),
            "matplotlib": ver("matplotlib"),
            "multiprocessing start method": mp.get_start_method()}


@muallif(OTABEK)
def fmt_table(headers, rows):
    """Markdown jadval matni."""
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


@muallif(OTABEK)
def make_charts(res, outdir):
    """9-band: grafiklar (o'q nomlari, birliklar, izohli sarlavha bilan)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOG("    matplotlib yo'q - grafiklar chizilmadi")
        return []
    files = []

    def save(fig, name):
        path = os.path.join(outdir, name)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        files.append(name)

    b = res["baseline"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar([r["pages"] for r in b], [r["mean"] for r in b],
                yerr=[r["std"] for r in b], marker="o", capsize=4)
    ax.set_xlabel("Sahifalar soni, n (dona)")
    ax.set_ylabel("Bajarilish vaqti, T (s)")
    ax.set_title("Ketma-ket versiya: vaqtning yuklamaga bog'liqligi")
    ax.grid(alpha=.3)
    save(fig, "1_bazaviy.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "fetch_workers", "Yuklash (I/O-bound)"),
                           (axes[1], "analyze_workers", "Tahlil (CPU-bound)")):
        rows = res[key]
        ws = sorted({r["workers"] for r in rows if r["backend"] != "ketma-ket"})
        ax.plot(ws, ws, "k--", alpha=.4, label="ideal S = n")
        for backend, lab in (("thread", "ThreadPoolExecutor"),
                             ("process", "ProcessPoolExecutor")):
            pts = [r for r in rows if r["backend"] == backend]
            ax.plot([r["workers"] for r in pts], [r["speedup"] for r in pts],
                    marker="o", label=lab)
        ax.set_xlabel("Worker'lar soni, n")
        ax.set_ylabel("Tezlashish, S = T_ketma-ket / T_n")
        ax.set_title(title)
        ax.set_xticks(ws)
        ax.grid(alpha=.3)
        ax.legend()
    fig.suptitle("Oqimlar va jarayonlar: tezlashish (GIL ta'siri o'ngda)")
    save(fig, "2_tezlashish.png")

    r7 = res["reduce"]
    fig, ax = plt.subplots(figsize=(6, 4))
    seq_t = r7[0]["mean"]
    ax.axhline(seq_t, color="gray", ls="--", label="ketma-ket")
    for backend, lab in (("process", "jarayonlar (ProcessPool)"),
                         ("thread", "oqimlar (ThreadPool)")):
        pts = [r for r in r7 if r["backend"] == backend]
        ax.plot([r["chunks"] for r in pts], [r["mean"] for r in pts], marker="o", label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks([r["chunks"] for r in r7 if r["backend"] == "process"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Bo'laklar soni, k")
    ax.set_ylabel("Bajarilish vaqti, T (s)")
    ax.set_title("parallel_reduce: so'zlar chastotasi")
    ax.grid(alpha=.3)
    ax.legend()
    save(fig, "3_reduce.png")

    for key, fname, title, label in (
            ("invoke", "4_invoke.png", "parallel_invoke: 3 ta mustaqil vazifa", "mode"),
            ("versions", "5_versiyalar.png", "Loyihaning 4 versiyasi", "version")):
        rows = res[key]["rows"] if key == "invoke" else res[key]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(range(len(rows)), [r["mean"] for r in rows],
                      yerr=[r["std"] for r in rows], capsize=4)
        for bar, r in zip(bars, rows):
            ax.annotate(f"{r['speedup']:.2f}x",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height() + r["std"]),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")
        ax.set_ylim(0, max(r["mean"] + r["std"] for r in rows) * 1.15)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[label] for r in rows], rotation=12, fontsize=8)
        ax.set_ylabel("Bajarilish vaqti, T (s)")
        ax.set_title(title + " (ustun ustida - tezlashish)")
        ax.grid(axis="y", alpha=.3)
        save(fig, fname)

    cw = res["profile"]["cpu_wall"]
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["Yuklash", "Tahlil"]
    ax.bar(names, [cw["fetch"]["wall"], cw["analyze"]["wall"]], label="devor vaqti")
    ax.bar(names, [cw["fetch"]["cpu"], cw["analyze"]["cpu"]], width=.4, label="CPU vaqti")
    ax.set_ylabel("Vaqt (s)")
    ax.set_title("CPU va devor vaqti: IO-bound vs CPU-bound")
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    save(fig, "6_cpu_devor.png")
    return files


@muallif(OTABEK)
def conclusions(res):
    """O'lchov natijalaridan avtomatik xulosa matni (9-band)."""
    fw = {(r["backend"], r["workers"]): r for r in res["fetch_workers"]}
    aw = {(r["backend"], r["workers"]): r for r in res["analyze_workers"]}
    wmax = max(w for _, w in fw)
    cw = res["profile"]["cpu_wall"]
    fr = cw["fetch"]["cpu"] / cw["fetch"]["wall"]
    ar = cw["analyze"]["cpu"] / cw["analyze"]["wall"]
    v = res["versions"]
    best_red = max((r for r in res["reduce"] if r["backend"] == "process"),
                   key=lambda r: r["speedup"])
    ft, fp = fw[("thread", wmax)]["speedup"], fw[("process", wmax)]["speedup"]
    at, ap_ = aw[("thread", wmax)]["speedup"], aw[("process", wmax)]["speedup"]
    best_ap = max((r for r in res["analyze_workers"] if r["backend"] == "process"),
                  key=lambda r: r["speedup"])
    out = [
        f"Yuklash bosqichi IO-bound: CPU/devor vaqti nisbati {fr:.2f} - vaqtning "
        f"{100 * (1 - fr):.0f}% qismi server javobini kutishga ketadi. Kutish paytida oqim "
        f"GIL ni bo'shatadi, shuning uchun {wmax} ta oqim S = {ft:.2f} beradi. Jarayonlar "
        f"bilan S = {fp:.2f}: " + ("ular ham tezlashtiradi, lekin jarayon yaratish va "
                                  "pickle xarajati tufayli oqimlardan ustunligi yo'q."
                                  if fp > 1 else "jarayon yaratish xarajati kutishdagi "
                                  "yutuqdan katta - I/O uchun jarayon ortiqcha."),
        f"Tahlil bosqichi CPU-bound: CPU/devor nisbati {ar:.2f}. Bitta jarayonda Python "
        f"baytkodini bir vaqtda faqat bitta oqim bajaradi (GIL), shuning uchun {wmax} ta oqim "
        f"S = {at:.2f} - tezlashish yo'q. Har bir jarayonda o'z GIL i bor: eng yaxshi natija "
        f"{best_ap['workers']} ta jarayonda S = {best_ap['speedup']:.2f}"
        + ((f"; foydali jarayonlar soni yadrolar soni ({CPU_COUNT}) bilan cheklanadi, undan "
            f"ko'pida ({wmax} ta: S = {ap_:.2f}) faqat qo'shimcha xarajat ortadi."
            if best_ap["workers"] < wmax else
            f"; ideal {wmax}x dan uzoqligiga jarayon yaratish (spawn) va HTML ni pickle "
            f"qilib uzatish xarajati sabab bo'ladi.")
           if best_ap["speedup"] > 1 else
           ". Bu yuklamada tahlil hajmi juda kichik - jarayon yaratish va HTML ni pickle "
           "qilish xarajati yutuqdan katta; --pages yoki --page-kb ni oshirib ko'ring."),
        f"parallel_reduce: eng yaxshi natija k = {best_red['chunks']} bo'lakda "
        f"(S = {best_red['speedup']:.2f}). " + (
            "Bo'laklar soni oshgani sari yuk yadrolar orasida tekisroq taqsimlanadi; bu "
            "hajmda qo'shimcha xarajat hali yutuqni yo'qqa chiqarmadi, lekin har bir yangi "
            "bo'lak pickle va ketma-ket reduce (Amdal qonuni) ulushini oshiradi."
            if best_red["chunks"] == max(r["chunks"] for r in res["reduce"]) else
            "Undan ko'p bo'lakda qo'shimcha xarajat (pickle, ko'p qisman natijani ketma-ket "
            "birlashtirish - Amdal qonuni) yutuqni kamaytiradi."),
        f"Loyiha uchun eng mos model - gibrid: yuklashga oqimlar, tahlilga jarayonlar, "
        f"ular orasida chegaralangan queue.Queue. Gibrid konveyer ketma-ket versiyadan "
        f"{v[-1]['speedup']:.2f} baravar tez ({v[-1]['throughput']:.1f} sahifa/s); "
        f"faqat oqimlar versiyasi: {v[1]['speedup']:.2f} baravar.",
    ]
    return out


@muallif(OTABEK)
def write_report(ctx, outdir):
    """Natijalarni natijalar.json, barcha_natijalar.csv va natijalar.md ga yozadi."""
    os.makedirs(outdir, exist_ok=True)
    res = ctx["res"]
    charts = make_charts(res, outdir)
    with open(os.path.join(outdir, "natijalar.json"), "w", encoding="utf-8") as f:
        json.dump({"env": ctx["env"], "params": ctx["params"], "results": res,
                   "checks": ctx["checks"], "authors": AUTHORS}, f,
                  ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(outdir, "barcha_natijalar.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["tajriba", "rejim", "parametr", "T_ortacha_s", "std_s", "tezlashish_S",
                    "samaradorlik_E"])
        for r in res["baseline"]:
            w.writerow(["bazaviy", "ketma-ket", f"n={r['pages']}", f"{r['mean']:.4f}",
                        f"{r['std']:.4f}", "", ""])
        for key in ("fetch_workers", "analyze_workers"):
            for r in res[key]:
                w.writerow([key, r["backend"], f"w={r['workers']}", f"{r['mean']:.4f}",
                            f"{r['std']:.4f}", f"{r['speedup']:.3f}", f"{r['eff']:.3f}"])
        for r in res["invoke"]["rows"]:
            w.writerow(["invoke", r["mode"], "3 vazifa", f"{r['mean']:.4f}",
                        f"{r['std']:.4f}", f"{r['speedup']:.3f}", ""])
        for r in res["reduce"]:
            w.writerow(["reduce", r["backend"], f"k={r['chunks']}", f"{r['mean']:.4f}",
                        f"{r['std']:.4f}", f"{r['speedup']:.3f}", ""])
        for r in res["versions"]:
            w.writerow(["versiyalar", r["version"], "", f"{r['mean']:.4f}",
                        f"{r['std']:.4f}", f"{r['speedup']:.3f}", ""])

    p, P = res["profile"], ctx["params"]
    md = ["# Parallel veb-skreyper va matn tahlili tizimi - 2-topshiriq natijalari",
          "", "**Yo'nalish:** B (loyiha). **Bajardi:** Boltayev Elbek, Soyibjonov Otabek.",
          f"**Sana:** {time.strftime('%Y-%m-%d %H:%M')}", "",
          "## Jamoa a'zolarining hissasi", "",
          fmt_table(["A'zo", "Funksiyalar"],
                    [[k, ", ".join(f"`{n}`" for n in v)] for k, v in AUTHORS.items()]),
          "", "## Sinov muhiti", "",
          fmt_table(["Parametr", "Qiymat"], [[k, v] for k, v in ctx["env"].items()]),
          "", "## O'lchash metodikasi", "",
          f"- Ma'lumot manbai: {P['source']}; tarmoq kechikishi {P['delay']} s/sahifa, "
          f"sahifa hajmi ~{P['page_kb']} KB.",
          f"- Har bir o'lchov {P['repeats']} marta takrorlanib o'rtacha olinadi; "
          "**birinchi (qizdirish) yurishi hisobga olinmaydi** - unda importlar, robots.txt "
          "keshi va OT kesh to'ladi.",
          "- Tezlashish S = T_ketma-ket / T_n, samaradorlik E = S / n.",
          "", f"## 1. Profiling (cProfile, {p['pages']} sahifa, ketma-ket)", "",
          fmt_table(["Funksiya", "chaqiruvlar", "tottime, s", "cumtime, s", "ulush, %"],
                    [[f"`{r['func']}`", r["ncalls"], f"{r['tottime']:.3f}",
                      f"{r['cumtime']:.3f}", f"{r['share']:.1f}"] for r in p["top"]]),
          "", f"Bosqichlar: yuklash {p['stage'].get('fetch_sequential', 0):.2f} s, "
          f"tahlil {p['stage'].get('analyze_sequential', 0):.2f} s (jami {p['total']:.2f} s). "
          "Eng ko'p vaqt soket kutishiga (`recv_into`) va HTML parsingga ketadi - shu ikki "
          "qism parallellashtirish uchun tanlandi.",
          "", "## 2. Bazaviy o'lchov (ketma-ket)", "",
          fmt_table(["Sahifalar", "T, s", "std, s", "sahifa/s"],
                    [[r["pages"], f"{r['mean']:.3f}", f"{r['std']:.3f}",
                      f"{r['throughput']:.1f}"] for r in res["baseline"]])]
    for key, title in (("fetch_workers", "## 3a. Yuklash: oqimlar va jarayonlar"),
                       ("analyze_workers", "## 3b. Tahlil: oqimlar va jarayonlar")):
        md += ["", title, "", fmt_table(
            ["Rejim", "Worker", "T, s", "std, s", "S", "E"],
            [[r["backend"], r["workers"], f"{r['mean']:.3f}", f"{r['std']:.3f}",
              f"{r['speedup']:.2f}", f"{r['eff']:.2f}"] for r in res[key]])]
    cw = p["cpu_wall"]
    md += ["", "## 4. Masala turi (CPU-bound / IO-bound) va GIL", "",
           fmt_table(["Bosqich", "Devor vaqti, s", "CPU vaqti, s", "CPU/devor", "Tur"],
                     [["Yuklash", f"{cw['fetch']['wall']:.3f}", f"{cw['fetch']['cpu']:.3f}",
                       f"{cw['fetch']['cpu'] / cw['fetch']['wall']:.2f}", "IO-bound"],
                      ["Tahlil", f"{cw['analyze']['wall']:.3f}", f"{cw['analyze']['cpu']:.3f}",
                       f"{cw['analyze']['cpu'] / cw['analyze']['wall']:.2f}", "CPU-bound"]]),
           "", "## 5. Sinxronizatsiya", "",
           "- `queue.Queue(maxsize=64)` - oqimlar va iste'molchi orasidagi yagona kanal; "
           "ichki qulfi bor, to'lsa `put()` kutadi (xotira cheklanadi).",
           "- `FetchStats` - umumiy hisoblagich, har bir yangilash `threading.Lock` ostida.",
           "- `PoliteGate` - robots.txt keshi (Lock) va domen bo'yicha `BoundedSemaphore`.",
           "- `threading.local()` - har bir oqimning o'z HTTP sessiyasi (bo'lishilmaydi).",
           "- Jarayonlar umumiy xotiraga yozmaydi - natijani qaytaradi (qulf kerak emas).",
           "- Deadlock oldini olish: bitta qulf, qisqa kritik seksiya, qulf ichida tarmoq "
           "so'rovi yo'q; SENTINEL `finally` da yuboriladi; `q.get(timeout=...)`.",
           f"- Poyga holati namoyishi: kutilgan {res['sync']['race']['expected']}, Lock'siz "
           f"{res['sync']['race']['lock_siz']}, Lock bilan {res['sync']['race']['lock_bilan']}.",
           "", "## 6. parallel_invoke: 3 ta mustaqil vazifa", "",
           "Vazifalar: " + ", ".join(f"`{k}` = {v:.2f} s" for k, v in
                                      res["invoke"]["task_times"].items()), "",
           fmt_table(["Rejim", "T, s", "std, s", "S"],
                     [[r["mode"], f"{r['mean']:.3f}", f"{r['std']:.3f}", f"{r['speedup']:.2f}"]
                      for r in res["invoke"]["rows"]]),
           "", "## 7. parallel_reduce: bo'laklar soni", "",
           fmt_table(["Rejim", "Bo'laklar", "T, s", "std, s", "S"],
                     [[r["backend"], r["chunks"], f"{r['mean']:.3f}", f"{r['std']:.3f}",
                       f"{r['speedup']:.2f}"] for r in res["reduce"]]),
           "", "## Loyihaning 4 versiyasi", "",
           fmt_table(["Versiya", "T, s", "std, s", "S", "sahifa/s"],
                     [[r["version"], f"{r['mean']:.3f}", f"{r['std']:.3f}",
                       f"{r['speedup']:.2f}", f"{r['throughput']:.1f}"] for r in res["versions"]]),
           "", "## 8. To'g'rilikni tekshirish", "",
           "Har bir parallel natija ketma-ket natija bilan solishtirildi: so'zlar chastotasi "
           "`Counter ==` (har bir so'z va soni), yuklangan sahifalar esa (url, sha1(html)) "
           "izlari bo'yicha.", "",
           fmt_table(["Tekshiruv", "Natija"],
                     [[c["name"], "MOS" if c["ok"] else "**XATO**"] for c in ctx["checks"]]),
           "", "## 9. Xulosa", ""]
    md += [f"{i}. {t}" for i, t in enumerate(conclusions(res), 1)]
    md += ["", "## Grafiklar", ""] + [f"![{c}]({c})" for c in charts]
    with open(os.path.join(outdir, "natijalar.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return charts


# #############################################################################
#   3-TOPSHIRIQ: UMUMIY XOTIRA (A5, Numba = OpenMP uslubi) VA TAQSIMLANGAN
#   XOTIRA (A6, MPI / mpi4py) MODELLARI
# -----------------------------------------------------------------------------
#   2-topshiriqda parallellashtirilgan qism - so'zlar chastotasini sanash
#   (HTML tozalash + so'zlarga ajratish + sanash) - endi Numba bilan mashina
#   kodiga kompilyatsiya qilinadigan yadro sifatida yozildi va:
#     * umumiy xotirada  : @njit(parallel=True) + prange (OpenMP parallel for)
#     * taqsimlangan     : mpi4py - Scatterv / Gatherv / allreduce
#   modellarida bajariladi.
#   Bandlar: 1-JIT o'lchovdan chiqariladi .... t3_exp_numba()
#            2-oqimlar 1,2,4,8 ............... t3_exp_numba()
#            3-static / dynamic .............. t3_exp_schedule()
#            4-MPI, >= 4 jarayon ............. mpi_worker(), mpi_count_once()
#            5-almashinuv ulushi ............. t3_exp_mpi()
#            6-kuchli / zaif masshtablanish .. t3_exp_mpi(), t3_exp_numba_weak()
#            7-to'g'rilik .................... same_counts(), t3_check_python()
#            8-xulosa ........................ t3_conclusions(), write_report3()
# #############################################################################
import shutil
import subprocess

try:
    import numpy as np
except ImportError:                                  # pragma: no cover
    np = None

if np is not None:
    # Moslik qatlami: NumPy 2.4+ da olib tashlangan funksiyalar. Ba'zi numba-cuda
    # versiyalari ularni import paytida chaqiradi (AttributeError: row_stack).
    # Har biri alohida funksiya - numba'dagi asl np.vstack va boshqalar bilan
    # to'qnashmaydi.
    def _np_row_stack(tup, **kw):
        return np.vstack(tup, **kw)

    def _np_trapz(*a, **kw):
        return np.trapezoid(*a, **kw)

    def _np_in1d(ar1, ar2, *a, **kw):
        return np.isin(ar1, ar2, *a, **kw).ravel()
    for _name, _fn in (("row_stack", _np_row_stack), ("trapz", _np_trapz), ("in1d", _np_in1d)):
        if not hasattr(np, _name):
            setattr(np, _name, _fn)

try:
    import numba
    from numba import njit, prange
    try:
        from numba import get_thread_id as _thread_id          # numba >= 0.57
    except ImportError:                                         # pragma: no cover
        from numba.np.ufunc.parallel import _get_thread_id as _thread_id
    HAVE_NUMBA = True
except ImportError:
    # Numba o'rnatilmagan bo'lsa ham kod ishlaydi (oddiy Python sifatida, juda
    # sekin) - faqat --test3 dagi mantiqiy tekshiruvlar uchun.
    HAVE_NUMBA = False
    numba = None
    prange = range

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

    def _thread_id():
        return 0

    if np is not None:
        np.seterr(over="ignore")       # Numba'siz rejim: uint64 ko'paytmasi kutilgan holda o'ralib ketadi

if np is not None:
    FNV_OFFSET = np.uint64(0xCBF29CE484222325)      # FNV-1a 64 bitli xesh
    FNV_PRIME = np.uint64(0x100000001B3)
    _T_SCRIPT = np.frombuffer(b"script", dtype=np.uint8).copy()
    _T_STYLE = np.frombuffer(b"style", dtype=np.uint8).copy()
    _T_NOSCRIPT = np.frombuffer(b"noscript", dtype=np.uint8).copy()


# ======================= SOYIBJONOV OTABEK: Numba yadrosi =====================
@muallif(OTABEK, 3)
def fnv1a(word):
    """So'zning FNV-1a xeshi (Python'da) - Numba yadrosidagi bilan bir xil."""
    h = 0xCBF29CE484222325
    for b in word.lower().encode("utf-8"):
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h or 1


@muallif(OTABEK, 3)
def stop_hashes():
    """To'xtash so'zlarining xeshlari (yadroga massiv sifatida beriladi)."""
    return np.array(sorted(fnv1a(w) for w in STOP_WORDS), dtype=np.uint64)


@njit(cache=False)
@muallif(OTABEK, 3)
def _is_alpha(c):
    """ASCII harfmi?"""
    return (c >= 65 and c <= 90) or (c >= 97 and c <= 122)


@njit(cache=False)
@muallif(OTABEK, 3)
def _lower(c):
    """ASCII katta harfni kichikka o'tkazadi."""
    if c >= 65 and c <= 90:
        return c + 32
    return c


@njit(cache=False)
@muallif(OTABEK, 3)
def _name_is(buf, a, n, word):
    """buf[a:a+n] (kichik harflarda) word ga tengmi? (teg nomini tekshirish)"""
    if n != word.shape[0]:
        return False
    for t in range(n):
        if _lower(buf[a + t]) != word[t]:
            return False
    return True


@njit(cache=False)
@muallif(OTABEK, 3)
def _skip_tag(buf, i, e):
    """'<' dan boshlangan tegni o'tkazib yuboradi va keyingi pozitsiyani qaytaradi.
    <script>, <style>, <noscript> bo'lsa - yopuvchi tegigacha hammasini o'tkazadi
    (BeautifulSoup'dagi decompose() bilan bir xil natija). <!-- izoh --> ham."""
    j = i + 1
    if j + 2 < e and buf[j] == 33 and buf[j + 1] == 45 and buf[j + 2] == 45:
        j += 3
        while j + 2 < e:
            if buf[j] == 45 and buf[j + 1] == 45 and buf[j + 2] == 62:
                return j + 3
            j += 1
        return e
    closing = False
    if j < e and buf[j] == 47:
        closing = True
        j += 1
    ns = j
    while j < e and _is_alpha(buf[j]):
        j += 1
    nl = j - ns
    while j < e and buf[j] != 62:
        j += 1
    j += 1
    if j > e:
        j = e
    raw = (_name_is(buf, ns, nl, _T_SCRIPT) or _name_is(buf, ns, nl, _T_STYLE)
           or _name_is(buf, ns, nl, _T_NOSCRIPT))
    if closing or not raw:
        return j
    while j + 1 < e:                                # "</script" ni qidirish
        if buf[j] == 60 and buf[j + 1] == 47 and j + 2 + nl <= e:
            ok = True
            for t in range(nl):
                if _lower(buf[j + 2 + t]) != _lower(buf[ns + t]):
                    ok = False
                    break
            if ok and (j + 2 + nl == e or not _is_alpha(buf[j + 2 + nl])):
                k = j + 2 + nl
                while k < e and buf[k] != 62:
                    k += 1
                return k + 1 if k < e else e
        j += 1
    return e


@njit(cache=False)
@muallif(OTABEK, 3)
def _skip_entity(buf, i, e):
    """&amp; &nbsp; &#39; kabi HTML belgilarini ajratuvchi sifatida o'tkazadi."""
    j = i + 1
    n = 0
    while j < e and n < 10 and (_is_alpha(buf[j]) or (buf[j] >= 48 and buf[j] <= 57)
                                or buf[j] == 35):
        j += 1
        n += 1
    if j < e and buf[j] == 59 and n > 0:
        return j + 1
    return i + 1


@njit(cache=False)
@muallif(OTABEK, 3)
def _insert(keys, counts, pos, lens, h, p, ln, add):
    """Ochiq adreslashli xesh-jadvalga qo'shish (chiziqli sinash).
    1 - yangi so'z qo'shildi, 0 - mavjud so'z soni oshirildi."""
    imask = keys.shape[0] - 1
    i = np.int64(h & np.uint64(imask))
    while True:
        k = keys[i]
        if k == h:
            counts[i] += add
            return 0
        if k == 0:
            keys[i] = h
            counts[i] = add
            pos[i] = p
            lens[i] = ln
            return 1
        i = (i + 1) & imask


@njit(cache=False)
@muallif(OTABEK, 3)
def _scan_doc(buf, s, e, stop, keys, counts, pos, lens, used, cap_lim):
    """HISOBLASH YADROSI: bitta sahifani (buf[s:e]) bir marta o'qib chiqadi -
    teglarni tashlaydi, so'zlarni ajratadi (kamida 3 harf, ' ichida bo'lishi
    mumkin), kichik harfga o'tkazadi, xeshlaydi va jadvalda sanaydi.
    2-topshiriqdagi html_to_text + TOKEN_RE + Counter bilan bir xil natija.
    Qaytaradi: (so'zlar soni, jadvaldagi band joylar, to'lib ketdimi)."""
    ntok = 0
    overflow = 0
    run = -1                                      # joriy so'z boshlanishi
    i = s
    while True:
        c = 32
        step = 1
        if i < e:
            c = buf[i]
            if _is_alpha(c) or c == 39:
                if run < 0:
                    run = i
                i += 1
                continue
            if c >= 128:                           # UTF-8: qo'shtirnoq/tire/nbsp - ajratuvchi
                if c == 0xE2 and i + 2 < e and buf[i + 1] == 0x80 and \
                        buf[i + 2] >= 0x93 and buf[i + 2] <= 0x9D:
                    step = 3
                elif c == 0xC2 and i + 1 < e and buf[i + 1] == 0xA0:
                    step = 2
                else:                              # boshqa UTF-8 baytlar - harf
                    if run < 0:
                        run = i
                    i += 1
                    continue
        if run >= 0:                               # so'z tugadi -> sanash
            a = run
            b = i
            while a < b and buf[a] == 39:
                a += 1
            while b > a and buf[b - 1] == 39:
                b -= 1
            if b - a >= 3:
                h = FNV_OFFSET
                for j in range(a, b):
                    h = (h ^ np.uint64(_lower(buf[j]))) * FNV_PRIME
                if h == 0:
                    h = np.uint64(1)
                is_stop = False
                for t in range(stop.shape[0]):
                    if stop[t] == h:
                        is_stop = True
                        break
                if not is_stop:
                    ntok += 1
                    if used < cap_lim:
                        used += _insert(keys, counts, pos, lens, h, a, b - a, 1)
                    else:
                        overflow = 1
            run = -1
        if i >= e:
            break
        if c == 60:
            nxt = buf[i + 1] if i + 1 < e else 32
            if _is_alpha(nxt) or nxt == 47 or nxt == 33:
                i = _skip_tag(buf, i, e)
            else:
                i += 1
        elif c == 38:
            i = _skip_entity(buf, i, e)
        else:
            i += step
    return ntok, used, overflow


@njit(nogil=True, cache=False)
@muallif(OTABEK, 3)
def _count_seq(buf, offs, stop, keys, counts, pos, lens, cap_lim, ntok_doc):
    """KETMA-KET versiya: oddiy range sikli, bitta xesh-jadval."""
    used = 0
    overflow = 0
    for d in range(offs.shape[0] - 1):
        nt, used, ov = _scan_doc(buf, offs[d], offs[d + 1], stop, keys, counts,
                                 pos, lens, used, cap_lim)
        ntok_doc[d] = nt
        if ov:
            overflow = 1
    return used, overflow


@njit(parallel=True, nogil=True, cache=False)       # nogil: GPU oqimi bilan bir vaqtda ishlash uchun
@muallif(OTABEK, 3)
def _count_parallel(buf, offs, stop, keys, counts, pos, lens, used, cap_lim,
                    work, ntok_doc, overflow):
    """UMUMIY XOTIRA versiyasi (OpenMP '#pragma omp parallel for' analogi):
    prange sahifalarni oqimlarga taqsimlaydi. Poyga holati bo'lmasligi uchun
    har bir oqim FAQAT o'z qatoriga (keys[t], counts[t] ...) yozadi - bu
    OpenMP'dagi reduction / threadprivate g'oyasi. work[t] - oqim qayta ishlagan
    baytlar (yuklama taqsimotini ko'rish uchun)."""
    for d in prange(offs.shape[0] - 1):
        t = _thread_id()
        s = offs[d]
        e = offs[d + 1]
        nt, u, ov = _scan_doc(buf, s, e, stop, keys[t], counts[t], pos[t], lens[t],
                              used[t], cap_lim)
        used[t] = u
        ntok_doc[d] = nt
        work[t] += e - s
        if ov:
            overflow[t] = 1


@njit(parallel=True, nogil=True, cache=False)
@muallif(OTABEK, 3)
def _count_parallel_cyclic(buf, offs, stop, keys, counts, pos, lens, used, cap_lim,
                           work, ntok_doc, overflow, nthreads):
    """SIKLIK static rejalashtirish (OpenMP 'schedule(static, 1)'): t-oqim
    t, t+n, t+2n, ... sahifalarni oladi. Taqsimot oldindan ma'lum (static), lekin
    katta sahifalar oqimlar orasida aralashib ketadi."""
    n = offs.shape[0] - 1
    for t in prange(nthreads):
        for d in range(t, n, nthreads):
            s = offs[d]
            e = offs[d + 1]
            nt, u, ov = _scan_doc(buf, s, e, stop, keys[t], counts[t], pos[t], lens[t],
                                  used[t], cap_lim)
            used[t] = u
            ntok_doc[d] = nt
            work[t] += e - s
            if ov:
                overflow[t] = 1


@njit(nogil=True, cache=False)
@muallif(OTABEK, 3)
def _merge_entries(src_keys, src_counts, src_pos, src_lens, cap):
    """Qisman jadvallarni bitta jadvalga birlashtirish (REDUCE, ketma-ket qism)."""
    keys = np.zeros(cap, np.uint64)
    counts = np.zeros(cap, np.int64)
    pos = np.zeros(cap, np.int64)
    lens = np.zeros(cap, np.int64)
    for i in range(src_keys.shape[0]):
        if src_keys[i] != 0:
            _insert(keys, counts, pos, lens, src_keys[i], src_pos[i], src_lens[i],
                    src_counts[i])
    return keys, counts, pos, lens


@muallif(OTABEK, 3)
def _next_pow2(n):
    """n dan katta yoki teng eng kichik 2 ning darajasi."""
    p = 1
    while p < n:
        p <<= 1
    return p


@muallif(OTABEK, 3)
def _compact(keys, counts, pos, lens, ntok_doc):
    """Jadvalning band joylarini olib, xesh bo'yicha saralaydi - natija
    tartibi oqimlar/jarayonlar soniga bog'liq bo'lmaydi (solishtirish uchun)."""
    m = keys != 0
    k = keys[m]
    order = np.argsort(k, kind="stable")
    return {"keys": k[order], "counts": counts[m][order], "pos": pos[m][order],
            "lens": lens[m][order], "ntok_doc": ntok_doc,
            "total": int(ntok_doc.sum())}


@muallif(OTABEK, 3)
def count_words_seq(buf, offs, cap=1 << 16):
    """Ketma-ket Numba versiyasi (range). Jadval to'lsa 4 marta kattalashtirib
    qaytadan hisoblaydi."""
    stop = stop_hashes()
    while True:
        keys = np.zeros(cap, np.uint64)
        counts = np.zeros(cap, np.int64)
        pos = np.zeros(cap, np.int64)
        lens = np.zeros(cap, np.int64)
        ntok = np.zeros(len(offs) - 1, np.int64)
        used, ov = _count_seq(buf, offs, stop, keys, counts, pos, lens, cap // 2, ntok)
        if not ov:
            return _compact(keys, counts, pos, lens, ntok)
        cap *= 4


@muallif(OTABEK, 3)
def count_words_parallel(buf, offs, threads, chunksize=0, cap=1 << 16, cyclic=False):
    """Umumiy xotira versiyasi. threads - numba.set_num_threads; chunksize=0 -
    STATIC rejalashtirish (iteratsiyalar oqimlarga teng bloklarga bo'linadi),
    chunksize>0 - DYNAMIC (shu o'lchamdagi bo'laklar bo'shagan oqimga beriladi),
    cyclic=True - siklik static (schedule(static,1)).
    Qaytaradi: natija + bosqichlar vaqti (scan - parallel, merge - ketma-ket)."""
    stop = stop_hashes()
    T = numba.config.NUMBA_NUM_THREADS if HAVE_NUMBA else 1
    old = None
    if HAVE_NUMBA:
        numba.set_num_threads(min(threads, T))
        old = numba.set_parallel_chunksize(chunksize)
    try:
        while True:
            t0 = time.perf_counter()
            keys = np.zeros((T, cap), np.uint64)
            counts = np.zeros((T, cap), np.int64)
            pos = np.zeros((T, cap), np.int64)
            lens = np.zeros((T, cap), np.int64)
            used = np.zeros(T, np.int64)
            work = np.zeros(T, np.int64)
            ov = np.zeros(T, np.int64)
            ntok = np.zeros(len(offs) - 1, np.int64)
            t1 = time.perf_counter()
            if cyclic:
                _count_parallel_cyclic(buf, offs, stop, keys, counts, pos, lens, used,
                                       cap // 2, work, ntok, ov, min(threads, T))
            else:
                _count_parallel(buf, offs, stop, keys, counts, pos, lens, used, cap // 2,
                                work, ntok, ov)
            t2 = time.perf_counter()
            if ov.sum() == 0:
                break
            cap *= 4
    finally:
        if HAVE_NUMBA:
            numba.set_parallel_chunksize(old)
    m = keys != 0
    mk, mc, mp_, ml = _merge_entries(keys[m], counts[m], pos[m], lens[m],
                                     _next_pow2(max(1024, 2 * int(m.sum()))))
    t3 = time.perf_counter()
    res = _compact(mk, mc, mp_, ml, ntok)
    res["work"] = work
    res["times"] = {"alloc": t1 - t0, "scan": t2 - t1, "merge": t3 - t2}
    return res


@muallif(OTABEK, 3)
def same_counts(a, b, rtol=1e-12):
    """7-band: ikki natija mosmi? Butun sonlar (har bir so'z soni, jami) aniq
    teng bo'lishi, nisbiy chastotalar esa rtol aniqlik chegarasida mos kelishi
    kerak. Qaytaradi: (mos, nisbiy chastotalardagi eng katta farq)."""
    if len(a["keys"]) != len(b["keys"]) or a["total"] != b["total"]:
        return False, float("inf")
    if not (np.array_equal(a["keys"], b["keys"]) and np.array_equal(a["counts"], b["counts"])):
        return False, float("inf")
    fa = a["counts"] / max(1, a["total"])
    fb = b["counts"] / max(1, b["total"])
    diff = float(np.max(np.abs(fa - fb))) if len(fa) else 0.0
    return bool(np.allclose(fa, fb, rtol=rtol, atol=0.0)), diff


@muallif(OTABEK, 3)
def decode_words(res, buf):
    """Xeshlarni so'zlarga qaytaradi: har bir so'z birinchi uchragan joyidan
    (pos, lens) o'qiladi. Natija: {so'z: soni}."""
    out = {}
    for p, ln, c in zip(res["pos"].tolist(), res["lens"].tolist(), res["counts"].tolist()):
        w = bytes(buf[p:p + ln]).decode("utf-8", "replace").lower()
        out[w] = out.get(w, 0) + c
    return out


# ===================== BOLTAYEV ELBEK: korpus va MPI qismi ====================
@muallif(ELBEK, 3)
def corpus_sizes(n, kind):
    """Sahifa hajmlari (KB). 'bir_xil' - hammasi 100 KB; 'notekis' - real
    saytlardagidek og'ir dumli (Pareto, alfa=1,1) taqsimot: ko'p kichik sahifa
    va bir nechta juda katta sahifa (8 KB ... 1,5 MB)."""
    if kind == "bir_xil":
        return [100] * n
    rng = random.Random(2026)
    return [int(min(1500, 8 * rng.paretovariate(1.1))) for _ in range(n)]


@muallif(ELBEK, 3)
def pack_corpus(htmls):
    """Sahifalar ro'yxatini bitta uint8 massiv + chegaralar (offs) ko'rinishiga
    keltiradi: i-sahifa = buf[offs[i]:offs[i+1]]. Numba va MPI aynan shu
    ko'rinish bilan ishlaydi (Python obyektlari emas, uzluksiz xotira)."""
    enc = [h.encode("utf-8") for h in htmls]
    offs = np.zeros(len(enc) + 1, np.int64)
    offs[1:] = np.cumsum([len(b) for b in enc])
    buf = np.frombuffer(bytearray(b"".join(enc)), dtype=np.uint8)
    return buf, offs


@muallif(ELBEK, 3)
def build_corpus(n, kind, path):
    """Korpusni loyihaning o'z skreyperi bilan yig'adi: lokal serverdan
    fetch_parallel (16 oqim) orqali yuklab, .npz faylga saqlaydi. Fayl bor
    bo'lsa qayta yuklamaydi."""
    if os.path.exists(path):
        with np.load(path) as d:                  # with - fayl darhol yopiladi (Windows)
            if len(d["offs"]) - 1 == n:
                return d["buf"].copy(), d["offs"].copy()
    LOG(f"    korpus yig'ilmoqda: {n} sahifa ({kind}) ...")
    with LocalTestServer(n, delay=0.0, sizes=corpus_sizes(n, kind)) as srv:
        res = fetch_parallel(srv.urls(n), 16, "thread")
    bad = [u for u, h, _ in res if h is None]
    if bad:
        raise RuntimeError(f"{len(bad)} ta sahifa yuklanmadi")
    buf, offs = pack_corpus([h for _, h, _ in res])
    np.savez(path, buf=buf, offs=offs)
    return buf, offs


@muallif(ELBEK, 3)
def split_docs_by_bytes(offs, parts):
    """Sahifalarni `parts` ta UZLUKSIZ guruhga bo'ladi, har bir guruhda baytlar
    soni taxminan teng bo'ladi (MPI uchun statik yuklama taqsimoti).
    Qaytaradi: hujjat chegaralari [0, ..., n]."""
    n = len(offs) - 1
    total = int(offs[-1])
    plan = [0]
    for r in range(1, parts):
        plan.append(max(plan[-1], min(n, int(np.searchsorted(offs, total * r / parts)))))
    plan.append(n)
    return plan


@muallif(ELBEK, 3)
def mpi_count_once(comm, MPI, buf, offs, plan):
    """MPI bilan bir marta to'liq hisoblash. Bosqichlar va vaqtlari (MPI.Wtime):
      1) taqsimlash : Bcast (reja) + Scatterv (baytlar) + Scatterv (chegaralar)
      2) hisoblash  : har bir jarayon o'z bo'lagini count_words_seq bilan sanaydi
      3) yig'ish    : gather (o'lchamlar) + Gatherv (xesh, son, joy, uzunlik)
                      + allreduce(SUM) - jami so'zlar soni
      4) birlashtirish: root jarayonda qisman jadvallarni bitta jadvalga."""
    rank, size = comm.Get_rank(), comm.Get_size()
    comm.Barrier()
    t0 = MPI.Wtime()
    meta = np.zeros((4, size), np.int64)
    rel = None
    if rank == 0:
        for r in range(size):
            meta[0, r] = offs[plan[r + 1]] - offs[plan[r]]        # baytlar soni
            meta[1, r] = offs[plan[r]]                            # baytlar siljishi
            meta[2, r] = plan[r + 1] - plan[r] + 1                # chegaralar soni
        meta[3, 1:] = np.cumsum(meta[2])[:-1]                     # chegaralar siljishi
        rel = np.concatenate([offs[plan[r]:plan[r + 1] + 1] - offs[plan[r]]
                              for r in range(size)]).astype(np.int64)
    comm.Bcast(meta, root=0)
    local_buf = np.empty(int(meta[0, rank]), np.uint8)
    local_offs = np.empty(int(meta[2, rank]), np.int64)
    comm.Scatterv([buf, meta[0].tolist(), meta[1].tolist(), MPI.UNSIGNED_CHAR]
                  if rank == 0 else None, local_buf, root=0)
    comm.Scatterv([rel, meta[2].tolist(), meta[3].tolist(), MPI.INT64_T]
                  if rank == 0 else None, local_offs, root=0)
    t1 = MPI.Wtime()

    res = count_words_seq(local_buf, local_offs)
    t2 = MPI.Wtime()
    comm.Barrier()                       # eng sekin jarayonni kutish hisoblashga yoziladi
    t2b = MPI.Wtime()

    n_loc = len(res["keys"])
    sizes = comm.gather(n_loc, root=0)
    total = comm.allreduce(res["total"], op=MPI.SUM)
    gk = gc = gp = gl = None
    spec = [None, None]
    if rank == 0:
        cnt = list(sizes)
        dsp = [0] + list(np.cumsum(cnt)[:-1].tolist())
        m = int(sum(cnt))
        gk, gc = np.empty(m, np.uint64), np.empty(m, np.int64)
        gp, gl = np.empty(m, np.int64), np.empty(m, np.int64)
        spec = [cnt, dsp]
    glob_pos = res["pos"] + meta[1, rank]          # root buferidagi global joy
    for arr, out, typ in ((res["keys"], gk, MPI.UINT64_T), (res["counts"], gc, MPI.INT64_T),
                          (glob_pos, gp, MPI.INT64_T), (res["lens"], gl, MPI.INT64_T)):
        comm.Gatherv(np.ascontiguousarray(arr),
                     [out, spec[0], spec[1], typ] if rank == 0 else None, root=0)
    t3 = MPI.Wtime()

    merged = None
    if rank == 0:
        mk, mc, mp_, ml = _merge_entries(gk, gc, gp, gl, _next_pow2(max(1024, 2 * len(gk))))
        merged = _compact(mk, mc, mp_, ml, np.array([total], np.int64))
    t4 = MPI.Wtime()
    tm = {"taqsimlash": t1 - t0, "hisoblash": t2b - t1, "yigish": t3 - t2b,
          "birlashtirish": t4 - t3, "jami": t4 - t0}
    comp = comm.gather(t2 - t1, root=0)            # har bir jarayonning sof hisobi
    if rank == 0:
        tm["hisoblash_max"], tm["hisoblash_min"] = max(comp), min(comp)
        tm["baytlar"] = meta[0].tolist()
    return merged, tm


@muallif(ELBEK, 3)
def mpi_worker(args, comm=None, MPI=None):
    """mpiexec ostida ishlaydigan dastur (har bir jarayonda bir xil kod).
    Masalan:  mpiexec -n 4 python parallel_skreyper.py --mpi-worker
    Har bir jarayonda Numba 1 oqim bilan ishlaydi - sof taqsimlangan model.
    Birinchi yurish (JIT kompilyatsiya, kesh) hisobga olinmaydi."""
    if comm is None:
        from mpi4py import MPI as _MPI
        MPI, comm = _MPI, _MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if HAVE_NUMBA:
        numba.set_num_threads(1)
    tb, to = pack_corpus([generate_page(0, 4)])   # JIT ni oldindan kompilyatsiya qilish
    count_words_seq(tb, to)
    buf = offs = plan = None
    if rank == 0:
        with np.load(args.corpus) as d:
            buf, offs = d["buf"], d["offs"]
        if args.mode == "weak":                   # zaif: har bir jarayonga per_proc sahifa
            n = min(args.per_proc * size, len(offs) - 1)
            offs = offs[:n + 1]
            buf = buf[:int(offs[-1])]
        buf = np.ascontiguousarray(buf)
        plan = split_docs_by_bytes(offs, size)
    runs, merged = [], None
    for r in range(args.repeats + 1):
        merged, tm = mpi_count_once(comm, MPI, buf, offs, plan)
        if r > 0 and rank == 0:
            runs.append(tm)
    if rank != 0:
        return 0
    keys = ("taqsimlash", "hisoblash", "yigish", "birlashtirish", "jami",
            "hisoblash_max", "hisoblash_min")
    out = {"p": size, "mode": args.mode, "pages": len(offs) - 1,
           "mb": float(offs[-1]) / 2 ** 20, "bytes_per_rank": runs[-1]["baytlar"],
           "runs": runs, **{k: statistics.mean(t[k] for t in runs) for k in keys},
           "std": statistics.stdev([t["jami"] for t in runs]) if len(runs) > 1 else 0.0,
           "total_words": merged["total"], "unique_words": len(merged["keys"])}
    if args.mpi_out:
        with open(args.mpi_out, "w", encoding="utf-8") as f:
            json.dump(out, f)
        np.savez(args.mpi_out + ".npz", keys=merged["keys"], counts=merged["counts"],
                 total=np.array([merged["total"]]))
    else:
        comm_t = out["taqsimlash"] + out["yigish"]
        LOG(f"MPI: p={size}, {out['pages']} sahifa ({out['mb']:.1f} MB), "
            f"{out['total_words']} so'z, {out['unique_words']} xil")
        LOG(f"  jami {out['jami']:.4f} s | taqsimlash {out['taqsimlash']:.4f} | hisoblash "
            f"{out['hisoblash']:.4f} | yig'ish {out['yigish']:.4f} | birlashtirish "
            f"{out['birlashtirish']:.4f} | almashinuv ulushi {100 * comm_t / out['jami']:.1f}%")
    return 0


@muallif(ELBEK, 3)
def find_mpiexec(user_path=None):
    """mpiexec ni topadi: --mpiexec, PATH yoki MS-MPI o'rnatilgan papka."""
    if user_path:
        return user_path
    p = shutil.which("mpiexec") or shutil.which("mpirun")
    if p:
        return p
    for base in (os.environ.get("MSMPI_BIN", ""), r"C:\Program Files\Microsoft MPI\Bin"):
        cand = os.path.join(base, "mpiexec.exe")
        if base and os.path.exists(cand):
            return cand
    return None


@muallif(ELBEK, 3)
def run_mpi(p, corpus, mode, per_proc, repeats, mpiexec, extra, outdir):
    """mpiexec -n p ... --mpi-worker ni alohida jarayon sifatida ishga tushiradi
    va root jarayon yozgan natijani o'qiydi."""
    out = os.path.join(outdir, f"_mpi_{mode}_{p}.json")
    prog = [sys.executable] if getattr(sys, "frozen", False) else \
        [sys.executable, os.path.abspath(__file__)]       # exe ichida - exe ning o'zi
    cmd = [mpiexec, *extra, "-n", str(p), *prog,
           "--mpi-worker", "--corpus", corpus, "--mode", mode, "--per-proc", str(per_proc),
           "--repeats", str(repeats), "--mpi-out", out]
    subprocess.run(cmd, check=True, timeout=3600)
    with open(out, encoding="utf-8") as f:
        res = json.load(f)
    with np.load(out + ".npz") as d:              # with - fayl yopiladi, keyin o'chiriladi
        res["_result"] = {"keys": d["keys"], "counts": d["counts"],
                          "total": int(d["total"][0])}
    for f_ in (out, out + ".npz"):
        try:
            os.remove(f_)
        except OSError:                           # o'chmasa ham natijaga ta'sir qilmaydi
            pass
    return res


# ============================== 3-topshiriq tajribalari =======================
@muallif(OTABEK, 3)
def measure_stable(fn, repeats=3, warmup=2, max_cv=0.10, tries=3):
    """measure() + barqarorlik nazorati: takrorlar orasidagi tebranish
    (std / o'rtacha) max_cv dan katta bo'lsa (fon jarayoni, antivirus, chastota
    o'zgarishi), o'lchov qayta o'tkaziladi va eng barqarori olinadi."""
    best, best_cv = None, None
    for k in range(tries):
        m = measure(fn, repeats, warmup if k == 0 else 0)
        cv = m["std"] / m["mean"] if m["mean"] else 0.0
        m["cv"], m["tries"] = cv, k + 1
        if best is None or cv < best_cv:
            best, best_cv = m, cv
        if cv <= max_cv:
            break
        LOG(f"    (tebranish {100 * cv:.0f}% > {100 * max_cv:.0f}% - o'lchov qaytariladi)")
    return best


@muallif(OTABEK, 3)
def t3_exp_numba(ctx):
    """1- va 2-band: JIT vaqti alohida o'lchanadi va o'lchovdan chiqariladi;
    ketma-ket (range) va prange versiyalari oqimlar 1, 2, 4, 8 da."""
    LOG("\n[1-2] NUMBA: JIT, ketma-ket va prange (oqimlar) ...")
    buf, offs = ctx["corpus"]["bir_xil"]
    tb, to = pack_corpus([generate_page(i, 20) for i in range(4)])
    jit = {}
    for name, fn in (("ketma-ket (range)", lambda: count_words_seq(tb, to)),
                     ("parallel (prange)", lambda: count_words_parallel(tb, to, 2))):
        t0 = time.perf_counter()
        fn()                                       # 1-chaqiruv: kompilyatsiya + bajarish
        t1 = time.perf_counter()
        fn()                                       # 2-chaqiruv: faqat bajarish
        t2 = time.perf_counter()
        jit[name] = (t1 - t0) - (t2 - t1)
        LOG(f"    JIT kompilyatsiya, {name}: {jit[name]:.2f} s (o'lchovga kirmaydi)")
    count_words_seq(buf, offs)                    # isitish: OT keshi, sahifalar xotirada
    seq = measure_stable(lambda: count_words_seq(buf, offs), ctx["repeats"])
    ref = seq["result"]
    ctx["ref"]["bir_xil"] = ref
    rows = [{"mode": "ketma-ket (range)", "threads": 1, "mean": seq["mean"], "std": seq["std"],
             "speedup": 1.0, "eff": 1.0, "scan": seq["mean"], "merge": 0.0,
             "times": seq["times"]}]
    LOG(f"    ketma-ket: {seq['mean']:.4f} s, {offs[-1] / 2 ** 20 / seq['mean']:.0f} MB/s")
    for t in ctx["threads"]:
        m = measure_stable(lambda: count_words_parallel(buf, offs, t, 0), ctx["repeats"])
        s = seq["mean"] / m["mean"]
        ok, diff = same_counts(m["result"], ref)
        tms = m["result"]["times"]
        rows.append({"mode": "prange (static)", "threads": t, "mean": m["mean"],
                     "std": m["std"], "speedup": s, "eff": s / t, "scan": tms["scan"],
                     "merge": tms["merge"] + tms["alloc"], "times": m["times"]})
        LOG(f"    oqimlar={t}: {m['mean']:.4f} s  S={s:.2f}  E={s / t:.2f}  "
            f"(scan {tms['scan']:.4f}, merge+alloc {tms['merge'] + tms['alloc']:.4f})")
        check(ctx, f"prange oqimlar={t} == ketma-ket", ok, f"(max farq {diff:.1e})")
    ctx["res"]["jit"] = jit
    ctx["res"]["numba"] = rows
    ctx["res"]["seq_mbps"] = offs[-1] / 2 ** 20 / seq["mean"]


SCHEDULES = (("static (blok)", 0, False), ("static, chunk=1 (siklik)", 0, True),
             ("dynamic, chunk=1", 1, False), ("dynamic, chunk=8", 8, False))


@muallif(OTABEK, 3)
def t3_exp_schedule(ctx):
    """3-band: static va dynamic rejalashtirish, bir xil va notekis korpusda.
    Har bir oqim qayta ishlagan baytlar orqali nomutanosiblik (max/o'rtacha)."""
    LOG("\n[3] STATIC va DYNAMIC rejalashtirish ...")
    t = max(ctx["threads"])
    rows = []
    for kind in ("bir_xil", "notekis"):
        buf, offs = ctx["corpus"][kind]
        if kind not in ctx["ref"]:
            ctx["ref"][kind] = count_words_seq(buf, offs)
        ref = ctx["ref"][kind]
        sizes = np.diff(offs) / 1024
        for label, cs, cyc in SCHEDULES:
            m = measure_stable(lambda: count_words_parallel(buf, offs, t, cs, cyclic=cyc),
                        ctx["repeats"])
            w = m["result"]["work"]
            w = w[w > 0]
            imb = float(w.max() / w.mean()) if len(w) else 1.0
            rows.append({"corpus": kind, "schedule": label, "threads": t, "mean": m["mean"],
                         "std": m["std"], "imbalance": imb, "active": int(len(w)),
                         "cv": float(sizes.std() / sizes.mean())})
            ok, diff = same_counts(m["result"], ref)
            LOG(f"    {kind:8s} {label:25s}: {m['mean']:.4f} s, nomutanosiblik "
                f"max/o'rt = {imb:.2f}")
            check(ctx, f"{kind}, {label} == ketma-ket", ok)
    ctx["res"]["schedule"] = rows


@muallif(OTABEK, 3)
def t3_exp_numba_weak(ctx):
    """6-band (umumiy xotira): zaif masshtablanish - har bir oqimga N/8 sahifa."""
    LOG("\n[6a] NUMBA: zaif masshtablanish ...")
    buf, offs = ctx["corpus"]["bir_xil"]
    per = (len(offs) - 1) // 8
    rows = []
    for t in ctx["threads"]:
        n = per * t
        o = offs[:n + 1]
        b = buf[:int(o[-1])]
        m = measure_stable(lambda: count_words_parallel(b, o, t, 0), ctx["repeats"])
        rows.append({"threads": t, "pages": n, "mean": m["mean"], "std": m["std"]})
        LOG(f"    oqimlar={t}, {n} sahifa: {m['mean']:.4f} s")
    for r in rows:
        r["eff"] = rows[0]["mean"] / r["mean"]
    ctx["res"]["numba_weak"] = rows


@muallif(ELBEK, 3)
def t3_exp_mpi(ctx):
    """4-, 5-, 6-band: MPI kuchli (N o'zgarmas) va zaif (har jarayonga N/8)
    masshtablanish; har bir bosqich vaqti va almashinuv ulushi."""
    a = ctx["args"]
    mpiexec = find_mpiexec(a.mpiexec)
    if mpiexec is None:
        LOG("\n[4-6] MPI: mpiexec topilmadi - MS-MPI o'rnating (README ga qarang). "
            "MPI tajribalari o'tkazib yuborildi.")
        ctx["res"]["mpi"] = None
        return
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        LOG("\n[4-6] MPI: mpi4py o'rnatilmagan (pip install mpi4py). O'tkazib yuborildi.")
        ctx["res"]["mpi"] = None
        return
    extra = a.mpi_args.split() if a.mpi_args else []
    corpus = ctx["corpus_path"]["bir_xil"]
    per = ctx["pages3"] // 8
    out = {"strong": [], "weak": [], "mpiexec": mpiexec}
    for mode in ("strong", "weak"):
        LOG(f"\n[4-6] MPI: {'kuchli' if mode == 'strong' else 'zaif'} masshtablanish ...")
        for p in ctx["procs"]:
            r = run_mpi(p, corpus, mode, per, ctx["repeats"], mpiexec, extra, a.out3)
            res = r.pop("_result")
            comm_t = r["taqsimlash"] + r["yigish"]
            r["comm_share"] = comm_t / r["jami"]
            out[mode].append(r)
            LOG(f"    p={p}: {r['pages']} sahifa, jami {r['jami']:.4f} s | taqsimlash "
                f"{r['taqsimlash']:.4f} | hisoblash {r['hisoblash']:.4f} | yig'ish "
                f"{r['yigish']:.4f} | birlashtirish {r['birlashtirish']:.4f} | "
                f"almashinuv {100 * r['comm_share']:.1f}%")
            if mode == "strong":
                ok, diff = same_counts(res, ctx["ref"]["bir_xil"])
                check(ctx, f"MPI p={p} == ketma-ket", ok, f"(max farq {diff:.1e})")
            else:
                b, o = ctx["corpus"]["bir_xil"]
                n = r["pages"]
                if p == ctx["procs"][-1] or p == 1:
                    ref = count_words_seq(b[:int(o[n])], o[:n + 1])
                    ok, _ = same_counts(res, ref)
                    check(ctx, f"MPI zaif p={p} == ketma-ket", ok)
    for mode in ("strong", "weak"):
        rows = out[mode]
        for r in rows:
            r["speedup"] = rows[0]["jami"] / r["jami"]
            r["eff"] = r["speedup"] / r["p"] if mode == "strong" else r["speedup"]
    ctx["res"]["mpi"] = out


@muallif(OTABEK, 3)
def t3_check_python(ctx):
    """7-band qo'shimcha: Numba yadrosi 2-topshiriqdagi Python tahlili
    (BeautifulSoup + regex + Counter) bilan ham aynan bir xil sanaydimi?"""
    LOG("\n[7] To'g'rilik: Numba yadrosi va 2-topshiriqdagi Python tahlili ...")
    for kind in ("bir_xil", "notekis"):
        buf, offs = ctx["corpus"][kind]
        n = min(40, len(offs) - 1)
        o = offs[:n + 1]
        b = buf[:int(o[-1])]
        htmls = [bytes(b[o[i]:o[i + 1]]).decode("utf-8") for i in range(n)]
        t0 = time.perf_counter()
        py = analyze_sequential(htmls)
        if kind == "bir_xil":
            ctx["res"]["python_mbps"] = float(o[-1]) / 2 ** 20 / (time.perf_counter() - t0)
        nb = decode_words(count_words_seq(b, o), b)
        check(ctx, f"{kind}: Numba == Python (bs4+Counter), {n} sahifa",
              nb == dict(py), f"({sum(py.values())} so'z, {len(py)} xil)")


@muallif(OTABEK, 3)
def t3_conclusions(res):
    """O'lchov natijalaridan 3-topshiriq xulosasi."""
    nb = res["numba"]
    best = max(nb[1:], key=lambda r: r["speedup"])
    last = nb[-1]
    f = last["merge"] / last["mean"] if last["mean"] else 0
    sch = {(r["corpus"], r["schedule"]): r for r in res["schedule"]}
    us, ud = sch[("notekis", "static (blok)")], sch[("notekis", "dynamic, chunk=1")]
    bs, bd = sch[("bir_xil", "static (blok)")], sch[("bir_xil", "dynamic, chunk=1")]
    uc = sch[("notekis", "static, chunk=1 (siklik)")]
    best_u = min((r for r in res["schedule"] if r["corpus"] == "notekis"),
                 key=lambda r: r["mean"])
    out = [
        f"Numba yadrosi (ketma-ket, range) {res['seq_mbps']:.0f} MB/s tezlikda ishlaydi, "
        f"2-topshiriqdagi Python tahlili (bs4 + regex) esa {res.get('python_mbps', 0):.1f} "
        f"MB/s - {res['seq_mbps'] / max(1e-9, res.get('python_mbps', 1e-9)):.0f} baravar "
        f"farq; mashina kodiga kompilyatsiya GIL va interpretatorni chetlab o'tadi. JIT kompilyatsiya "
        f"{max(res['jit'].values()):.1f} s gacha vaqt oladi, shuning uchun birinchi "
        f"chaqiruv o'lchovdan chiqarildi.",
        f"prange bilan eng yaxshi natija {best['threads']} oqimda: S = {best['speedup']:.2f}, "
        f"E = {best['eff']:.2f}. Idealdan chetlanish sabablari: ketma-ket birlashtirish va "
        f"jadval ajratish ({100 * f:.0f}% vaqt, Amdal qonuni), ko'p yadro band bo'lganda "
        f"protsessor chastotasining (turbo) pasayishi, SMT - bitta fizik yadroni ikki "
        f"mantiqiy oqim bo'lishishi va oqimlarni ishga tushirish xarajati.",
        f"Rejalashtirish: bir xil sahifalarda static {bs['mean']:.4f} s, dynamic "
        f"{bd['mean']:.4f} s; notekis korpusda static nomutanosiblik "
        f"{us['imbalance']:.2f}, siklik {uc['imbalance']:.2f}, dynamic {ud['imbalance']:.2f}; "
        f"vaqt: static {us['mean']:.4f} s, siklik {uc['mean']:.4f} s, dynamic "
        f"{ud['mean']:.4f} s; notekis korpusda eng tezi - {best_u['schedule']}. "
        + (f"Numba threading layer = {res.get('threading_layer', '?')}: bu qatlamda "
           f"chunksize bo'laklari ham oqimlarga static bloklab beriladi, shuning uchun "
           f"dynamic nomutanosiblikni kamaytirmadi; yuklamani faqat siklik taqsimot "
           f"aralashtirdi. "
           if res.get("threading_layer") != "tbb" and abs(ud["imbalance"] - us["imbalance"]) < 0.05
           else "") +
        "Real saytlarda sahifa hajmi juda farq qiladi va oldindan noma'lum, shuning uchun "
        "loyiha uchun yuklamani ish vaqtida yoki aralashtirib taqsimlaydigan "
        "rejalashtirish mos keladi.",
    ]
    mpi = res.get("mpi")
    if mpi:
        st, wk = mpi["strong"], mpi["weak"]
        out.append(
            f"MPI: kuchli masshtablanishda p = {st[-1]['p']} da S = {st[-1]['speedup']:.2f}, "
            f"almashinuv ulushi p = {st[0]['p']} da {100 * st[0]['comm_share']:.0f}% dan "
            f"p = {st[-1]['p']} da {100 * st[-1]['comm_share']:.0f}% gacha o'zgardi: hisoblash "
            f"p marta kamayadi, lekin {st[-1]['mb']:.0f} MB matnni root'dan tarqatish "
            f"kamaymaydi. Zaif masshtablanishda samaradorlik p = {wk[-1]['p']} da "
            f"{wk[-1]['eff']:.2f}.")
        bm = min(st, key=lambda r: r["jami"])
        if best["mean"] <= bm["jami"]:
            verdict = (f"bitta kompyuterda umumiy xotira (Numba, {best['threads']} oqim: "
                       f"{best['mean']:.4f} s) MPI dan (p = {bm['p']}: {bm['jami']:.4f} s) tez - "
                       f"ma'lumot nusxalanmaydi, oqimlar bitta buferni o'qiydi.")
        else:
            verdict = (f"bu o'lchovda MPI (p = {bm['p']}: {bm['jami']:.4f} s) Numba dan "
                       f"({best['threads']} oqim: {best['mean']:.4f} s) tez chiqdi - "
                       f"jarayonlar alohida xotira va keshlar bilan ishlaydi.")
        out.append(
            f"Loyiha uchun: {verdict} MPI ma'lumotni jarayonlar orasida uzatishga majbur; "
            "uning afzalligi - bir nechta kompyuterga kengaytirish mumkinligi va "
            "jarayonlar xotirasining ajratilganligi (bir jarayondagi xato boshqasiga ta'sir "
            "qilmaydi).")
    return out


@muallif(OTABEK, 3)
def make_charts3(res, outdir):
    """3-topshiriq grafiklari (o'q nomlari, birliklar, izohli sarlavha bilan)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, name), dpi=130)
        plt.close(fig)
        files.append(name)

    nb = res["numba"][1:]
    ts = [r["threads"] for r in nb]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ts, ts, "k--", alpha=.4, label="ideal S = n")
    axes[0].plot(ts, [r["speedup"] for r in nb], marker="o", label="prange (static)")
    axes[0].set_ylabel("Tezlashish, S = T_ketma-ket / T_n")
    axes[1].axhline(1, color="k", ls="--", alpha=.4, label="ideal E = 1")
    axes[1].plot(ts, [r["eff"] for r in nb], marker="o", color="C1", label="prange (static)")
    axes[1].set_ylabel("Samaradorlik, E = S / n")
    axes[1].set_ylim(0, 1.15)
    for ax, title in zip(axes, ("Tezlashish", "Samaradorlik")):
        ax.set_xlabel("Oqimlar soni, n")
        ax.set_xticks(ts)
        ax.set_title(title)
        ax.grid(alpha=.3)
        ax.legend()
    fig.suptitle("Numba (umumiy xotira): oqimlar bo'yicha masshtablanish")
    save(fig, "1_numba_oqimlar.png")

    sc = res["schedule"]
    labels = [lb for lb, _, _ in SCHEDULES]
    nl = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, kind in enumerate(("bir_xil", "notekis")):
        rr = [next(r for r in sc if r["corpus"] == kind and r["schedule"] == lb) for lb in labels]
        axes[0].bar([j + i * 0.38 for j in range(nl)], [r["mean"] for r in rr], width=0.38,
                    yerr=[r["std"] for r in rr], capsize=3,
                    label="bir xil sahifalar" if i == 0 else "notekis sahifalar")
        axes[1].bar([j + i * 0.38 for j in range(nl)], [r["imbalance"] for r in rr], width=0.38,
                    label="bir xil sahifalar" if i == 0 else "notekis sahifalar")
    for ax in axes:
        ax.set_xticks([j + 0.19 for j in range(nl)])
        ax.set_xticklabels([lb.replace(" (", "\n(") for lb in labels], fontsize=8)
        ax.grid(axis="y", alpha=.3)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        ax.legend(loc="upper center", ncol=2, fontsize=9)
    axes[0].set_ylabel("Bajarilish vaqti, T (s)")
    axes[0].set_title("Vaqt")
    axes[1].axhline(1, color="k", ls="--", alpha=.4)
    axes[1].set_ylabel("Yuklama nomutanosibligi, max / o'rtacha")
    axes[1].set_title("Oqimlar orasida baytlar taqsimoti")
    fig.suptitle(f"Static va dynamic rejalashtirish ({sc[0]['threads']} oqim)")
    save(fig, "2_rejalashtirish.png")

    mpi = res.get("mpi")
    if mpi:
        st = mpi["strong"]
        ps = [str(r["p"]) for r in st]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bottom = [0.0] * len(st)
        for key, lab in (("taqsimlash", "taqsimlash (Scatterv)"), ("hisoblash", "hisoblash"),
                         ("yigish", "yig'ish (Gatherv)"), ("birlashtirish", "birlashtirish (root)")):
            vals = [r[key] for r in st]
            ax.bar(ps, vals, bottom=bottom, label=lab)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax2 = ax.twinx()
        ax2.plot(ps, [100 * r["comm_share"] for r in st], "ko-", label="almashinuv ulushi")
        ax2.set_ylabel("Almashinuv ulushi (%)")
        ax2.set_ylim(0, 100)
        ax.set_xlabel("MPI jarayonlari soni, p")
        ax.set_ylabel("Vaqt (s)")
        ax.set_title("MPI: hisoblash va almashinuv vaqti (kuchli masshtablanish)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.set_ylim(0, max(bottom) * 1.6)
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", ncol=2)
        ax.grid(axis="y", alpha=.3)
        save(fig, "3_mpi_almashinuv.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhline(1, color="k", ls="--", alpha=.4, label="ideal")
    if mpi:
        ax.plot([r["p"] for r in mpi["strong"]], [r["eff"] for r in mpi["strong"]], "o-",
                label="MPI, kuchli: T1 / (p·Tp)")
        ax.plot([r["p"] for r in mpi["weak"]], [r["eff"] for r in mpi["weak"]], "s-",
                label="MPI, zaif: T1 / Tp")
    p1 = res["numba"][1]["mean"]
    ax.plot(ts, [p1 / (r["threads"] * r["mean"]) for r in nb], "o--",
            label="Numba, kuchli: T1 / (n·Tn)")
    ax.plot([r["threads"] for r in res["numba_weak"]], [r["eff"] for r in res["numba_weak"]],
            "s--", label="Numba, zaif: T1 / Tn")
    ax.set_xlabel("Jarayonlar / oqimlar soni")
    ax.set_ylabel("Masshtablanish samaradorligi")
    ax.set_ylim(0, 1.2)
    ax.set_xticks(ts)
    ax.set_title("Kuchli va zaif masshtablanish")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    save(fig, "4_kuchli_zaif.png")
    return files


@muallif(OTABEK, 3)
def write_report3(ctx, outdir):
    """natijalar3.md, natijalar3.json, barcha_natijalar3.csv va grafiklar."""
    res = ctx["res"]
    charts = make_charts3(res, outdir)
    with open(os.path.join(outdir, "natijalar3.json"), "w", encoding="utf-8") as f:
        json.dump({"env": ctx["env"], "results": res, "checks": ctx["checks"],
                   "authors": AUTHORS3}, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(outdir, "barcha_natijalar3.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["tajriba", "rejim", "parametr", "T_ortacha_s", "std_s", "S", "E",
                    "qoshimcha"])
        for r in res["numba"]:
            w.writerow(["numba", r["mode"], f"oqim={r['threads']}", f"{r['mean']:.5f}",
                        f"{r['std']:.5f}", f"{r['speedup']:.3f}", f"{r['eff']:.3f}",
                        f"merge={r['merge']:.5f}"])
        for r in res["schedule"]:
            w.writerow(["rejalashtirish", f"{r['corpus']} {r['schedule']}", f"oqim={r['threads']}",
                        f"{r['mean']:.5f}", f"{r['std']:.5f}", "", "",
                        f"nomutanosiblik={r['imbalance']:.3f}"])
        for r in res["numba_weak"]:
            w.writerow(["numba zaif", "prange", f"oqim={r['threads']}", f"{r['mean']:.5f}",
                        f"{r['std']:.5f}", "", f"{r['eff']:.3f}", f"sahifa={r['pages']}"])
        if res.get("mpi"):
            for mode in ("strong", "weak"):
                for r in res["mpi"][mode]:
                    w.writerow([f"mpi {mode}", "mpi4py", f"p={r['p']}", f"{r['jami']:.5f}",
                                f"{r['std']:.5f}", f"{r['speedup']:.3f}", f"{r['eff']:.3f}",
                                f"almashinuv={100 * r['comm_share']:.1f}%"])
    md = ["# Parallel veb-skreyper - 3-topshiriq natijalari (Numba va MPI)", "",
          f"**Sana:** {time.strftime('%Y-%m-%d %H:%M')}", "", "## Hissa", "",
          fmt_table(["A'zo", "Funksiyalar"],
                    [[k, ", ".join(f"`{n}`" for n in v)] for k, v in AUTHORS3.items() if v]),
          "", "## Sinov muhiti", "",
          fmt_table(["Parametr", "Qiymat"], [[k, v] for k, v in ctx["env"].items()]),
          "", "## Korpus", "",
          fmt_table(["Korpus", "Sahifalar", "MB", "min KB", "o'rtacha KB", "max KB"],
                    [[k, v["pages"], f"{v['mb']:.1f}", v["min_kb"], f"{v['mean_kb']:.1f}",
                      v["max_kb"]] for k, v in res["corpus"].items()]),
          "", "## JIT kompilyatsiya (o'lchovdan chiqarilgan)", "",
          fmt_table(["Funksiya", "JIT, s"], [[k, f"{v:.2f}"] for k, v in res["jit"].items()]),
          "", "## Numba: oqimlar", "",
          fmt_table(["Rejim", "Oqim", "T, s", "std", "S", "E", "merge+alloc, s"],
                    [[r["mode"], r["threads"], f"{r['mean']:.4f}", f"{r['std']:.4f}",
                      f"{r['speedup']:.2f}", f"{r['eff']:.2f}", f"{r['merge']:.4f}"]
                     for r in res["numba"]]),
          "", "## Static va dynamic", "",
          fmt_table(["Korpus", "Rejalashtirish", "T, s", "std", "max/o'rtacha"],
                    [[r["corpus"], r["schedule"], f"{r['mean']:.4f}", f"{r['std']:.4f}",
                      f"{r['imbalance']:.2f}"] for r in res["schedule"]]),
          "", "## Numba: zaif masshtablanish", "",
          fmt_table(["Oqim", "Sahifalar", "T, s", "E_zaif"],
                    [[r["threads"], r["pages"], f"{r['mean']:.4f}", f"{r['eff']:.2f}"]
                     for r in res["numba_weak"]])]
    if res.get("mpi"):
        for mode, title in (("strong", "MPI: kuchli masshtablanish"),
                            ("weak", "MPI: zaif masshtablanish")):
            md += ["", f"## {title}", "", fmt_table(
                ["p", "Sahifalar", "Jami, s", "Taqsimlash", "Hisoblash", "Yig'ish",
                 "Birlashtirish", "Almashinuv, %", "S", "E"],
                [[r["p"], r["pages"], f"{r['jami']:.4f}", f"{r['taqsimlash']:.4f}",
                  f"{r['hisoblash']:.4f}", f"{r['yigish']:.4f}", f"{r['birlashtirish']:.4f}",
                  f"{100 * r['comm_share']:.1f}", f"{r['speedup']:.2f}", f"{r['eff']:.2f}"]
                 for r in res["mpi"][mode]])]
    else:
        md += ["", "## MPI", "", "MPI tajribalari bajarilmadi (mpiexec yoki mpi4py yo'q)."]
    md += ["", "## To'g'rilik", "",
           fmt_table(["Tekshiruv", "Natija"],
                     [[c["name"] + " " + c["detail"], "MOS" if c["ok"] else "**XATO**"]
                      for c in ctx["checks"]]),
           "", "## Xulosa", ""]
    md += [f"{i}. {t}" for i, t in enumerate(t3_conclusions(res), 1)]
    md += ["", "## Grafiklar", ""] + [f"![{c}]({c})" for c in charts]
    with open(os.path.join(outdir, "natijalar3.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return charts


@muallif(BOTH, 3)
def run_task3(args):
    """3-topshiriqning barcha tajribalari."""
    if np is None:
        sys.exit("numpy kerak: pip install numpy")
    if not HAVE_NUMBA:
        sys.exit("Numba o'rnatilmagan: pip install numba")
    os.makedirs(args.out3, exist_ok=True)
    env = collect_env()
    env["numpy"] = np.__version__
    env["numba"] = numba.__version__
    env["Numba oqimlari (NUMBA_NUM_THREADS)"] = numba.config.NUMBA_NUM_THREADS
    try:
        import mpi4py
        from mpi4py import MPI
        env["mpi4py"] = mpi4py.__version__
        env["MPI kutubxonasi"] = MPI.Get_library_version().strip().splitlines()[0][:80]
    except Exception:
        env["mpi4py"] = "o'rnatilmagan"
    threads = [t for t in args.t3_threads if t <= numba.config.NUMBA_NUM_THREADS]
    ctx = {"args": args, "res": {}, "checks": [], "env": env, "repeats": args.repeats,
           "threads": threads, "procs": args.t3_procs, "pages3": args.t3_pages,
           "corpus": {}, "corpus_path": {}, "ref": {}}
    LOG("3-TOPSHIRIQ. Sinov muhiti: " + ", ".join(f"{k}: {v}" for k, v in env.items()))
    LOG("\n[0] Korpus (loyihaning skreyperi bilan yig'iladi) ...")
    ctx["res"]["corpus"] = {}
    for kind in ("bir_xil", "notekis"):
        path = os.path.join(args.out3, f"korpus_{kind}.npz")
        buf, offs = build_corpus(args.t3_pages, kind, path)
        ctx["corpus"][kind], ctx["corpus_path"][kind] = (buf, offs), path
        kb = np.diff(offs) / 1024
        ctx["res"]["corpus"][kind] = {"pages": len(kb), "mb": float(offs[-1]) / 2 ** 20,
                                      "min_kb": int(kb.min()), "mean_kb": float(kb.mean()),
                                      "max_kb": int(kb.max())}
        LOG(f"    {kind}: {len(kb)} sahifa, {offs[-1] / 2 ** 20:.1f} MB "
            f"(sahifa {kb.min():.0f}...{kb.max():.0f} KB)")
    if not hasattr(numba, "set_parallel_chunksize"):
        sys.exit("Numba >= 0.57 kerak: pip install -U numba")
    for exp in (t3_exp_numba, t3_exp_schedule, t3_exp_numba_weak, t3_exp_mpi,
                t3_check_python):
        exp(ctx)
        if exp is t3_exp_numba:
            env["Numba threading layer"] = numba.threading_layer()
            ctx["res"]["threading_layer"] = env["Numba threading layer"]
            if env["Numba threading layer"] != "tbb":
                LOG(f"    Eslatma: threading layer = {env['Numba threading layer']}. Haqiqiy "
                    "dynamic rejalashtirish uchun TBB kerak: pip install tbb")
    charts = write_report3(ctx, args.out3)
    ok = sum(c["ok"] for c in ctx["checks"])
    LOG(f"\nTo'g'rilik: {ok}/{len(ctx['checks'])} tekshiruv MOS")
    LOG("\nXULOSA:")
    for i, t in enumerate(t3_conclusions(ctx["res"]), 1):
        LOG(f"  {i}. {t}")
    LOG(f"\nNatijalar: {os.path.abspath(args.out3)} (natijalar3.md, natijalar3.json, "
        f"barcha_natijalar3.csv, {len(charts)} ta grafik)")
    return 0


@muallif(OTABEK, 3)
def run_self_tests3(args):
    """3-topshiriq uchun tez avtomatik testlar (--test3): ketma-ket, prange
    (static/dynamic), Python tahlili va (o'rnatilgan bo'lsa) MPI p=4 natijalari."""
    if np is None:
        sys.exit("numpy kerak: pip install numpy")
    LOG("3-TOPSHIRIQ TESTLARI" + ("" if HAVE_NUMBA else
                                 " (Numba yo'q - oddiy Python rejimi, sekin)"))
    ctx = {"checks": []}
    htmls = [generate_page(i, 6 if i % 3 else 14) for i in range(9)]
    htmls.append("<html><head><title>Sarlavha Soz</title></head><body><!-- izohsoz -->"
                 "<p>Abc<b>DEF</b> x &amp; y&nbsp;zzz 'qqq' a < b ma'lumot "
                 "\u2018qoshtirnoq\u2019 soz123soz o'zbek</p><SCRIPT>skr skr</SCRIPT>"
                 "<style>.a{}</style></body></html>")
    buf, offs = pack_corpus(htmls)
    with np.errstate(over="ignore"):
        seq = count_words_seq(buf, offs)
        py = analyze_sequential(htmls)
        check(ctx, "Numba yadrosi == Python (bs4 + regex + Counter)",
              decode_words(seq, buf) == dict(py), f"({len(py)} xil so'z)")
        check(ctx, "kichik jadval to'lganda qayta hisoblash",
              same_counts(count_words_seq(buf, offs, cap=16), seq)[0])
        for t, cs, cyc in ((2, 0, False), (4, 1, False), (3, 2, False), (3, 0, True)):
            check(ctx, f"prange oqim={t} chunk={cs} siklik={cyc} == ketma-ket",
                  same_counts(count_words_parallel(buf, offs, t, cs, cyclic=cyc), seq)[0])
        plan = split_docs_by_bytes(offs, 4)
        check(ctx, "split_docs_by_bytes chegaralari", plan[0] == 0 and plan[-1] == len(htmls)
              and all(a <= b for a, b in zip(plan, plan[1:])))
    mpiexec = find_mpiexec(getattr(args, "mpiexec", None))
    try:
        import mpi4py  # noqa: F401
        have_mpi = mpiexec is not None
    except ImportError:
        have_mpi = False
    if have_mpi:
        os.makedirs(args.out3, exist_ok=True)
        path = os.path.join(args.out3, "_test_korpus.npz")
        np.savez(path, buf=buf, offs=offs)
        extra = args.mpi_args.split() if args.mpi_args else []
        r = run_mpi(4, path, "strong", 2, 1, mpiexec, extra, args.out3)
        check(ctx, "MPI p=4 == ketma-ket", same_counts(r["_result"], seq)[0])
        try:
            os.remove(path)
        except OSError:
            pass
    else:
        LOG("    MPI (mpiexec + mpi4py) topilmadi - MPI testi o'tkazib yuborildi")
    failed = [c for c in ctx["checks"] if not c["ok"]]
    LOG(f"\n{len(ctx['checks']) - len(failed)}/{len(ctx['checks'])} test o'tdi")
    return 1 if failed else 0


# #############################################################################
#   4-TOPSHIRIQ: GPU (A7 - CUDA, Numba CUDA) VA GETEROGEN HISOBLASH (A8)
# -----------------------------------------------------------------------------
#   So'zlarni sanash yadrosi GPU'ga ko'chirildi. GPU minglab oqim bilan
#   ishlagani uchun sahifalar ~2 KB li SEGMENTLARGA bo'linadi (har segmentni
#   bitta CUDA oqimi qayta ishlaydi). Segment chegarasi '>' dan keyingi joyga
#   suriladi - shunda so'z, &entity; va UTF-8 belgisi ikkiga bo'linmaydi.
#   Segment boshidagi HTML holati (matn, teg, <script> ichi, izoh ...) chekli
#   avtomat (DFA) bilan aniqlanadi:
#     k_bound -> k_spec (har segment "matn" holatidan boshlanadi deb taxmin)
#     -> k_fix (hujjat bo'yicha taxminni tuzatish) -> k_count (sanash)
#     -> k_compact (natijani ixchamlash) -> Device->Host.
#   So'zlar GPU xotirasidagi umumiy xesh-jadvalga atomik CAS/ADD bilan yoziladi.
#   Bandlar: 1-GPU ga ko'chirish .......... GpuCounter, gpu_count_words()
#            2-blok o'lchami ............. t4_exp_block()
#            3-H2D / hisoblash / D2H ..... t4_exp_phases()
#            4-geterogen sxema ........... hetero_static(), hetero_dynamic()
#            5-optimal nisbat ............ t4_exp_hetero()
#            6-pinned + stream ........... t4_exp_overlap()
#            7-to'g'rilik (float32/64) ... t4_exp_precision(), same_counts()
#            8-muhit ..................... gpu_env()
# #############################################################################
from types import SimpleNamespace

CUDA_IMPORT_ERROR = None                             # diagnostika: nega CUDA yuklanmadi
try:
    from numba import cuda
except Exception as _e:                              # numba yoki numba-cuda yo'q
    cuda = None
    CUDA_IMPORT_ERROR = f"{_e.__class__.__name__}: {_e}"


# ===================== SOYIBJONOV OTABEK: DFA va CUDA yadrolari ==============
DFA_T, DFA_LT, DFA_TAG, DFA_BANG, DFA_BANG1, DFA_C0, DFA_C1, DFA_C2, DFA_RAWCLOSE = range(9)
_RAW_WORDS = ("script", "style", "noscript")


_DFA_CACHE = []


@muallif(OTABEK, 4)
def markup_dfa():
    """DFA jadvali (bir marta quriladi)."""
    if not _DFA_CACHE:
        _DFA_CACHE.append(build_markup_dfa())
    return _DFA_CACHE[0]


@muallif(OTABEK, 4)
def build_markup_dfa():
    """HTML belgilash holatlari uchun chekli avtomat (DFA) jadvali: table[s*256+c].
    Holatlar: T (matn), LT ('<' dan keyin), TAG, izoh (<!-- -->), teg nomi
    prefikslari (script/style/noscript), xom matn (<script> ichi) va uning
    yopuvchi tegini qidirish. 3-topshiriqdagi _skip_tag bilan aynan bir xil
    qoidalar - bu tests4 da avtomatik tekshiriladi."""
    names = {}

    def sid(key):
        if key not in names:
            names[key] = len(names) + 9
        return names[key]
    prefixes = sorted({w[:k] for w in _RAW_WORDS for k in range(1, len(w) + 1)})
    for p in prefixes:
        sid(("name", p))
    for w in _RAW_WORDS:
        sid(("tagraw", w))
        for k in range(len(w) + 3):
            sid(("raw", w, k))
    ns = 9 + len(names)
    tab = np.zeros((ns, 256), dtype=np.uint8)

    def alpha(c):
        return 65 <= c <= 90 or 97 <= c <= 122

    def low(c):
        return c + 32 if 65 <= c <= 90 else c
    for c in range(256):
        tab[DFA_T, c] = DFA_LT if c == 60 else DFA_T
    for c in range(256):
        if alpha(c):
            p = chr(low(c))
            tab[DFA_LT, c] = names[("name", p)] if p in prefixes else DFA_TAG
        elif c == 47:
            tab[DFA_LT, c] = DFA_TAG                 # yopuvchi teg: '>' gacha
        elif c == 33:
            tab[DFA_LT, c] = DFA_BANG
        else:
            tab[DFA_LT, c] = tab[DFA_T, c]           # '<' ajratuvchi, c matnda qayta ishlanadi
        tab[DFA_TAG, c] = DFA_T if c == 62 else DFA_TAG
        tab[DFA_BANG, c] = DFA_BANG1 if c == 45 else (DFA_T if c == 62 else DFA_TAG)
        tab[DFA_BANG1, c] = DFA_C0 if c == 45 else (DFA_T if c == 62 else DFA_TAG)
        tab[DFA_C0, c] = DFA_C1 if c == 45 else DFA_C0
        tab[DFA_C1, c] = DFA_C2 if c == 45 else DFA_C0
        tab[DFA_C2, c] = DFA_C2 if c == 45 else (DFA_T if c == 62 else DFA_C0)
        tab[DFA_RAWCLOSE, c] = DFA_T if c == 62 else DFA_RAWCLOSE
    for p in prefixes:
        s = names[("name", p)]
        for c in range(256):
            if alpha(c):
                q = p + chr(low(c))
                tab[s, c] = names[("name", q)] if q in prefixes else DFA_TAG
            elif p in _RAW_WORDS:
                tab[s, c] = names[("raw", p, 0)] if c == 62 else names[("tagraw", p)]
            else:
                tab[s, c] = DFA_T if c == 62 else DFA_TAG
    for w in _RAW_WORDS:
        tr = names[("tagraw", w)]
        full = len(w) + 2
        pattern = "</" + w
        for c in range(256):
            tab[tr, c] = names[("raw", w, 0)] if c == 62 else tr
            for k in range(full + 1):
                s = names[("raw", w, k)]
                if k < full:
                    ok = (low(c) == ord(pattern[k])) if k >= 2 else (c == ord(pattern[k]))
                    if ok:
                        tab[s, c] = names[("raw", w, k + 1)]
                    else:
                        tab[s, c] = names[("raw", w, 1)] if c == 60 else names[("raw", w, 0)]
                else:
                    if alpha(c):
                        tab[s, c] = names[("raw", w, 0)]
                    else:
                        tab[s, c] = DFA_T if c == 62 else DFA_RAWCLOSE
    return tab.ravel().copy()


@muallif(OTABEK, 4)
def make_segments(offs, seg):
    """Har bir hujjatni seg baytli nominal segmentlarga bo'ladi (host, numpy).
    Qaytaradi: nominal boshlanishlar, segment->hujjat, birinchi segment
    belgisi, hujjatning birinchi segmenti va segmentlar soni."""
    L = np.diff(offs)
    cnt = np.maximum(1, (L + seg - 1) // seg)
    first = np.zeros(len(L), np.int64)
    first[1:] = np.cumsum(cnt)[:-1]
    nseg = int(cnt.sum())
    seg_doc = np.repeat(np.arange(len(L), dtype=np.int32), cnt)
    local = np.arange(nseg, dtype=np.int64) - np.repeat(first, cnt)
    nom = (offs[:-1][seg_doc] + local * seg).astype(np.int64)
    return nom, seg_doc, (local == 0).astype(np.uint8), first, cnt.astype(np.int64)


_GPU = None


@muallif(OTABEK, 4)
def build_gpu_kernels(cu=None):
    """CUDA yadrolarini yasaydi (@cuda.jit). Fabrika ko'rinishida - numba-cuda
    o'rnatilmagan kompyuterda ham fayl import qilinaveradi."""
    cu = cu or cuda
    T, LT = DFA_T, DFA_LT
    OFF = np.uint64(0xCBF29CE484222325)
    PRIME = np.uint64(0x100000001B3)
    ONE = np.uint64(1)
    ZERO = np.uint64(0)

    @cu.jit(device=True)
    def d_alpha(c):
        return (c >= 65 and c <= 90) or (c >= 97 and c <= 122)

    @cu.jit(device=True)
    def d_lower(c):
        if c >= 65 and c <= 90:
            return c + 32
        return c

    if hasattr(cu.atomic, "cas"):
        @cu.jit(device=True)
        def d_cas(arr, i, old, new):
            return cu.atomic.cas(arr, i, old, new)
    else:                                             # eski numba: faqat 0-element
        @cu.jit(device=True)
        def d_cas(arr, i, old, new):
            return cu.atomic.compare_and_swap(arr[i:], old, new)

    @cu.jit(device=True)
    def d_insert(keys, counts, pos, lens, h, p, ln):
        """Umumiy GPU xesh-jadvaliga qo'shish: bo'sh katakni atomik CAS bilan
        egallash, sonni atomik ADD bilan oshirish. 1 - yangi so'z."""
        mask = keys.shape[0] - 1
        i = np.int64(h & np.uint64(mask))
        while True:
            k = keys[i]
            if k == h:
                cu.atomic.add(counts, i, ONE)
                return 0
            if k == ZERO:
                old = d_cas(keys, i, ZERO, h)
                if old == ZERO:
                    pos[i] = p
                    lens[i] = ln
                    cu.atomic.add(counts, i, ONE)
                    return 1
                if old == h:
                    cu.atomic.add(counts, i, ONE)
                    return 0
            i = (i + 1) & mask

    @cu.jit(device=True)
    def d_entity(buf, i, de):
        j = i + 1
        n = 0
        while j < de and n < 10 and (d_alpha(buf[j]) or (buf[j] >= 48 and buf[j] <= 57)
                                     or buf[j] == 35):
            j += 1
            n += 1
        if j < de and buf[j] == 59 and n > 0:
            return j + 1
        return i + 1

    @cu.jit(device=True)
    def d_seg_end(k, seg_start, seg_doc, doc_end):
        n = seg_start.shape[0]
        if k + 1 < n and seg_doc[k + 1] == seg_doc[k]:
            return seg_start[k + 1]
        return doc_end[seg_doc[k]]

    @cu.jit(device=True)
    def d_scan(buf, s, e, de, st, table, stop, keys, counts, pos, lens, used, cap_lim,
               overflow, base):
        """Bitta segmentni sanash (3-topshiriqdagi _scan_doc bilan bir xil natija),
        lekin HTML holati DFA jadvali orqali - segment istalgan holatdan
        boshlanishi mumkin."""
        ntok = 0
        run = -1
        i = s
        while True:
            c = 32
            step = 1
            if i < e:
                c = buf[i]
                if st == LT:
                    if d_alpha(c) or c == 47 or c == 33:
                        st = table[np.int64(st) * 256 + np.int64(c)]
                        i += 1
                        continue
                    st = T                               # '<' ajratuvchi edi
                if st == T:
                    if d_alpha(c) or c == 39:
                        if run < 0:
                            run = i
                        i += 1
                        continue
                    if c >= 128:
                        if c == 0xE2 and i + 2 < de and buf[i + 1] == 0x80 and \
                                buf[i + 2] >= 0x93 and buf[i + 2] <= 0x9D:
                            step = 3
                        elif c == 0xC2 and i + 1 < de and buf[i + 1] == 0xA0:
                            step = 2
                        else:
                            if run < 0:
                                run = i
                            i += 1
                            continue
                else:
                    st = table[np.int64(st) * 256 + np.int64(c)]             # teg / skript / izoh ichida
                    i += 1
                    continue
            if run >= 0:
                a = run
                b = i
                while a < b and buf[a] == 39:
                    a += 1
                while b > a and buf[b - 1] == 39:
                    b -= 1
                if b - a >= 3:
                    h = OFF
                    for j in range(a, b):
                        h = (h ^ np.uint64(d_lower(buf[j]))) * PRIME
                    if h == ZERO:
                        h = ONE
                    is_stop = False
                    for t in range(stop.shape[0]):
                        if stop[t] == h:
                            is_stop = True
                            break
                    if not is_stop:
                        ntok += 1
                        if used[0] < cap_lim:
                            if d_insert(keys, counts, pos, lens, h, base + a, b - a) == 1:
                                cu.atomic.add(used, 0, ONE)
                        else:
                            overflow[0] = 1
                run = -1
            if i >= e:
                break
            if c == 60:
                st = LT
                i += 1
            elif c == 38:
                i = d_entity(buf, i, de)
            else:
                i += step
        return ntok

    @cu.jit
    def k_zero(keys, counts):
        """Xesh-jadvalni GPU ning o'zida nollash (Host->Device ko'chirishsiz)."""
        i = cu.grid(1)
        if i < keys.shape[0]:
            keys[i] = ZERO
            counts[i] = ZERO

    @cu.jit
    def k_bound(buf, seg_nom, seg_doc, seg_first, doc_end, seg_start):
        """Segment chegarasini keyingi '>' dan keyingi joyga surish."""
        k = cu.grid(1)
        if k < seg_nom.shape[0]:
            if seg_first[k] == 1:
                seg_start[k] = seg_nom[k]
            else:
                de = doc_end[seg_doc[k]]
                j = seg_nom[k]
                while j < de and buf[j] != 62:
                    j += 1
                seg_start[k] = j + 1 if j < de else de

    @cu.jit
    def k_spec(buf, seg_start, seg_doc, doc_end, table, seg_out):
        """SPEKULYATSIYA: har bir segment 'matn' holatidan boshlanadi deb DFA ni
        yurgizadi va oxirgi holatni yozadi (barcha segmentlar parallel)."""
        k = cu.grid(1)
        if k < seg_start.shape[0]:
            e = d_seg_end(k, seg_start, seg_doc, doc_end)
            st = T
            for i in range(seg_start[k], e):
                st = table[np.int64(st) * 256 + np.int64(buf[i])]
            seg_out[k] = st

    @cu.jit
    def k_fix(buf, seg_start, seg_doc, doc_end, doc_first, doc_cnt, table, seg_in, seg_out):
        """TUZATISH: har bir hujjat uchun bitta oqim segmentlarni ketma-ket
        ko'radi; haqiqiy boshlang'ich holat taxmindan farq qilsa (masalan
        segment <script> ichida boshlangan), faqat o'sha segmentni qayta yurgizadi."""
        d = cu.grid(1)
        if d < doc_first.shape[0]:
            st = T
            for j in range(doc_cnt[d]):
                k = doc_first[d] + j
                seg_in[k] = st
                if st != T:
                    x = st
                    for i in range(seg_start[k], d_seg_end(k, seg_start, seg_doc, doc_end)):
                        x = table[np.int64(x) * 256 + np.int64(buf[i])]
                    seg_out[k] = x
                st = seg_out[k]

    @cu.jit
    def k_count(buf, seg_start, seg_doc, doc_end, table, stop, seg_in, keys, counts, pos,
                lens, used, cap_lim, overflow, total, base):
        """ASOSIY YADRO: har bir CUDA oqimi bitta segmentdagi so'zlarni sanaydi."""
        k = cu.grid(1)
        if k < seg_start.shape[0]:
            e = d_seg_end(k, seg_start, seg_doc, doc_end)
            de = doc_end[seg_doc[k]]
            nt = d_scan(buf, seg_start[k], e, de, seg_in[k], table, stop, keys, counts,
                        pos, lens, used, cap_lim, overflow, base)
            if nt > 0:
                cu.atomic.add(total, 0, np.uint64(nt))

    @cu.jit
    def k_compact(keys, counts, pos, lens, n_out, ok, oc, op, ol):
        """Band kataklarni ixcham massivga yig'ish - D2H faqat natijani ko'chiradi."""
        i = cu.grid(1)
        if i < keys.shape[0] and keys[i] != ZERO:
            j = cu.atomic.add(n_out, 0, ONE)
            ok[j] = keys[i]
            oc[j] = counts[i]
            op[j] = pos[i]
            ol[j] = lens[i]

    @cu.jit
    def k_freq32(counts, total, out):
        """Nisbiy chastota float32 da (GPU uchun odatiy aniqlik)."""
        i = cu.grid(1)
        if i < counts.shape[0]:
            out[i] = np.float32(counts[i]) / np.float32(total[0])

    @cu.jit
    def k_freq64(counts, total, out):
        """Nisbiy chastota float64 da."""
        i = cu.grid(1)
        if i < counts.shape[0]:
            out[i] = np.float64(counts[i]) / np.float64(total[0])

    @cu.jit
    def k_sum32(x, acc):
        """Chastotalar yig'indisi (=1 bo'lishi kerak) float32 atomik qo'shish bilan."""
        i = cu.grid(1)
        if i < x.shape[0]:
            cu.atomic.add(acc, 0, x[i])

    @cu.jit
    def k_sum64(x, acc):
        i = cu.grid(1)
        if i < x.shape[0]:
            cu.atomic.add(acc, 0, x[i])

    return SimpleNamespace(**{k: v for k, v in locals().items() if k.startswith("k_")})


@muallif(OTABEK, 4)
def get_gpu():
    """Yadrolarni bir marta yasaydi (keyingi chaqiruvlar keshdan)."""
    global _GPU
    if _GPU is None:
        _GPU = build_gpu_kernels()
    return _GPU


@muallif(OTABEK, 4)
def _grid(n, tpb):
    """Bloklar soni: ceil(n / tpb)."""
    return max(1, (int(n) + tpb - 1) // tpb)


class GpuTableFull(Exception):
    """GPU xesh-jadvali to'lganini bildiradi (jadval kattalashtirilib qaytariladi)."""


class GpuCounter:
    """GPU dagi so'z hisoblagichi. Xesh-jadval GPU xotirasida turadi; add()
    korpus bo'lagini (bir yoki bir nechta) qo'shadi, result() natijani oladi.
    Bir nechta CUDA stream bilan bo'laklar parallel yuborilishi mumkin."""

    @muallif(OTABEK, 4)
    def __init__(self, cap=1 << 20, tpb=256, seg=2048):
        k = get_gpu()
        self.k, self.cap, self.tpb, self.seg = k, cap, tpb, seg
        self.keys = cuda.device_array(cap, np.uint64)
        self.counts = cuda.device_array(cap, np.uint64)
        self.pos = cuda.device_array(cap, np.int64)
        self.lens = cuda.device_array(cap, np.int64)
        self.used = cuda.device_array(1, np.uint64)
        self.overflow = cuda.device_array(1, np.int32)
        self.total = cuda.device_array(1, np.uint64)
        self.table = cuda.to_device(markup_dfa())
        self.stop = cuda.to_device(stop_hashes())
        self._keep = []
        self.reset()

    @muallif(OTABEK, 4)
    def reset(self, stream=0):
        """Jadval va hisoblagichlarni nollash (GPU ichida)."""
        self.k.k_zero[_grid(self.cap, 256), 256, stream](self.keys, self.counts)
        for arr, z in ((self.used, np.zeros(1, np.uint64)), (self.overflow, np.zeros(1, np.int32)),
                       (self.total, np.zeros(1, np.uint64))):
            arr.copy_to_device(z, stream=stream)
        self._keep = []

    @muallif(OTABEK, 4)
    def add(self, buf, offs, base=0, stream=0, ev=None):
        """Korpus bo'lagini (buf, offs - lokal) GPU ga yuborib sanaydi. ev -
        CUDA hodisalari lug'ati: bosqichlar vaqtini o'lchash uchun."""
        k, tpb = self.k, self.tpb

        def mark(name):
            if ev is not None:
                ev[name] = cuda.event(timing=True)
                ev[name].record(stream=stream)
        nom, sdoc, sfirst, dfirst, dcnt = make_segments(offs, self.seg)
        doc_end = np.ascontiguousarray(offs[1:], dtype=np.int64)
        mark("h2d0")
        d_buf = cuda.to_device(buf, stream=stream)
        d_nom = cuda.to_device(nom, stream=stream)
        d_sdoc = cuda.to_device(sdoc, stream=stream)
        d_sfirst = cuda.to_device(sfirst, stream=stream)
        d_dend = cuda.to_device(doc_end, stream=stream)
        d_dfirst = cuda.to_device(dfirst, stream=stream)
        d_dcnt = cuda.to_device(dcnt, stream=stream)
        mark("h2d1")
        n = len(nom)
        d_start = cuda.device_array(n, np.int64, stream=stream)
        d_in = cuda.device_array(n, np.uint8, stream=stream)
        d_out = cuda.device_array(n, np.uint8, stream=stream)
        g = _grid(n, tpb)
        k.k_bound[g, tpb, stream](d_buf, d_nom, d_sdoc, d_sfirst, d_dend, d_start)
        mark("spec0")
        k.k_spec[g, tpb, stream](d_buf, d_start, d_sdoc, d_dend, self.table, d_out)
        mark("spec1")
        k.k_fix[_grid(len(dfirst), tpb), tpb, stream](d_buf, d_start, d_sdoc, d_dend, d_dfirst,
                                                      d_dcnt, self.table, d_in, d_out)
        mark("count0")
        k.k_count[g, tpb, stream](d_buf, d_start, d_sdoc, d_dend, self.table, self.stop, d_in,
                                  self.keys, self.counts, self.pos, self.lens, self.used,
                                  self.cap // 2, self.overflow, self.total, int(base))
        mark("count1")
        self._keep.append((d_buf, d_nom, d_sdoc, d_sfirst, d_dend, d_dfirst, d_dcnt, d_start,
                           d_in, d_out))
        return n

    @muallif(OTABEK, 4)
    def result(self, stream=0, ev=None):
        """Natijani ixchamlab Host ga ko'chiradi (Device->Host)."""
        k = self.k
        n_out = cuda.to_device(np.zeros(1, np.uint64), stream=stream)
        ok = cuda.device_array(self.cap // 2 + 1024, np.uint64, stream=stream)
        oc = cuda.device_array(self.cap // 2 + 1024, np.uint64, stream=stream)
        op = cuda.device_array(self.cap // 2 + 1024, np.int64, stream=stream)
        ol = cuda.device_array(self.cap // 2 + 1024, np.int64, stream=stream)
        k.k_compact[_grid(self.cap, 256), 256, stream](self.keys, self.counts, self.pos,
                                                       self.lens, n_out, ok, oc, op, ol)
        if ev is not None:
            ev["d2h0"] = cuda.event(timing=True)
            ev["d2h0"].record(stream=stream)
        m = int(n_out.copy_to_host(stream=stream)[0])
        if stream:
            stream.synchronize()
        if int(self.overflow.copy_to_host()[0]):
            raise GpuTableFull("GPU xesh-jadvali to'ldi")
        if m > 0:
            hk, hc = ok[:m].copy_to_host(), oc[:m].copy_to_host().astype(np.int64)
            hp, hl = op[:m].copy_to_host(), ol[:m].copy_to_host()
        else:
            hk, hc = np.zeros(0, np.uint64), np.zeros(0, np.int64)
            hp, hl = np.zeros(0, np.int64), np.zeros(0, np.int64)
        tot = self.total.copy_to_host().astype(np.int64)
        if ev is not None:
            ev["d2h1"] = cuda.event(timing=True)
            ev["d2h1"].record(stream=stream)
            ev["d2h1"].synchronize()
        self._keep = []
        return _compact(hk, hc, hp, hl, tot)


@muallif(ELBEK, 4)
def split_by_fraction(offs, f):
    """Hujjatlarni baytlar ulushi bo'yicha ikkiga bo'ladi: [0, k) va [k, n)."""
    if f <= 0:
        return 0
    if f >= 1:
        return len(offs) - 1
    return int(np.searchsorted(offs, f * float(offs[-1])))


_GC_CACHE = {}                                       # GPU xotirasini qayta ajratmaslik uchun
GPU_CAP = 1 << 20                                    # xesh-jadval kataklari (2^20 = 1 048 576)


@muallif(ELBEK, 4)
def gpu_count_words(buf, offs, tpb=256, streams=1, seg=2048, cap=None, ev=None):
    """To'liq GPU yo'li: H2D -> yadrolar -> D2H. streams > 1 bo'lsa korpus
    bo'laklarga bo'linadi va har bir bo'lak o'z CUDA stream'ida yuboriladi -
    bo'lak i+1 ko'chirilayotganda bo'lak i hisoblanadi (pinned xotira bilan)."""
    cap = cap or GPU_CAP
    while True:
        try:
            key = (cap, tpb, seg)
            gc = _GC_CACHE.get(key)
            if gc is None:
                gc = _GC_CACHE[key] = GpuCounter(cap, tpb, seg)
            else:
                gc.reset()
            if streams <= 1:
                gc.add(buf, offs, 0, 0, ev)
                return gc.result(0, ev)
            plan = split_docs_by_bytes(offs, streams)
            ss = [cuda.stream() for _ in range(streams)]
            for i in range(streams):
                a, b = plan[i], plan[i + 1]
                if b > a:
                    gc.add(buf[int(offs[a]):int(offs[b])], offs[a:b + 1] - offs[a],
                           int(offs[a]), ss[i])
            for s in ss:
                s.synchronize()
            return gc.result()
        except GpuTableFull:                     # jadval to'ldi -> 4 marta kattaroq
            cap *= 4


# ======================= BOLTAYEV ELBEK: geterogen sxema =====================
@muallif(ELBEK, 4)
def merge_results(parts):
    """Bir nechta qisman natijani (GPU va CPU) bitta natijaga birlashtirish."""
    parts = [p for p in parts if p is not None and len(p["keys"])]
    if not parts:
        return _compact(np.zeros(0, np.uint64), np.zeros(0, np.int64), np.zeros(0, np.int64),
                        np.zeros(0, np.int64), np.zeros(1, np.int64))
    k = np.concatenate([p["keys"] for p in parts])
    c = np.concatenate([p["counts"] for p in parts]).astype(np.int64)
    po = np.concatenate([p["pos"] for p in parts]).astype(np.int64)
    ln = np.concatenate([p["lens"] for p in parts]).astype(np.int64)
    mk, mc, mp_, ml = _merge_entries(k, c, po, ln, _next_pow2(max(1024, 2 * len(k))))
    return _compact(mk, mc, mp_, ml, np.array([sum(p["total"] for p in parts)], np.int64))


@muallif(ELBEK, 4)
def _timed(fn, *a, **kw):
    """fn ni bajarib, (natija, sarflangan vaqt) qaytaradi."""
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return r, time.perf_counter() - t0


@muallif(ELBEK, 4)
def hetero_static(buf, gbuf, offs, f, threads, gcfg):
    """STATIK NISBAT: baytlarning f ulushi GPU ga, qolgani CPU ga (Numba prange,
    siklik taqsimot). Ikkalasi ThreadPoolExecutor orqali BIR VAQTDA ishlaydi
    (CPU yadrosi nogil, CUDA chaqiruvlari ham GIL ni bo'shatadi).
    gbuf - GPU uchun (pinned bo'lishi mumkin) bufer."""
    k = split_by_fraction(offs, f)
    o = int(offs[k])
    tg = tc = 0.0
    rg = rc = None
    with ThreadPoolExecutor(max_workers=2) as ex:
        fg = ex.submit(_timed, gpu_count_words, gbuf[:o], offs[:k + 1], **gcfg) if k > 0 else None
        fc = (ex.submit(_timed, count_words_parallel, buf[o:], offs[k:] - o, threads, 0,
                        cyclic=True) if k < len(offs) - 1 else None)
        if fg:
            rg, tg = fg.result()
        if fc:
            rc, tc = fc.result()
            rc["pos"] = rc["pos"] + o                 # CPU qismida joy global bo'lsin
    res = merge_results([rg, rc])
    res["t_gpu"], res["t_cpu"], res["f_bytes"] = tg, tc, o / float(offs[-1])
    return res


@muallif(ELBEK, 4)
def hetero_dynamic(buf, gbuf, offs, threads, gcfg, nchunks=32):
    """DINAMIK NAVBAT: korpus nchunks bo'lakka bo'linadi va queue.Queue ga
    qo'yiladi. GPU ishchisi va CPU ishchisi bo'shashi bilan navbatdan keyingi
    bo'lakni oladi - tezroq qurilma ko'proq bo'lak oladi, nisbatni oldindan
    bilish shart emas."""
    plan = split_docs_by_bytes(offs, nchunks)
    q = queue.Queue()
    for i in range(nchunks):
        if plan[i + 1] > plan[i]:
            q.put((plan[i], plan[i + 1]))
    gc = GpuCounter(gcfg.get("cap", GPU_CAP), gcfg.get("tpb", 256), gcfg.get("seg", 2048))
    st = cuda.stream()
    done = {"gpu": 0, "cpu": 0}
    cpu_parts = []

    def gpu_worker():
        while True:
            try:
                a, b = q.get_nowait()
            except queue.Empty:
                return
            o = int(offs[a])
            gc.add(gbuf[o:int(offs[b])], offs[a:b + 1] - o, o, st)
            st.synchronize()
            done["gpu"] += int(offs[b]) - o

    def cpu_worker():
        while True:
            try:
                a, b = q.get_nowait()
            except queue.Empty:
                return
            o = int(offs[a])
            r = count_words_parallel(buf[o:int(offs[b])], offs[a:b + 1] - o, threads, 0,
                                     cyclic=True)
            r["pos"] = r["pos"] + o
            cpu_parts.append(r)
            done["cpu"] += int(offs[b]) - o
    with ThreadPoolExecutor(max_workers=2) as ex:
        for f in [ex.submit(gpu_worker), ex.submit(cpu_worker)]:
            f.result()
    res = merge_results([gc.result()] + cpu_parts)
    res["f_bytes"] = done["gpu"] / float(offs[-1])
    return res


@muallif(ELBEK, 4)
def gpu_env():
    """8-band: GPU modeli, hisoblash qobiliyati, xotira, CUDA versiyalari va
    muhit (mahalliy kompyuter yoki Google Colab)."""
    env = {}
    try:
        dev = cuda.get_current_device()
        name = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
        env["GPU"] = name
        env["Compute capability"] = ".".join(map(str, dev.compute_capability))
        env["Multiprotsessorlar (SM)"] = getattr(dev, "MULTIPROCESSOR_COUNT", "?")
        free, total = cuda.current_context().get_memory_info()
        env["GPU xotirasi"] = f"{total / 2 ** 30:.1f} GB"
    except Exception as e:                            # noqa: BLE001
        env["GPU"] = f"aniqlanmadi ({e})"
    try:
        env["CUDA runtime"] = ".".join(map(str, cuda.runtime.get_version()))
    except Exception:
        pass
    try:
        from numba.cuda.cudadrv.driver import driver
        env["CUDA driver"] = ".".join(map(str, driver.get_version()))
    except Exception:
        pass
    try:
        import numba_cuda
        env["numba-cuda"] = numba_cuda.__version__
    except Exception:
        pass
    colab = "google.colab" in sys.modules or "COLAB_GPU" in os.environ or \
        os.path.exists("/content")
    env["Muhit"] = "Google Colab (bulutli)" if colab else "mahalliy kompyuter"
    return env


@muallif(ELBEK, 5)
def gpu_diagnostics():
    """GPU nima uchun topilmayotganini aniqlash (exe uchun ayniqsa foydali):
    import xatosi, cuda.is_available(), drayver, numba-cuda va NVIDIA papkalari."""
    import traceback
    lines = [f"python: {sys.executable}", f"frozen (exe): {getattr(sys, 'frozen', False)}",
             f"numba: {getattr(numba, '__version__', 'yo') if HAVE_NUMBA else 'yo'}"]
    lines.append(f"numba.cuda import: {'OK' if cuda is not None else CUDA_IMPORT_ERROR}")
    if cuda is None:
        try:
            from numba import cuda as _c  # noqa: F401
        except Exception:
            lines.append(traceback.format_exc())
    else:
        try:
            lines.append(f"cuda.is_available(): {cuda.is_available()}")
            lines.append(f"GPU: {gpu_env().get('GPU')}")
        except Exception:
            lines.append(traceback.format_exc())
    base = getattr(sys, "_MEIPASS", None)
    if base:
        for name in ("numba_cuda", "nvidia", "_pth", "_numba_cuda_redirector.py"):
            lines.append(f"{name}: {'bor' if os.path.exists(os.path.join(base, name)) else 'YOQ'}")
    lines.append("nvidia-smi: " + (shutil.which("nvidia-smi") or "topilmadi"))
    return "\n".join(lines)


COLAB_HELP = """GPU (NVIDIA CUDA) topilmadi. Variantlar:
 1) Mahalliy kompyuterda NVIDIA GPU bo'lsa: NVIDIA drayverini yangilang va
       pip install "numba-cuda[cu12]"
 2) Google Colab (bepul GPU):
       colab.research.google.com -> Runtime -> Change runtime type -> T4 GPU
       parallel_skreyper.py ni yuklang (chapdagi papka belgisi), keyin:
       !pip install -q "numba-cuda[cu12]"
       !python parallel_skreyper.py --test4
       !python parallel_skreyper.py --task4
       !zip -r natijalar4.zip natijalar4 -x "*.npz"
"""


# ============================== 4-topshiriq tajribalari =======================
@muallif(OTABEK, 4)
def gpu_measure(fn, repeats=3, warmup=2):
    """GPU uchun o'lchov: har bir yurish oxirida cuda.synchronize() - asinxron
    ishlar tugaganini kafolatlaydi; birinchi yurishlar (JIT) hisobga olinmaydi."""
    def run():
        r = fn()
        cuda.synchronize()
        return r
    return measure_stable(run, repeats, warmup)


@muallif(OTABEK, 4)
def t4_exp_baseline(ctx):
    """1-band: CPU etalonlari (ketma-ket va prange 8 oqim) va GPU (JIT vaqti
    alohida o'lchanadi)."""
    LOG("\n[1] GPU ga ko'chirish: CPU va GPU etalonlari ...")
    buf, offs = ctx["buf"], ctx["offs"]
    tb, to = pack_corpus([generate_page(i, 20) for i in range(4)])
    t0 = time.perf_counter()
    gpu_count_words(tb, to)
    cuda.synchronize()
    t1 = time.perf_counter()
    gpu_count_words(tb, to)
    cuda.synchronize()
    t2 = time.perf_counter()
    ctx["res"]["jit_gpu"] = (t1 - t0) - (t2 - t1)
    LOG(f"    CUDA JIT kompilyatsiya: {ctx['res']['jit_gpu']:.2f} s (o'lchovga kirmaydi)")
    seq = measure_stable(lambda: count_words_seq(buf, offs), ctx["repeats"])
    ctx["ref"] = seq["result"]
    cpu = measure_stable(lambda: count_words_parallel(buf, offs, ctx["threads"], 0, cyclic=True),
                         ctx["repeats"])
    check(ctx, f"CPU prange {ctx['threads']} oqim == ketma-ket",
          same_counts(cpu["result"], ctx["ref"])[0])
    gpu = gpu_measure(lambda: gpu_count_words(buf, offs, 256), ctx["repeats"])
    ok, diff = same_counts(gpu["result"], ctx["ref"])
    check(ctx, "GPU (tpb=256) == CPU ketma-ket", ok, f"(max farq {diff:.1e})")
    mb = float(offs[-1]) / 2 ** 20
    ctx["res"]["base"] = {"seq": seq["mean"], "seq_std": seq["std"], "cpu": cpu["mean"],
                          "cpu_std": cpu["std"], "gpu": gpu["mean"], "gpu_std": gpu["std"],
                          "mb": mb, "threads": ctx["threads"]}
    LOG(f"    ketma-ket {seq['mean']:.4f} s | CPU {ctx['threads']} oqim {cpu['mean']:.4f} s | "
        f"GPU (to'liq) {gpu['mean']:.4f} s  -> S(GPU/CPU) = {cpu['mean'] / gpu['mean']:.2f}")


@muallif(OTABEK, 4)
def _phase_times(ev):
    """CUDA hodisalaridan bosqichlar vaqti (s)."""
    for e in ev.values():
        e.synchronize()

    def dt(a, b):
        return cuda.event_elapsed_time(ev[a], ev[b]) / 1000.0
    return {"h2d": dt("h2d0", "h2d1"), "spec": dt("spec0", "spec1"),
            "count": dt("count0", "count1"), "kernels": dt("h2d1", "count1") + dt("count1", "d2h0"),
            "d2h": dt("d2h0", "d2h1"), "total": dt("h2d0", "d2h1")}


@muallif(OTABEK, 4)
def t4_exp_block(ctx):
    """2-band: blok o'lchami (threads_per_block) 64...1024 - yadro vaqti CUDA
    hodisalari bilan. Ishga tushmasa (registrlar yetmasa) - qayd etiladi."""
    LOG("\n[2] Blok o'lchami (threads_per_block) ...")
    buf, offs = ctx["gbuf_pinned"], ctx["offs"]
    rows = []
    for tpb in ctx["tpbs"]:
        try:
            samples = []
            for r in range(ctx["repeats"] + 2):
                ev = {}
                res = gpu_count_words(buf, offs, tpb, ev=ev)
                if r >= 2:
                    samples.append(_phase_times(ev))
            ok = same_counts(res, ctx["ref"])[0]
            check(ctx, f"GPU tpb={tpb} == CPU", ok)
            row = {"tpb": tpb, "ok": True}
            for key in ("spec", "count", "kernels", "total"):
                vals = [s[key] for s in samples]
                row[key] = statistics.mean(vals)
                row[key + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            try:
                row["regs"] = int(get_gpu().k_count.get_regs_per_thread())
            except Exception:
                row["regs"] = None
            LOG(f"    tpb={tpb:5d}: k_count {1000 * row['count']:.2f} ms, k_spec "
                f"{1000 * row['spec']:.2f} ms, barcha yadrolar {1000 * row['kernels']:.2f} ms")
        except Exception as e:                        # noqa: BLE001
            row = {"tpb": tpb, "ok": False, "error": str(e)[:120]}
            LOG(f"    tpb={tpb:5d}: ishga tushmadi - {str(e)[:100]}")
        rows.append(row)
    ctx["res"]["block"] = rows
    good = [r for r in rows if r["ok"]]
    ctx["best_tpb"] = min(good, key=lambda r: r["kernels"])["tpb"] if good else 256


@muallif(OTABEK, 4)
def t4_exp_phases(ctx):
    """3-band: Host->Device, yadrolar va Device->Host vaqtlari alohida
    (oddiy va pinned xotira), ularning ulushi va ikki xil tezlashish."""
    LOG("\n[3] Xotira almashinuvi: H2D / yadrolar / D2H ...")
    out = {}
    for name, b in (("oddiy (pageable)", ctx["buf"]), ("pinned", ctx["gbuf_pinned"])):
        samples = []
        for r in range(ctx["repeats"] + 2):
            ev = {}
            res = gpu_count_words(b, ctx["offs"], ctx["best_tpb"], ev=ev)
            if r >= 2:
                samples.append(_phase_times(ev))
        check(ctx, f"GPU {name} == CPU", same_counts(res, ctx["ref"])[0])
        row = {k: statistics.mean(s[k] for s in samples) for k in samples[0]}
        row["other"] = max(0.0, row["total"] - row["h2d"] - row["kernels"] - row["d2h"])
        out[name] = row
        LOG(f"    {name:17s}: H2D {1000 * row['h2d']:.2f} ms | yadrolar {1000 * row['kernels']:.2f}"
            f" ms | D2H {1000 * row['d2h']:.2f} ms | jami {1000 * row['total']:.2f} ms | "
            f"ko'chirish ulushi {100 * (row['h2d'] + row['d2h']) / row['total']:.1f}%")
    ctx["res"]["phases"] = out
    ctx["res"]["pin_time"] = ctx["pin_time"]


@muallif(ELBEK, 4)
def t4_exp_overlap(ctx):
    """6-band: ko'chirish va hisoblashni ustma-ust qo'yish - oddiy xotira,
    pinned, pinned + 2/4/8 CUDA stream."""
    LOG("\n[6] Pinned xotira va CUDA stream (overlap) ...")
    rows = []
    cfgs = [("oddiy xotira, 1 stream", ctx["buf"], 1), ("pinned, 1 stream", ctx["gbuf_pinned"], 1)]
    cfgs += [(f"pinned, {s} stream", ctx["gbuf_pinned"], s) for s in ctx["streams"] if s > 1]
    for name, b, s in cfgs:
        m = gpu_measure(lambda: gpu_count_words(b, ctx["offs"], ctx["best_tpb"], s), ctx["repeats"])
        check(ctx, f"GPU {name} == CPU", same_counts(m["result"], ctx["ref"])[0])
        rows.append({"config": name, "streams": s, "mean": m["mean"], "std": m["std"]})
        LOG(f"    {name:24s}: {1000 * m['mean']:.2f} ms")
    for r in rows:
        r["speedup"] = rows[0]["mean"] / r["mean"]
    ctx["res"]["overlap"] = rows
    best = min(rows[1:], key=lambda r: r["mean"])
    ctx["gcfg"] = {"tpb": ctx["best_tpb"], "streams": best["streams"]}


@muallif(ELBEK, 4)
def t4_exp_hetero(ctx):
    """4-, 5-band: GPU ulushi 0, 25, 50, 75, 90, 100% (statik nisbat) va
    dinamik navbat; nazariy optimal nisbat f* = R_gpu / (R_gpu + R_cpu)."""
    LOG("\n[4-5] Geterogen CPU + GPU ...")
    buf, gbuf, offs = ctx["buf"], ctx["gbuf_pinned"], ctx["offs"]
    gcfg, th = ctx["gcfg"], ctx["threads"]
    b = ctx["res"]["base"]
    t_gpu = min(r["mean"] for r in ctx["res"]["overlap"][1:])
    r_gpu, r_cpu = b["mb"] / t_gpu, b["mb"] / b["cpu"]
    f_star = r_gpu / (r_gpu + r_cpu)
    ratios = sorted(set(ctx["ratios"] + [round(f_star, 3)]))
    rows = []
    for f in ratios:
        m = measure_stable(lambda: hetero_static(buf, gbuf, offs, f, th, gcfg), ctx["repeats"])
        r = m["result"]
        ok = same_counts(r, ctx["ref"])[0]
        check(ctx, f"geterogen f={f:.3f} == CPU", ok)
        rows.append({"f": f, "f_bytes": r["f_bytes"], "mean": m["mean"], "std": m["std"],
                     "t_gpu": r["t_gpu"], "t_cpu": r["t_cpu"], "predicted": f == round(f_star, 3)})
        LOG(f"    GPU ulushi {100 * f:5.1f}%: {1000 * m['mean']:.2f} ms (GPU qismi "
            f"{1000 * r['t_gpu']:.1f} ms, CPU qismi {1000 * r['t_cpu']:.1f} ms)")
    dyn = measure_stable(lambda: hetero_dynamic(buf, gbuf, offs, th, gcfg, ctx["chunks"]),
                         ctx["repeats"])
    check(ctx, "geterogen dinamik navbat == CPU", same_counts(dyn["result"], ctx["ref"])[0])
    LOG(f"    dinamik navbat ({ctx['chunks']} bo'lak): {1000 * dyn['mean']:.2f} ms, GPU olgan "
        f"ulush {100 * dyn['result']['f_bytes']:.1f}%")
    ctx["res"]["hetero"] = {"rows": rows, "f_star": f_star, "r_gpu": r_gpu, "r_cpu": r_cpu,
                            "dynamic": {"mean": dyn["mean"], "std": dyn["std"],
                                        "f_bytes": dyn["result"]["f_bytes"],
                                        "chunks": ctx["chunks"]}}


@muallif(OTABEK, 4)
def t4_exp_precision(ctx):
    """7-band: butun sonlar (so'zlar soni) aniq teng; nisbiy chastotalar GPU da
    float32 va float64 da hisoblanib CPU (float64) bilan solishtiriladi;
    chastotalar yig'indisi (=1) atomik qo'shish bilan."""
    LOG("\n[7] Aniqlik: float32 va float64 ...")
    ref = ctx["ref"]
    k = get_gpu()
    cnt = cuda.to_device(ref["counts"].astype(np.uint64))
    tot = cuda.to_device(np.array([ref["total"]], np.uint64))
    n = len(ref["counts"])
    f32 = cuda.device_array(n, np.float32)
    f64 = cuda.device_array(n, np.float64)
    k.k_freq32[_grid(n, 256), 256](cnt, tot, f32)
    k.k_freq64[_grid(n, 256), 256](cnt, tot, f64)
    cpu = ref["counts"].astype(np.float64) / float(ref["total"])
    g32, g64 = f32.copy_to_host().astype(np.float64), f64.copy_to_host()
    rel32 = float(np.max(np.abs(g32 - cpu) / cpu))
    rel64 = float(np.max(np.abs(g64 - cpu) / cpu))
    sums = {}
    for name, arr, dt in (("float32", f32, np.float32), ("float64", f64, np.float64)):
        vals = []
        for _ in range(3):
            acc = cuda.to_device(np.zeros(1, dt))
            (k.k_sum32 if dt == np.float32 else k.k_sum64)[_grid(n, 256), 256](arr, acc)
            vals.append(float(acc.copy_to_host()[0]))
        sums[name] = vals
    cpu_sum = float(np.sum(cpu))
    out = {"n_words": n, "total": int(ref["total"]), "rel32": rel32, "rel64": rel64,
           "eps32": float(np.finfo(np.float32).eps), "eps64": float(np.finfo(np.float64).eps),
           "sum32": sums["float32"], "sum64": sums["float64"], "cpu_sum": cpu_sum,
           "total_exact_f32": int(np.float32(ref["total"])) == int(ref["total"])}
    check(ctx, "float64 chastotalar: nisbiy xato <= 1e-12", rel64 <= 1e-12, f"({rel64:.1e})")
    check(ctx, "float32 chastotalar: nisbiy xato <= 1e-6", rel32 <= 1e-6, f"({rel32:.1e})")
    LOG(f"    nisbiy xato: float32 {rel32:.2e}, float64 {rel64:.2e}; yig'indi float32 "
        f"{sums['float32']}, float64 {sums['float64']}")
    ctx["res"]["precision"] = out


@muallif(OTABEK, 4)
def t4_conclusions(res):
    """4-topshiriq xulosasi o'lchovlardan avtomatik."""
    b, ph, het = res["base"], res["phases"]["pinned"], res["hetero"]
    good = [r for r in res["block"] if r["ok"]]
    bb = min(good, key=lambda r: r["kernels"])
    ov = res["overlap"]
    best_ov = min(ov, key=lambda r: r["mean"])
    best_h = min(het["rows"], key=lambda r: r["mean"])
    kshare = 100 * (ph["h2d"] + ph["d2h"]) / ph["total"]
    out = [
        f"GPU (to'liq, ko'chirish bilan) {1000 * b['gpu']:.1f} ms, CPU prange "
        f"{b['threads']} oqim {1000 * b['cpu']:.1f} ms - S = {b['cpu'] / b['gpu']:.2f}; "
        f"faqat yadrolar bo'yicha S = {b['cpu'] / ph['kernels']:.2f}.",
        f"Eng yaxshi blok o'lchami {bb['tpb']} ({1000 * bb['kernels']:.2f} ms). Blok o'lchami "
        f"warp (32) ga karrali; kichik blokda SM dagi bloklar soni chegarasi, katta blokda "
        f"registrlar yetishmasligi egallanganlikni (occupancy) pasaytiradi.",
        f"Ko'chirish (H2D + D2H) pinned xotirada umumiy vaqtning {kshare:.0f}% ini oladi: "
        f"H2D {1000 * ph['h2d']:.1f} ms, yadrolar {1000 * ph['kernels']:.1f} ms.",
        f"Ustma-ust bajarish: eng yaxshisi '{best_ov['config']}' - oddiy versiyadan "
        f"{best_ov['speedup']:.2f} baravar tez.",
        f"Geterogen: nazariy optimal GPU ulushi f* = {100 * het['f_star']:.0f}% "
        f"(GPU {het['r_gpu']:.0f} MB/s, CPU {het['r_cpu']:.0f} MB/s); tajribada eng tezi "
        f"{100 * best_h['f']:.0f}% ({1000 * best_h['mean']:.1f} ms) - "
        + ("nazariy nisbatga mos keladi." if abs(best_h["f"] - het["f_star"]) <= 0.1 else
           "nazariy nisbatdan farq qiladi: CPU va GPU bir vaqtda ishlaganda bir-biriga "
           "xalaqit beradi (GPU ni boshqaruvchi oqim ham CPU yadrosini talab qiladi, "
           "xotira shinasi umumiy).") +
        f" Dinamik navbat {1000 * het['dynamic']['mean']:.1f} ms, GPU o'zi "
        f"{100 * het['dynamic']['f_bytes']:.0f}% ni oldi.",
    ]
    return out


@muallif(ELBEK, 4)
def make_charts4(res, outdir):
    """4-topshiriq grafiklari."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, name), dpi=130)
        plt.close(fig)
        files.append(name)

    good = [r for r in res["block"] if r["ok"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = [str(r["tpb"]) for r in good]
    ax.bar(xs, [1000 * r["count"] for r in good], label="k_count (sanash)",
           yerr=[1000 * r["count_std"] for r in good], capsize=3)
    ax.bar(xs, [1000 * r["spec"] for r in good], bottom=[1000 * r["count"] for r in good],
           label="k_spec (DFA spekulyatsiya)")
    ax.set_xlabel("Blokdagi oqimlar soni, threads_per_block")
    ax.set_ylabel("Yadro vaqti (ms)")
    ax.set_title("Blok o'lchamiga qarab GPU yadrolari vaqti")
    ax.grid(axis="y", alpha=.3)
    ax.legend()
    save(fig, "1_blok_olchami.png")

    ph = res["phases"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = list(ph)
    bottom = [0.0] * len(names)
    for key, lab in (("h2d", "Host→Device"), ("kernels", "yadrolar"), ("d2h", "Device→Host"),
                     ("other", "boshqa (tayyorlash)")):
        vals = [1000 * ph[n][key] for n in names]
        ax.bar(names, vals, bottom=bottom, label=lab)
        bottom = [b_ + v for b_, v in zip(bottom, vals)]
    ax.set_ylabel("Vaqt (ms)")
    ax.set_title("GPU yo'li bosqichlari: ko'chirish va hisoblash")
    ax.set_ylim(0, max(bottom) * 1.45)
    ax.legend(loc="upper center", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=.3)
    save(fig, "2_kochirish.png")

    het = res["hetero"]
    rows = sorted(het["rows"], key=lambda r: r["f"])
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    fx = [100 * r["f"] for r in rows]
    ax.plot(fx, [1000 * r["mean"] for r in rows], "o-", lw=2, label="jami (CPU + GPU bir vaqtda)")
    ax.plot(fx, [1000 * r["t_gpu"] if r["t_gpu"] > 0 else float("nan") for r in rows], "s--",
            label="GPU qismi")
    ax.plot(fx, [1000 * r["t_cpu"] if r["t_cpu"] > 0 else float("nan") for r in rows], "^--",
            label="CPU qismi")
    ax.axvline(100 * het["f_star"], color="gray", ls=":", label=f"nazariy f* = {100 * het['f_star']:.0f}%")
    ax.axhline(1000 * het["dynamic"]["mean"], color="C3", ls="-.",
               label=f"dinamik navbat (GPU {100 * het['dynamic']['f_bytes']:.0f}%)")
    ax.set_xlabel("GPU ga berilgan ulush (baytlar, %)")
    ax.set_ylabel("Bajarilish vaqti (ms)")
    ax.set_yscale("log")
    ax.set_title("Geterogen hisoblash: GPU ulushiga qarab vaqt")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    save(fig, "3_geterogen.png")

    ov = res["overlap"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(range(len(ov)), [1000 * r["mean"] for r in ov],
                  yerr=[1000 * r["std"] for r in ov], capsize=3)
    for bar, r in zip(bars, ov):
        ax.annotate(f"{r['speedup']:.2f}x", (bar.get_x() + bar.get_width() / 2,
                                            bar.get_height() + 1000 * r["std"]),
                    xytext=(0, 4), textcoords="offset points", ha="center")
    ax.set_xticks(range(len(ov)))
    ax.set_xticklabels([r["config"] for r in ov], rotation=12, fontsize=8)
    ax.set_ylabel("To'liq GPU vaqti (ms)")
    ax.set_title("Pinned xotira va CUDA stream: ko'chirish + hisoblash ustma-ust")
    ax.grid(axis="y", alpha=.3)
    ax.set_ylim(0, max(1000 * (r["mean"] + r["std"]) for r in ov) * 1.2)
    save(fig, "4_overlap.png")
    return files


@muallif(ELBEK, 4)
def write_report4(ctx, outdir):
    """natijalar4.md, natijalar4.json, barcha_natijalar4.csv va grafiklar."""
    res = ctx["res"]
    charts = make_charts4(res, outdir)
    with open(os.path.join(outdir, "natijalar4.json"), "w", encoding="utf-8") as f:
        json.dump({"env": ctx["env"], "results": res, "checks": ctx["checks"],
                   "authors": AUTHORS4}, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(outdir, "barcha_natijalar4.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["tajriba", "parametr", "qiymat_ms", "std_ms", "izoh"])
        b = res["base"]
        for k_ in ("seq", "cpu", "gpu"):
            w.writerow(["etalon", k_, f"{1000 * b[k_]:.3f}", f"{1000 * b[k_ + '_std']:.3f}", ""])
        for r in res["block"]:
            if r["ok"]:
                w.writerow(["blok", f"tpb={r['tpb']}", f"{1000 * r['kernels']:.3f}",
                            f"{1000 * r['kernels_std']:.3f}", f"k_count={1000 * r['count']:.3f}"])
            else:
                w.writerow(["blok", f"tpb={r['tpb']}", "", "", r["error"]])
        for n_, r in res["phases"].items():
            w.writerow(["bosqichlar", n_, f"{1000 * r['total']:.3f}", "",
                        f"h2d={1000 * r['h2d']:.3f};yadro={1000 * r['kernels']:.3f};"
                        f"d2h={1000 * r['d2h']:.3f}"])
        for r in res["overlap"]:
            w.writerow(["overlap", r["config"], f"{1000 * r['mean']:.3f}", f"{1000 * r['std']:.3f}",
                        f"S={r['speedup']:.3f}"])
        for r in res["hetero"]["rows"]:
            w.writerow(["geterogen", f"f={r['f']:.3f}", f"{1000 * r['mean']:.3f}",
                        f"{1000 * r['std']:.3f}",
                        f"gpu={1000 * r['t_gpu']:.3f};cpu={1000 * r['t_cpu']:.3f}"])
        d = res["hetero"]["dynamic"]
        w.writerow(["geterogen", "dinamik", f"{1000 * d['mean']:.3f}", f"{1000 * d['std']:.3f}",
                    f"gpu_ulush={d['f_bytes']:.3f}"])
    b, ph, p = res["base"], res["phases"], res["precision"]
    md = ["# Parallel veb-skreyper - 4-topshiriq natijalari (CUDA va geterogen)", "",
          f"**Sana:** {time.strftime('%Y-%m-%d %H:%M')}", "", "## Hissa", "",
          fmt_table(["A'zo", "Funksiyalar"],
                    [[k_, ", ".join(f"`{n_}`" for n_ in v)] for k_, v in AUTHORS4.items() if v]),
          "", "## Sinov muhiti", "",
          fmt_table(["Parametr", "Qiymat"], [[k_, v] for k_, v in ctx["env"].items()]),
          "", f"Korpus: {len(ctx['offs']) - 1} sahifa, {b['mb']:.1f} MB. CUDA JIT: "
          f"{res['jit_gpu']:.2f} s (o'lchovga kirmaydi). Pinned buferga nusxalash: "
          f"{1000 * res['pin_time']:.1f} ms (bir martalik).",
          "", "## Etalonlar", "",
          fmt_table(["Versiya", "T, ms", "std, ms"],
                    [["CPU ketma-ket", f"{1000 * b['seq']:.2f}", f"{1000 * b['seq_std']:.2f}"],
                     [f"CPU prange {b['threads']} oqim", f"{1000 * b['cpu']:.2f}",
                      f"{1000 * b['cpu_std']:.2f}"],
                     ["GPU (to'liq)", f"{1000 * b['gpu']:.2f}", f"{1000 * b['gpu_std']:.2f}"]]),
          "", "## Blok o'lchami", "",
          fmt_table(["tpb", "k_count, ms", "k_spec, ms", "yadrolar, ms", "registr/oqim", "izoh"],
                    [[r["tpb"], f"{1000 * r['count']:.3f}", f"{1000 * r['spec']:.3f}",
                      f"{1000 * r['kernels']:.3f}", r.get("regs") or "-", ""] if r["ok"] else
                     [r["tpb"], "-", "-", "-", "-", r["error"]] for r in res["block"]]),
          "", "## Ko'chirish va hisoblash", "",
          fmt_table(["Xotira", "H2D, ms", "Yadrolar, ms", "D2H, ms", "Jami, ms", "Ko'chirish, %"],
                    [[n_, f"{1000 * r['h2d']:.2f}", f"{1000 * r['kernels']:.2f}",
                      f"{1000 * r['d2h']:.2f}", f"{1000 * r['total']:.2f}",
                      f"{100 * (r['h2d'] + r['d2h']) / r['total']:.1f}"] for n_, r in ph.items()]),
          "", "## Overlap", "",
          fmt_table(["Konfiguratsiya", "T, ms", "std", "S"],
                    [[r["config"], f"{1000 * r['mean']:.2f}", f"{1000 * r['std']:.2f}",
                      f"{r['speedup']:.2f}"] for r in res["overlap"]]),
          "", "## Geterogen", "",
          fmt_table(["GPU ulushi", "Jami, ms", "std", "GPU qismi, ms", "CPU qismi, ms"],
                    [[f"{100 * r['f']:.1f}%" + (" (f*)" if r["predicted"] else ""),
                      f"{1000 * r['mean']:.2f}", f"{1000 * r['std']:.2f}",
                      f"{1000 * r['t_gpu']:.2f}", f"{1000 * r['t_cpu']:.2f}"]
                     for r in res["hetero"]["rows"]]),
          "", f"Dinamik navbat: {1000 * res['hetero']['dynamic']['mean']:.2f} ms, GPU ulushi "
          f"{100 * res['hetero']['dynamic']['f_bytes']:.1f}%.",
          "", "## Aniqlik", "",
          fmt_table(["Ko'rsatkich", "float32", "float64"],
                    [["max nisbiy xato", f"{p['rel32']:.2e}", f"{p['rel64']:.2e}"],
                     ["mashina epsiloni", f"{p['eps32']:.2e}", f"{p['eps64']:.2e}"],
                     ["chastotalar yig'indisi (3 marta)", ", ".join(f"{v:.9f}" for v in p["sum32"]),
                      ", ".join(f"{v:.15f}" for v in p["sum64"])]]),
          "", "## To'g'rilik", "",
          fmt_table(["Tekshiruv", "Natija"],
                    [[c["name"] + " " + c["detail"], "MOS" if c["ok"] else "**XATO**"]
                     for c in ctx["checks"]]),
          "", "## Xulosa", ""]
    md += [f"{i}. {t}" for i, t in enumerate(t4_conclusions(res), 1)]
    md += ["", "## Grafiklar", ""] + [f"![{c}]({c})" for c in charts]
    with open(os.path.join(outdir, "natijalar4.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return charts


@muallif(ELBEK, 4)
def load_corpus4(args):
    """Korpus: 3-topshiriqda yig'ilgan bo'lsa o'shani oladi, aks holda yig'adi."""
    for d in (args.out3, args.out4):
        path = os.path.join(d, "korpus_bir_xil.npz")
        if os.path.exists(path):
            with np.load(path) as z:
                if len(z["offs"]) - 1 == args.t3_pages:
                    return z["buf"].copy(), z["offs"].copy()
    os.makedirs(args.out4, exist_ok=True)
    return build_corpus(args.t3_pages, "bir_xil", os.path.join(args.out4, "korpus_bir_xil.npz"))


@muallif(BOTH, 4)
def run_task4(args):
    """4-topshiriqning barcha tajribalari."""
    if np is None or not HAVE_NUMBA:
        sys.exit("numpy va numba kerak: pip install numpy numba")
    if cuda is None or not cuda.is_available():
        LOG(COLAB_HELP)
        return 1
    os.makedirs(args.out4, exist_ok=True)
    env = collect_env()
    env.update({"numpy": np.__version__, "numba": numba.__version__})
    env.update(gpu_env())
    LOG("4-TOPSHIRIQ. Sinov muhiti: " + ", ".join(f"{k}: {v}" for k, v in env.items()))
    buf, offs = load_corpus4(args)
    t0 = time.perf_counter()
    pinned = cuda.pinned_array(buf.shape, np.uint8)     # page-locked bufer (bir martalik)
    pinned[:] = buf
    pin_time = time.perf_counter() - t0
    ctx = {"args": args, "res": {}, "checks": [], "env": env, "repeats": args.repeats,
           "buf": buf, "offs": offs, "gbuf_pinned": pinned, "pin_time": pin_time,
           "threads": min(8, numba.config.NUMBA_NUM_THREADS), "tpbs": args.t4_tpb,
           "streams": args.t4_streams, "ratios": args.t4_ratios, "chunks": args.t4_chunks}
    LOG(f"Korpus: {len(offs) - 1} sahifa, {offs[-1] / 2 ** 20:.1f} MB; pinned nusxa "
        f"{1000 * pin_time:.1f} ms")
    for exp in (t4_exp_baseline, t4_exp_block, t4_exp_phases, t4_exp_overlap, t4_exp_hetero,
                t4_exp_precision):
        exp(ctx)
    charts = write_report4(ctx, args.out4)
    ok = sum(c["ok"] for c in ctx["checks"])
    LOG(f"\nTo'g'rilik: {ok}/{len(ctx['checks'])} tekshiruv MOS")
    LOG("\nXULOSA:")
    for i, t in enumerate(t4_conclusions(ctx["res"]), 1):
        LOG(f"  {i}. {t}")
    LOG(f"\nNatijalar: {os.path.abspath(args.out4)} (natijalar4.md, natijalar4.json, "
        f"barcha_natijalar4.csv, {len(charts)} ta grafik)")
    return 0


@muallif(ELBEK, 4)
def tricky_docs():
    """Testlar uchun qiyin HTML holatlari (teg, skript, izoh, entity, UTF-8)."""
    return [
        "<html><head><title>Sarlavha Soz</title></head><body><!-- izohsoz -->"
        "<p>Abc<b>DEF</b> x &amp; y&nbsp;zzz 'qqq' a < b ma'lumot "
        "‘qoshtirnoq’ soz123soz o'zbek</p><SCRIPT>skr skr</SCRIPT>"
        "<style>.a{}</style></body></html>",
        "<p>matn <script type='x'>if (a > b) { yozuv > boshqa }</script> davom "
        "<scripts>teg nomi uzun</scripts> <noscript>yashirin > soz</noscript> oxiri</p>",
        "<!DOCTYPE html><!-x>bir <!--- -- -> > ichki --> ikki <!-> uch &#39;apos &zz; "
        "&uzunentitynomi; <élève> café — dash nbsp",
        "<style>p > a { color: red }</style>matn<style>x</styl></style>keyin "
        "<script>a</scriptx>b</script >tugadi <a href='x>y'>havola</a>",
    ]


@muallif(OTABEK, 4)
def run_self_tests4(args):
    """4-topshiriq testlari (--test4): GPU natijasi (har xil segment o'lchami,
    blok o'lchami, stream soni, geterogen nisbat) CPU natijasi bilan aynan bir xil."""
    if np is None:
        sys.exit("numpy kerak")
    if cuda is None or not cuda.is_available():
        LOG(COLAB_HELP)
        return 1
    LOG("4-TOPSHIRIQ TESTLARI (GPU: " + gpu_env().get("GPU", "?") + ")")
    ctx = {"checks": []}
    htmls = [generate_page(i, 6 if i % 3 else 14) for i in range(9)] + tricky_docs()
    buf, offs = pack_corpus(htmls)
    ref = count_words_seq(buf, offs)
    for seg in (16, 64, 2048):
        for tpb in (32, 256):
            check(ctx, f"GPU seg={seg} tpb={tpb} == CPU",
                  same_counts(gpu_count_words(buf, offs, tpb, seg=seg), ref)[0])
    check(ctx, "GPU 3 stream == CPU", same_counts(gpu_count_words(buf, offs, 64, 3, seg=64), ref)[0])
    check(ctx, "GPU kichik jadval (qayta urinish) == CPU",
          same_counts(gpu_count_words(buf, offs, 64, cap=64), ref)[0])
    for f in (0.0, 0.5, 1.0):
        check(ctx, f"geterogen f={f} == CPU",
              same_counts(hetero_static(buf, buf, offs, f, 2, {"tpb": 64, "seg": 64}), ref)[0])
    check(ctx, "dinamik navbat == CPU",
          same_counts(hetero_dynamic(buf, buf, offs, 2, {"tpb": 64, "seg": 64}, 5), ref)[0])
    failed = [c for c in ctx["checks"] if not c["ok"]]
    LOG(f"\n{len(ctx['checks']) - len(failed)}/{len(ctx['checks'])} test o'tdi")
    return 1 if failed else 0


# #############################################################################
#   5-TOPSHIRIQ: SAMARADORLIKNI BAHOLASH VA QIYOSIY TAHLIL (A9) + MVP
# -----------------------------------------------------------------------------
#   Semestr davomidagi barcha versiyalar bitta metodika bilan o'lchanadi:
#     1-yig'ma jadval (5 takror: o'rtacha, min, std) ... t5_exp_summary()
#     2-ketma-ket ulush f (3 usul) ..................... estimate_f()
#     3-Amdal va Gustafson ............................. t5_exp_scaling()
#     4-yuklama hajmiga bog'liqlik, ustunlik nuqtalari . t5_exp_sizes()
#     5-amaliy ta'sir (javob vaqti, resurs) ............ t5_exp_practical()
#     6-tavsiya jadvali ................................ t5_recommendations()
#     7-muhit .......................................... collect_env() + gpu_env()
#   Desktop ilova (MVP) - skreyper_app.py, shu fayldagi funksiyalarni chaqiradi.
# #############################################################################


@muallif(OTABEK, 5)
def measure5(fn, repeats=5, warmup=2):
    """5-topshiriq metodikasi: warmup yurish(lar) tashlanadi, keyin kamida 5
    marta o'lchanadi; o'rtacha, MINIMAL va standart og'ish qaytariladi."""
    m = measure_stable(fn, repeats, warmup)
    m["min"] = min(m["times"])
    return m


@muallif(OTABEK, 5)
def karp_flatt(S, n):
    """Karp-Flatt metrikasi: tajribadan aniqlangan ketma-ket ulush
    e = (1/S - 1/n) / (1 - 1/n)."""
    return (1.0 / S - 1.0 / n) / (1.0 - 1.0 / n) if n > 1 else None


@muallif(OTABEK, 5)
def fit_amdahl(ns, ss):
    """Amdal modelini S(n) = 1 / (f + (1 - f) / n) o'lchangan tezlashishlarga
    eng kichik kvadratlar usulida moslab, f ni topadi (0...1 oralig'ida qidiruv)."""
    best_f, best_e = 0.0, float("inf")
    for i in range(10001):
        f = i / 10000.0
        e = sum((s - 1.0 / (f + (1.0 - f) / n)) ** 2 for n, s in zip(ns, ss))
        if e < best_e:
            best_f, best_e = f, e
    return best_f


@muallif(OTABEK, 5)
def amdahl(f, n):
    """Amdal qonuni: masala hajmi o'zgarmas."""
    return 1.0 / (f + (1.0 - f) / n)


@muallif(OTABEK, 5)
def gustafson(f, n):
    """Gustafson qonuni: masala hajmi ishchilar soniga mutanosib o'sadi."""
    return n - f * (n - 1)


class ResourceMonitor:
    """Resurs sarfini kuzatish (amaliy ta'sir, 5-band): har 50 ms da tizim
    bo'yicha CPU yuklamasi va jarayon xotirasi, mavjud bo'lsa nvidia-smi orqali
    GPU quvvati (W). CPU yadro-soniyalari = o'rtacha CPU% x yadrolar x vaqt."""

    @muallif(ELBEK, 5)
    def __init__(self, gpu=False, interval=0.05):
        self.gpu, self.interval = gpu, interval
        self.cpu, self.rss, self.power = [], [], []
        self._stop = threading.Event()
        try:
            import psutil
            self.ps = psutil
            self.proc = psutil.Process()
        except ImportError:
            self.ps = None

    def _cpu_loop(self):
        while not self._stop.wait(self.interval):
            self.cpu.append(self.ps.cpu_percent(None))
            self.rss.append(self.proc.memory_info().rss)

    def _gpu_loop(self):
        exe = shutil.which("nvidia-smi")
        while exe and not self._stop.wait(0.2):
            try:
                out = subprocess.run([exe, "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=2).stdout
                self.power.append(float(out.strip().splitlines()[0]))
            except Exception:
                return

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.threads = []
        if self.ps:
            self.ps.cpu_percent(None)
            self.threads.append(threading.Thread(target=self._cpu_loop, daemon=True))
        if self.gpu:
            self.threads.append(threading.Thread(target=self._gpu_loop, daemon=True))
        for t in self.threads:
            t.start()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.t0
        self._stop.set()
        for t in self.threads:
            t.join(3)

    @muallif(ELBEK, 5)
    def summary(self):
        """O'rtacha CPU%, CPU yadro-soniyalari, eng katta xotira, GPU quvvati va energiyasi."""
        out = {"wall": self.wall}
        if self.cpu:
            avg = statistics.mean(self.cpu)
            out.update({"cpu_pct": avg, "core_s": avg / 100.0 * CPU_COUNT * self.wall,
                        "rss_mb": max(self.rss) / 2 ** 20})
        if self.power:
            out.update({"gpu_w": statistics.mean(self.power),
                        "gpu_j": statistics.mean(self.power) * self.wall})
        return out


@muallif(ELBEK, 5)
def slice_corpus(buf, offs, n):
    """Korpusning birinchi n sahifasi (bir xil sahifalar - kichik/o'rta/katta yuklama)."""
    n = min(n, len(offs) - 1)
    return np.ascontiguousarray(buf[:int(offs[n])]), offs[:n + 1].copy()


@muallif(ELBEK, 5)
def corpus_htmls(buf, offs):
    """Python versiyalari uchun sahifalarni matn ro'yxatiga qaytarish."""
    return [bytes(buf[offs[i]:offs[i + 1]]).decode("utf-8") for i in range(len(offs) - 1)]


@muallif(BOTH, 5)
def gpu_ok():
    """GPU (CUDA) haqiqatan ishlaydimi?"""
    try:
        return cuda is not None and cuda.is_available()
    except Exception:
        return False


@muallif(BOTH, 5)
def mpi_ok(args):
    """mpiexec va mpi4py mavjudmi?"""
    try:
        import mpi4py  # noqa: F401
        return find_mpiexec(getattr(args, "mpiexec", None)) is not None
    except ImportError:
        return False


@muallif(ELBEK, 5)
def t5_exp_fetch(ctx):
    """Yig'ma jadval, I/O qismi: sahifalarni yuklash - ketma-ket va oqimlar."""
    LOG("\n[1a] YUKLASH (I/O): ketma-ket va oqimlar ...")
    n = ctx["n_mid"]
    rows = []
    with LocalTestServer(n, 100, ctx["args"].t5_fetch_delay) as srv:
        urls = srv.urls(n)
        seq = measure5(lambda: fetch_sequential(urls), ctx["repeats"], 1)
        ref = pages_digest(seq["result"])
        rows.append({"tech": "Yuklash: ketma-ket", "n": 1, "mean": seq["mean"], "min": seq["min"],
                     "std": seq["std"], "S": 1.0, "E": 1.0})
        for w in (2, 4, 8, 16):
            m = measure5(lambda: fetch_parallel(urls, w, "thread"), ctx["repeats"], 1)
            check(ctx, f"yuklash oqimlar={w} == ketma-ket", pages_digest(m["result"]) == ref)
            s = seq["mean"] / m["mean"]
            rows.append({"tech": "Yuklash: ThreadPool", "n": w, "mean": m["mean"], "min": m["min"],
                         "std": m["std"], "S": s, "E": s / w})
    for r in rows:
        LOG(f"    {r['tech']:22s} n={r['n']:2d}: {r['mean']:.3f} s (min {r['min']:.3f}) S={r['S']:.2f}")
    ctx["res"]["fetch"] = rows


@muallif(OTABEK, 5)
def t5_exp_summary(ctx):
    """1-band: tahlil (CPU) qismining barcha versiyalari bitta korpusda (o'rta
    yuklama). S - o'z ketma-ket versiyasiga nisbatan, S_py - boshlang'ich Python
    ketma-ket versiyasiga nisbatan."""
    LOG("\n[1b] YIG'MA JADVAL: tahlilning barcha versiyalari ...")
    buf, offs = ctx["mid"]
    htmls = corpus_htmls(buf, offs)
    R = ctx["repeats"]
    rows = []

    def add(tech, model, n, m, t1, ok):
        s = t1 / m["mean"]
        rows.append({"tech": tech, "model": model, "n": n, "mean": m["mean"], "min": m["min"],
                     "std": m["std"], "S": s, "E": s / n if isinstance(n, int) else None,
                     "S_py": ctx["t_py"] / m["mean"] if ctx.get("t_py") else 1.0})
        check(ctx, f"{tech} n={n} natijasi mos", ok)
        LOG(f"    {tech:28s} n={str(n):>4s}: {1000 * m['mean']:9.2f} ms (min {1000 * m['min']:.2f},"
            f" std {1000 * m['std']:.2f})  S={s:6.2f}  S_py={rows[-1]['S_py']:7.1f}")

    py = measure5(lambda: analyze_sequential(htmls), R, 1)
    ctx["t_py"] = py["mean"]
    ref_py = dict(py["result"])
    add("Python ketma-ket (bs4)", "ketma-ket", 1, py, py["mean"], True)
    for w in ctx["workers"]:
        m = measure5(lambda: analyze_parallel(htmls, w, "thread"), R, 1)
        add("Python ThreadPool", "oqimlar (GIL)", w, m, py["mean"], dict(m["result"]) == ref_py)
    for w in ctx["workers"]:
        m = measure5(lambda: analyze_parallel(htmls, w, "process"), R, 1)
        add("Python ProcessPool", "jarayonlar", w, m, py["mean"], dict(m["result"]) == ref_py)
    nb = measure5(lambda: count_words_seq(buf, offs), R)
    ctx["ref"] = nb["result"]
    add("Numba ketma-ket", "kompilyatsiya", 1, nb, nb["mean"],
        decode_words(nb["result"], buf) == ref_py)
    ctx["t_nb"] = nb["mean"]
    for w in ctx["threads"]:
        m = measure5(lambda: count_words_parallel(buf, offs, w, 0, cyclic=True), R)
        add("Numba prange (OpenMP)", "umumiy xotira", w, m, nb["mean"],
            same_counts(m["result"], ctx["ref"])[0])
    if ctx["mpi"]:
        extra = ctx["args"].mpi_args.split() if ctx["args"].mpi_args else []
        for p in [1] + list(ctx["workers"]):
            r = run_mpi(p, ctx["mid_path"], "strong", 1, R, ctx["mpiexec"], extra, ctx["out"])
            js = [t["jami"] for t in r["runs"]]
            m = {"mean": statistics.mean(js), "min": min(js),
                 "std": statistics.stdev(js) if len(js) > 1 else 0.0}
            add("MPI (mpi4py)", "taqsimlangan xotira", p, m, nb["mean"],
                same_counts(r["_result"], ctx["ref"])[0])
    if ctx["gpu"]:
        m = measure5(lambda: gpu_count_words(ctx["mid_pinned"], offs, 256, 4), R)
        add("GPU CUDA (ko'chirish bilan)", "GPU", "1 GPU", m, nb["mean"],
            same_counts(m["result"], ctx["ref"])[0])
        m = measure5(lambda: hetero_static(buf, ctx["mid_pinned"], offs, 0.9, 8,
                                           {"tpb": 256, "streams": 4}), R)
        add("Geterogen CPU+GPU (90%)", "geterogen", "8+GPU", m, nb["mean"],
            same_counts(m["result"], ctx["ref"])[0])
    ctx["res"]["summary"] = rows


@muallif(OTABEK, 5)
def estimate_f(ctx):
    """2-band: ketma-ket ulush f uch usulda: (1) Amdal modelini tezlashish
    egri chizig'iga moslash, (2) Karp-Flatt metrikasi, (3) ketma-ket qismlar
    vaqtini bevosita o'lchash (jadval ajratish + birlashtirish)."""
    LOG("\n[2] KETMA-KET ULUSH f ...")
    out = {}
    rows = ctx["res"]["summary"]
    for tech in ("Numba prange (OpenMP)", "Python ProcessPool", "MPI (mpi4py)"):
        rr = [r for r in rows if r["tech"] == tech and isinstance(r["n"], int)]
        if len(rr) < 2:
            continue
        base = next((r for r in rr if r["n"] == 1), None)
        t1 = base["mean"] if base else (ctx["t_nb"] if "Numba" in tech or "MPI" in tech else ctx["t_py"])
        ns = [r["n"] for r in rr]
        ss = [t1 / r["mean"] for r in rr]
        f = fit_amdahl(ns, ss)
        kf = {n: karp_flatt(s, n) for n, s in zip(ns, ss) if n > 1}
        out[tech] = {"ns": ns, "S": ss, "f_amdahl": f, "karp_flatt": kf}
        LOG(f"    {tech:24s}: f (Amdal) = {f:.3f}; Karp-Flatt: " +
            ", ".join(f"n={n}: {e:.3f}" for n, e in kf.items()))
    buf, offs = ctx["mid"]
    r1 = count_words_parallel(buf, offs, 1, 0, cyclic=True)
    t = r1["times"]
    f_direct = (t["alloc"] + t["merge"]) / (t["alloc"] + t["scan"] + t["merge"])
    out["direct"] = {"alloc": t["alloc"], "scan": t["scan"], "merge": t["merge"], "f": f_direct}
    LOG(f"    bevosita (Numba, 1 oqim): ajratish {1000 * t['alloc']:.2f} ms + birlashtirish "
        f"{1000 * t['merge']:.2f} ms / jami -> f = {f_direct:.4f}")
    ctx["res"]["f"] = out


@muallif(OTABEK, 5)
def t5_exp_scaling(ctx):
    """3-band: Amdal (kuchli) va Gustafson (zaif) - nazariya va o'lchov. Kuchli:
    katta korpus, n oqim; zaif: har bir oqimga bir xil hajm (250 sahifa)."""
    LOG("\n[3] AMDAL va GUSTAFSON ...")
    buf, offs = ctx["big"]
    per = max(1, (len(offs) - 1) // max(ctx["threads"]))   # zaif: har oqimga bir xil hajm
    strong, weak = [], []
    t1 = None
    for n in ctx["threads"]:
        m = measure5(lambda: count_words_parallel(buf, offs, n, 0, cyclic=True), ctx["repeats"])
        t1 = t1 or m["mean"]
        strong.append({"n": n, "mean": m["mean"], "S": t1 / m["mean"]})
        b, o = slice_corpus(buf, offs, per * n)
        w = measure5(lambda: count_words_parallel(b, o, n, 0, cyclic=True), ctx["repeats"])
        weak.append({"n": n, "pages": len(o) - 1, "mean": w["mean"]})
    for r in weak:
        r["S_scaled"] = r["n"] * weak[0]["mean"] / r["mean"]
    f = fit_amdahl([r["n"] for r in strong], [r["S"] for r in strong])
    for r in strong:
        LOG(f"    n={r['n']:2d}: kuchli S={r['S']:.2f} (Amdal {amdahl(f, r['n']):.2f}) | zaif "
            f"S={next(w for w in weak if w['n'] == r['n'])['S_scaled']:.2f} "
            f"(Gustafson {gustafson(f, r['n']):.2f})")
    ctx["res"]["scaling"] = {"strong": strong, "weak": weak, "f": f, "per": per}


@muallif(ELBEK, 5)
def t5_exp_sizes(ctx):
    """4-band: kichik/o'rta/katta yuklamada eng yaxshi texnologiyalar va
    ustunlik almashadigan nuqtalar."""
    LOG("\n[4] YUKLAMA HAJMIGA BOG'LIQLIK ...")
    buf, offs = ctx["big"]
    techs = ["Numba prange 8", "Python ProcessPool 8"]
    if ctx["gpu"]:
        techs.insert(0, "GPU CUDA")
    if ctx["mpi"]:
        techs.append("MPI 8")
    extra = ctx["args"].mpi_args.split() if ctx["args"].mpi_args else []
    rows = []
    for n in ctx["sizes"]:
        b, o = slice_corpus(buf, offs, n)
        mb = float(o[-1]) / 2 ** 20
        ref = count_words_seq(b, o)
        row = {"pages": len(o) - 1, "mb": mb}
        for t in techs:
            if t == "Numba prange 8":
                m = measure5(lambda: count_words_parallel(b, o, 8, 0, cyclic=True), ctx["repeats"])
                ok = same_counts(m["result"], ref)[0]
            elif t == "GPU CUDA":
                pin = cuda.pinned_array(b.shape, np.uint8)
                pin[:] = b
                m = measure5(lambda: gpu_count_words(pin, o, 256, 4 if n >= 200 else 1),
                             ctx["repeats"])
                ok = same_counts(m["result"], ref)[0]
            elif t == "Python ProcessPool 8":
                if n > ctx["args"].t5_py_max:
                    continue
                hs = corpus_htmls(b, o)
                m = measure5(lambda: analyze_parallel(hs, 8, "process"), ctx["repeats"], 1)
                ok = dict(m["result"]) == decode_words(ref, b)
            else:
                path = os.path.join(ctx["out"], f"_korpus_{n}.npz")
                np.savez(path, buf=b, offs=o)
                r = run_mpi(8, path, "strong", 1, ctx["repeats"], ctx["mpiexec"], extra, ctx["out"])
                js = [x["jami"] for x in r["runs"]]
                m = {"mean": statistics.mean(js), "min": min(js)}
                ok = same_counts(r["_result"], ref)[0]
                os.remove(path)
            row[t] = m["mean"]
            check(ctx, f"{t}, {n} sahifa == ketma-ket", ok)
        rows.append(row)
        LOG(f"    {n:5d} sahifa ({mb:6.1f} MB): " + ", ".join(
            f"{t} {1000 * row[t]:.1f} ms" for t in techs if t in row))
    cross = []
    for i in range(1, len(rows)):
        a, b_ = rows[i - 1], rows[i]
        best_a = min((t for t in techs if t in a), key=lambda t: a[t])
        best_b = min((t for t in techs if t in b_), key=lambda t: b_[t])
        if best_a != best_b:
            cross.append({"from": a["pages"], "to": b_["pages"], "before": best_a, "after": best_b})
    ctx["res"]["sizes"] = {"rows": rows, "techs": techs, "cross": cross}
    for c in cross:
        LOG(f"    ustunlik almashdi: {c['from']}..{c['to']} sahifa oralig'ida "
            f"{c['before']} -> {c['after']}")


@muallif(ELBEK, 5)
def t5_exp_practical(ctx):
    """5-band: foydalanuvchi sezadigan ta'sir - N sahifani yuklab tahlil qilish:
    semestr boshidagi ketma-ket versiya, 2-topshiriq konveyeri va MVP.
    Javob vaqti, o'tkazuvchanlik, CPU yadro-soniyalari, xotira, GPU energiyasi."""
    LOG("\n[5] AMALIY TA'SIR (foydalanuvchi nuqtai nazaridan) ...")
    n = ctx["n_practical"]
    rows = []
    with LocalTestServer(n, 100, ctx["args"].t5_delay) as srv:
        urls = srv.urls(n)

        def s_seq():
            return analyze_sequential([h for _, h, _ in fetch_sequential(urls) if h])

        def s_pipe():
            return run_pipeline(urls, 8, min(8, CPU_COUNT))

        def s_mvp_cpu():
            b, o = pack_corpus([h for _, h, _ in fetch_parallel(urls, 16, "thread") if h])
            return decode_words(count_words_parallel(b, o, 8, 0, cyclic=True), b)

        def s_mvp_gpu():
            b, o = pack_corpus([h for _, h, _ in fetch_parallel(urls, 16, "thread") if h])
            return decode_words(gpu_count_words(b, o, 256, 4), b)
        sc = [("Semestr boshi: ketma-ket yuklash + Python tahlil", s_seq, False),
              ("2-topshiriq: konveyer (oqimlar + jarayonlar)", s_pipe, False),
              ("MVP: oqimlar + Numba prange", s_mvp_cpu, False)]
        if ctx["gpu"]:
            sc.append(("MVP: oqimlar + GPU", s_mvp_gpu, True))
        ref = None
        for name, fn, g in sc:
            fn()                                            # isitish (JIT, kesh)
            samples, mons = [], []
            for _ in range(ctx["repeats"]):
                with ResourceMonitor(gpu=g) as mon:
                    r = fn()
                samples.append(mon.wall)
                mons.append(mon.summary())
            r = dict(r)
            ref = ref or r
            check(ctx, f"amaliy: {name}", r == ref)
            row = {"name": name, "mean": statistics.mean(samples), "min": min(samples),
                   "std": statistics.stdev(samples), "pages_s": n / statistics.mean(samples)}
            for k in ("cpu_pct", "core_s", "rss_mb", "gpu_w", "gpu_j"):
                vals = [mm[k] for mm in mons if k in mm]
                if vals:
                    row[k] = statistics.mean(vals)
            rows.append(row)
            LOG(f"    {name:48s}: {row['mean']:.2f} s, {row['pages_s']:.1f} sahifa/s"
                + (f", CPU {row['core_s']:.1f} yadro-s" if "core_s" in row else ""))
    ctx["res"]["practical"] = {"rows": rows, "pages": n, "delay": ctx["args"].t5_delay}


@muallif(ELBEK, 5)
def t5_recommendations(res):
    """6-band: tavsiya jadvali - har bir qator o'lchangan raqam bilan."""
    rec = []
    fe = res["fetch"]
    best_f = max(fe, key=lambda r: r["S"])
    rec.append(["Sahifalarni yuklash (I/O)", f"ThreadPoolExecutor, {best_f['n']} oqim",
                f"S = {best_f['S']:.1f} (ketma-ket {fe[0]['mean']:.2f} s -> {best_f['mean']:.2f} s)"])
    sm = res["summary"]
    th = max((r for r in sm if r["tech"] == "Python ThreadPool"), key=lambda r: r["S"])
    pp = max((r for r in sm if r["tech"] == "Python ProcessPool"), key=lambda r: r["S"])
    rec.append(["Tahlil, faqat toza Python (Numba yo'q)", f"ProcessPoolExecutor, {pp['n']} jarayon",
                f"S = {pp['S']:.2f}; ThreadPool esa S = {th['S']:.2f} (GIL)"])
    sz = res["sizes"]
    for row, label in ((sz["rows"][0], "kichik"), (sz["rows"][-1], "katta")):
        cand = [t for t in sz["techs"] if t in row]
        best = min(cand, key=lambda t: row[t])
        others = ", ".join(f"{t}: {1000 * row[t]:.1f} ms" for t in cand if t != best)
        rec.append([f"Tahlil, {label} yuklama ({row['pages']} sahifa, {row['mb']:.1f} MB)", best,
                    f"{1000 * row[best]:.1f} ms" + (f" (boshqalar: {others})" if others else "")])
    np8 = next((r for r in sm if r["tech"] == "Numba prange (OpenMP)" and r["n"] == 8), None)
    if np8:
        rec.append(["Tahlil, GPU siz kompyuter", "Numba prange, 8 oqim, siklik taqsimot",
                    f"S = {np8['S']:.2f} Numba ketma-ketga, S_py = {np8['S_py']:.0f} Python ga"])
    mpi = [r for r in sm if r["tech"] == "MPI (mpi4py)"]
    if mpi:
        m8 = max(mpi, key=lambda r: r["n"] if isinstance(r["n"], int) else 0)
        rec.append(["Bir nechta kompyuter (klaster)", f"MPI, {m8['n']} jarayon",
                    f"S = {m8['S']:.2f}, E = {m8['E']:.2f}; bitta kompyuterda Numba dan sekin"])
    het = next((r for r in sm if r["tech"].startswith("Geterogen")), None)
    gpu = next((r for r in sm if r["tech"].startswith("GPU")), None)
    if het and gpu:
        gain = gpu["mean"] / het["mean"]
        choice = ("Geterogen statik nisbat (90% GPU)" if gain > 1.1 else
                  "Faqat GPU (CPU ni yuklash va interfeysga qoldirish)")
        rec.append(["CPU va GPU ni birga ishlatish", choice,
                    f"geterogen {1000 * het['mean']:.1f} ms, faqat GPU {1000 * gpu['mean']:.1f} ms "
                    f"(yutuq {gain:.2f}x)"])
    return rec


@muallif(ELBEK, 5)
def make_charts5(res, outdir):
    """5-topshiriq grafiklari."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, name), dpi=130)
        plt.close(fig)
        files.append(name)

    sm = res["summary"]
    best = {}
    for r in sm:
        if r["tech"] not in best or r["mean"] < best[r["tech"]]["mean"]:
            best[r["tech"]] = r
    rr = sorted(best.values(), key=lambda r: -r["mean"])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.barh([f"{r['tech']} (n={r['n']})" for r in rr], [1000 * r["mean"] for r in rr],
                   xerr=[1000 * r["std"] for r in rr], capsize=3, color="C0")
    for bar, r in zip(bars, rr):
        ax.annotate(f"{r['S_py']:.0f}x", (bar.get_width() + 1000 * r["std"],
                                          bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(right=ax.get_xlim()[1] * 3)
    ax.set_xlabel("Tahlil vaqti, ms (logarifmik shkala)")
    ax.set_title("Barcha versiyalar (eng yaxshi n); yozuv - Python ga nisbatan S", fontsize=10)
    ax.grid(axis="x", alpha=.3, which="both")
    ax.tick_params(axis="y", labelsize=8)
    save(fig, "1_yigma.png")

    sc = res["scaling"]
    f = sc["f"]
    ns = list(range(1, max(r["n"] for r in sc["strong"]) + 1))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(ns, ns, "k:", alpha=.5, label="ideal S = n")
    ax.plot(ns, [amdahl(f, n) for n in ns], "-", color="C0", label=f"Amdal (f = {f:.3f})")
    ax.plot(ns, [gustafson(f, n) for n in ns], "-", color="C1", label=f"Gustafson (f = {f:.3f})")
    ax.plot([r["n"] for r in sc["strong"]], [r["S"] for r in sc["strong"]], "o--", color="C0",
            label="o'lchov: kuchli (N o'zgarmas)")
    ax.plot([r["n"] for r in sc["weak"]], [r["S_scaled"] for r in sc["weak"]], "s--", color="C1",
            label="o'lchov: zaif, masshtablangan S")
    ax.set_xlabel("Oqimlar soni, n")
    ax.set_ylabel("Tezlashish, S")
    ax.set_title("Amdal va Gustafson qonunlari: nazariya va o'lchov (Numba prange)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    save(fig, "2_amdal_gustafson.png")

    sz = res["sizes"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for t in sz["techs"]:
        pts = [(r["mb"], 1000 * r[t]) for r in sz["rows"] if t in r]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=t)
    for c in sz["cross"]:
        x = next(r["mb"] for r in sz["rows"] if r["pages"] == c["to"])
        ax.axvline(x, color="gray", ls=":", alpha=.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Korpus hajmi, MB (log)")
    ax.set_ylabel("Tahlil vaqti, ms (log)")
    ax.set_title("Yuklama hajmiga bog'liqlik (nuqtali chiziq - ustunlik almashishi)")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    save(fig, "3_hajm.png")

    pr = res["practical"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    import textwrap
    labels = [textwrap.fill(r["name"], 20) for r in pr]
    axes[0].bar(range(len(pr)), [r["mean"] for r in pr], yerr=[r["std"] for r in pr], capsize=3)
    axes[0].set_ylabel("Javob vaqti, s")
    axes[0].set_title(f"{res['practical']['pages']} sahifani yuklash va tahlil qilish")
    if all("core_s" in r for r in pr):
        axes[1].bar(range(len(pr)), [r["core_s"] for r in pr], color="C2")
        axes[1].set_ylabel("CPU sarfi, yadro-soniya")
        axes[1].set_title("Resurs sarfi (tizim bo'yicha CPU)")
    for ax in axes:
        ax.set_xticks(range(len(pr)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(axis="y", alpha=.3)
    save(fig, "4_amaliy.png")
    return files


@muallif(ELBEK, 5)
def write_report5(ctx, outdir):
    """natijalar5.md, natijalar5.json, barcha_natijalar5.csv va grafiklar."""
    res = ctx["res"]
    res["recommendations"] = t5_recommendations(res)
    charts = make_charts5(res, outdir)
    with open(os.path.join(outdir, "natijalar5.json"), "w", encoding="utf-8") as f:
        json.dump({"env": ctx["env"], "results": res, "checks": ctx["checks"],
                   "authors": AUTHORS5}, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(outdir, "barcha_natijalar5.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["texnologiya", "n", "ortacha_ms", "min_ms", "std_ms", "S", "E", "S_python"])
        for r in res["fetch"] + res["summary"]:
            w.writerow([r["tech"], r["n"], f"{1000 * r['mean']:.3f}", f"{1000 * r['min']:.3f}",
                        f"{1000 * r['std']:.3f}", f"{r['S']:.3f}",
                        "" if r.get("E") is None else f"{r['E']:.3f}",
                        f"{r.get('S_py', 0):.2f}"])

    def fmt_e(r):
        return "-" if r.get("E") is None else f"{r['E']:.2f}"
    md = ["# Parallel veb-skreyper - 5-topshiriq: yakuniy qiyosiy tahlil", "",
          f"**Sana:** {time.strftime('%Y-%m-%d %H:%M')}", "", "## Sinov muhiti", "",
          fmt_table(["Parametr", "Qiymat"], [[k, v] for k, v in ctx["env"].items()]),
          "", f"O'rta yuklama: {len(ctx['mid'][1]) - 1} sahifa, "
          f"{ctx['mid'][1][-1] / 2 ** 20:.1f} MB. Har bir o'lchov {ctx['repeats']} marta; "
          "birinchi yurish(lar) hisobga olinmaydi.",
          "", "## 1. Yig'ma jadval: yuklash (I/O)", "",
          fmt_table(["Versiya", "n", "O'rtacha, ms", "Min, ms", "Std, ms", "S", "E"],
                    [[r["tech"], r["n"], f"{1000 * r['mean']:.1f}", f"{1000 * r['min']:.1f}",
                      f"{1000 * r['std']:.1f}", f"{r['S']:.2f}", f"{r['E']:.2f}"]
                     for r in res["fetch"]]),
          "", "## 1. Yig'ma jadval: tahlil (CPU)", "",
          fmt_table(["Versiya", "Model", "n", "O'rtacha, ms", "Min, ms", "Std, ms", "S", "E", "S_py"],
                    [[r["tech"], r["model"], r["n"], f"{1000 * r['mean']:.2f}",
                      f"{1000 * r['min']:.2f}", f"{1000 * r['std']:.2f}", f"{r['S']:.2f}",
                      fmt_e(r), f"{r['S_py']:.1f}"] for r in res["summary"]]),
          "", "## 2. Ketma-ket ulush f", ""]
    for k, v in res["f"].items():
        if k == "direct":
            md.append(f"- Bevosita (Numba, 1 oqim): ajratish {1000 * v['alloc']:.2f} ms, "
                      f"birlashtirish {1000 * v['merge']:.2f} ms, scan {1000 * v['scan']:.2f} ms "
                      f"-> f = {v['f']:.4f}")
        else:
            md.append(f"- {k}: f (Amdal moslash) = {v['f_amdahl']:.3f}; Karp-Flatt: " +
                      ", ".join(f"n={n}: {e:.3f}" for n, e in v["karp_flatt"].items()))
    sc = res["scaling"]
    md += ["", "## 3. Amdal va Gustafson", "",
           fmt_table(["n", "Kuchli S", "Amdal", "Zaif S (masshtab.)", "Gustafson"],
                     [[r["n"], f"{r['S']:.2f}", f"{amdahl(sc['f'], r['n']):.2f}",
                       f"{w['S_scaled']:.2f}", f"{gustafson(sc['f'], r['n']):.2f}"]
                      for r, w in zip(sc["strong"], sc["weak"])]),
           "", "## 4. Yuklama hajmi", "",
           fmt_table(["Sahifalar", "MB"] + [f"{t}, ms" for t in res["sizes"]["techs"]],
                     [[r["pages"], f"{r['mb']:.1f}"] + [f"{1000 * r[t]:.2f}" if t in r else "-"
                                                        for t in res["sizes"]["techs"]]
                      for r in res["sizes"]["rows"]])]
    md += [f"- Ustunlik almashishi: {c['from']}..{c['to']} sahifa: {c['before']} -> {c['after']}"
           for c in res["sizes"]["cross"]] or ["- Ustunlik almashmadi."]
    pr = res["practical"]
    md += ["", f"## 5. Amaliy ta'sir ({pr['pages']} sahifa, server kechikishi {pr['delay']} s)", "",
           fmt_table(["Versiya", "Javob vaqti, s", "Min, s", "Sahifa/s", "CPU, %", "CPU yadro-s",
                      "Xotira, MB", "GPU, W"],
                     [[r["name"], f"{r['mean']:.2f}", f"{r['min']:.2f}", f"{r['pages_s']:.1f}",
                       f"{r['cpu_pct']:.0f}" if "cpu_pct" in r else "-",
                       f"{r['core_s']:.1f}" if "core_s" in r else "-",
                       f"{r['rss_mb']:.0f}" if "rss_mb" in r else "-",
                       f"{r['gpu_w']:.1f}" if "gpu_w" in r else "-"] for r in pr["rows"]]),
           "", "## 6. Tavsiya jadvali", "",
           fmt_table(["Loyiha qismi / sharoit", "Texnologiya", "O'lchangan asos"],
                     res["recommendations"]),
           "", "## To'g'rilik", "",
           fmt_table(["Tekshiruv", "Natija"],
                     [[c["name"], "MOS" if c["ok"] else "**XATO**"] for c in ctx["checks"]]),
           "", "## Grafiklar", ""] + [f"![{c}]({c})" for c in charts]
    with open(os.path.join(outdir, "natijalar5.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return charts


@muallif(BOTH, 5)
def run_task5(args):
    """5-topshiriqning barcha o'lchovlari."""
    if np is None or not HAVE_NUMBA:
        sys.exit("numpy va numba kerak: pip install numpy numba")
    os.makedirs(args.out5, exist_ok=True)
    env = collect_env()
    env.update({"numpy": np.__version__, "numba": numba.__version__})
    g = gpu_ok()
    env.update(gpu_env() if g else {"GPU": "CUDA topilmadi"})
    m = mpi_ok(args)
    if m:
        import mpi4py
        env["mpi4py"] = mpi4py.__version__
    LOG("5-TOPSHIRIQ. Sinov muhiti: " + ", ".join(f"{k}: {v}" for k, v in env.items()))
    big = load_corpus4(args)
    mid = slice_corpus(*big, args.t5_mid)
    ctx = {"args": args, "res": {}, "checks": [], "env": env, "repeats": args.t5_repeats,
           "workers": [2, 4, 8], "threads": [t for t in (1, 2, 4, 8, 16)
                                             if t <= numba.config.NUMBA_NUM_THREADS],
           "gpu": g, "mpi": m, "mpiexec": find_mpiexec(args.mpiexec) if m else None,
           "big": big, "mid": mid, "n_mid": args.t5_mid, "out": args.out5,
           "sizes": [s for s in args.t5_sizes if s <= len(big[1]) - 1],
           "n_practical": args.t5_practical}
    ctx["mid_path"] = os.path.join(args.out5, "_korpus_orta.npz")
    np.savez(ctx["mid_path"], buf=mid[0], offs=mid[1])
    if g:
        pin = cuda.pinned_array(mid[0].shape, np.uint8)
        pin[:] = mid[0]
        ctx["mid_pinned"] = pin
    for exp in (t5_exp_fetch, t5_exp_summary, estimate_f, t5_exp_scaling, t5_exp_sizes,
                t5_exp_practical):
        exp(ctx)
    try:
        os.remove(ctx["mid_path"])
    except OSError:
        pass
    charts = write_report5(ctx, args.out5)
    ok = sum(c["ok"] for c in ctx["checks"])
    LOG(f"\nTo'g'rilik: {ok}/{len(ctx['checks'])} tekshiruv MOS")
    LOG("\nTAVSIYALAR:")
    for r in ctx["res"]["recommendations"]:
        LOG(f"  - {r[0]}: {r[1]} ({r[2]})")
    LOG(f"\nNatijalar: {os.path.abspath(args.out5)} (natijalar5.md, natijalar5.json, "
        f"barcha_natijalar5.csv, {len(charts)} ta grafik)")
    return 0


# #############################################################################
#                       AVTOMATIK TESTLAR VA ASOSIY QISM
# #############################################################################
@muallif(OTABEK)
def run_self_tests():
    """8-band: tez avtomatik testlar (--test). Barcha parallel yo'llar ketma-ket
    natija bilan bir xil ekanini tekshiradi."""
    LOG("AVTOMATIK TESTLAR ...")
    ctx = {"checks": []}
    with LocalTestServer(12, size_kb=8, delay=0.01) as srv:
        urls = srv.urls(10)
        seq = fetch_sequential(urls)
        htmls = [h for _, h, _ in seq if h]
        ref = analyze_sequential(htmls)
        check(ctx, "10 ta sahifa yuklandi", len(htmls) == 10)
        check(ctx, "script/style matni hisobga olinmadi",
              "skriptsoz" not in ref and "stilsoz" not in ref)
        check(ctx, "robots.txt /private/ ni taqiqlaydi",
              not PoliteGate().allowed(f"http://127.0.0.1:{srv.port}/private/x"))
        for b in ("thread", "process"):
            check(ctx, f"fetch_parallel {b}",
                  pages_digest(fetch_parallel(urls, 4, b)) == pages_digest(seq))
            check(ctx, f"analyze_parallel {b}", analyze_parallel(htmls, 2, b) == ref)
            check(ctx, f"parallel_reduce {b}", parallel_reduce(htmls, 4, backend=b) == ref)
        check(ctx, "gibrid konveyer", run_pipeline(urls, 4, 2) == ref)
        check(ctx, "split_chunks barcha elementlarni qamraydi",
              sum(split_chunks(list(range(37)), 8), []) == list(range(37)))
        r = race_condition_demo(4, 500)
        check(ctx, "Lock bilan hisoblagich", r["lock_bilan"] == r["expected"])
    failed = [c for c in ctx["checks"] if not c["ok"]]
    LOG(f"\n{len(ctx['checks']) - len(failed)}/{len(ctx['checks'])} test o'tdi")
    return 1 if failed else 0


@muallif(BOTH)
def parse_args(argv=None):
    """Buyruq qatori parametrlari."""
    ap = argparse.ArgumentParser(description="Parallel veb-skreyper - 2-, 3- va 4-topshiriq")
    ap.add_argument("--delay", type=float, default=0.05, help="sun'iy tarmoq kechikishi, s")
    ap.add_argument("--page-kb", type=int, default=100, help="sinov sahifasi hajmi, KB")
    ap.add_argument("--loads", type=int, nargs="+", default=[20, 40, 60],
                    help="bazaviy o'lchov yuklamalari (sahifalar soni)")
    ap.add_argument("--pages", type=int, default=60, help="asosiy tajribalardagi sahifalar")
    ap.add_argument("--batch", type=int, default=20, help="parallel_invoke yuklash to'plami")
    ap.add_argument("--repeats", type=int, default=3, help="takrorlar soni (>=3)")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--chunks", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--urls", help="real URL'lar fayli (har qatorda bitta)")
    ap.add_argument("--out", default="natijalar", help="natijalar papkasi")
    ap.add_argument("--quick", action="store_true", help="tez rejim")
    ap.add_argument("--test", action="store_true", help="faqat avtomatik testlar")
    g = ap.add_argument_group("3-topshiriq (Numba va MPI)")
    g.add_argument("--task3", action="store_true", help="3-topshiriq tajribalari")
    g.add_argument("--test3", action="store_true", help="3-topshiriq testlari")
    g.add_argument("--out3", default="natijalar3", help="3-topshiriq natijalari papkasi")
    g.add_argument("--t3-pages", type=int, default=2000, help="korpusdagi sahifalar soni")
    g.add_argument("--t3-threads", type=int, nargs="+", default=[1, 2, 4, 8])
    g.add_argument("--t3-procs", type=int, nargs="+", default=[1, 2, 4, 8])
    g.add_argument("--mpiexec", help="mpiexec yo'li (topilmasa)")
    g.add_argument("--mpi-args", default="", help="mpiexec ga qo'shimcha kalitlar")
    g.add_argument("--mpi-worker", action="store_true", help="mpiexec ostida ishlash")
    g.add_argument("--corpus", help="MPI uchun korpus (.npz)")
    g.add_argument("--mode", choices=["strong", "weak"], default="strong")
    g.add_argument("--per-proc", type=int, default=250, help="zaif: jarayonga sahifa")
    g.add_argument("--mpi-out", help="MPI natijasi (JSON) fayli")
    g4 = ap.add_argument_group("4-topshiriq (CUDA va geterogen)")
    g4.add_argument("--task4", action="store_true", help="4-topshiriq tajribalari")
    g4.add_argument("--test4", action="store_true", help="4-topshiriq testlari")
    g4.add_argument("--out4", default="natijalar4", help="4-topshiriq natijalari papkasi")
    g4.add_argument("--t4-tpb", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    g4.add_argument("--t4-streams", type=int, nargs="+", default=[2, 4, 8])
    g4.add_argument("--t4-ratios", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    g4.add_argument("--t4-chunks", type=int, default=32, help="dinamik navbat bo'laklari")
    g5 = ap.add_argument_group("5-topshiriq (yakuniy qiyosiy tahlil)")
    g5.add_argument("--task5", action="store_true", help="5-topshiriq o'lchovlari")
    g5.add_argument("--out5", default="natijalar5", help="5-topshiriq natijalari papkasi")
    g5.add_argument("--t5-repeats", type=int, default=5, help="takrorlar soni (>= 5)")
    g5.add_argument("--t5-mid", type=int, default=200, help="o'rta yuklama, sahifa")
    g5.add_argument("--t5-sizes", type=int, nargs="+", default=[20, 50, 200, 500, 2000])
    g5.add_argument("--t5-py-max", type=int, default=500, help="Python versiyasi uchun max sahifa")
    g5.add_argument("--t5-practical", type=int, default=100, help="amaliy ssenariy, sahifa")
    g5.add_argument("--t5-delay", type=float, default=0.1, help="amaliy ssenariy kechikishi, s")
    g5.add_argument("--t5-fetch-delay", type=float, default=0.02, help="yuklash jadvali kechikishi")
    g5.add_argument("--quick5", action="store_true", help="tez rejim (kichik yuklama)")
    g5.add_argument("--gpu-diag", action="store_true", help="GPU diagnostikasi -> gpu_diag.txt")
    a = ap.parse_args(argv)
    if a.quick5:
        a.t5_mid, a.t5_sizes, a.t5_py_max, a.t5_practical = 60, [20, 60, 200], 60, 40
    if a.corpus is None:
        a.corpus = os.path.join(a.out3, "korpus_bir_xil.npz")
    if a.quick:
        a.loads, a.pages, a.batch, a.page_kb = [10, 20, 30], 30, 10, 60
    return a


@muallif(BOTH)
def run_all(args, urls_all=None, batch_urls=None, source="lokal sinov serveri"):
    """Barcha tajribalarni ketma-ket bajaradi va hisobotni yozadi.
    Desktop ilova (GUI) keyinchalik shu funksiyani chaqiradi."""
    ctx = {"args": args, "res": {}, "checks": [], "env": collect_env(),
           "loads": args.loads, "repeats": args.repeats, "workers": args.workers,
           "chunks": args.chunks, "urls": urls_all,
           "main_urls": urls_all[:args.pages], "batch_urls": batch_urls,
           "strict_fetch": args.urls is None,
           "params": {"source": source, "delay": args.delay if not args.urls else "real",
                      "page_kb": args.page_kb, "repeats": args.repeats,
                      "loads": args.loads, "pages": args.pages}}
    LOG("Sinov muhiti: " + ", ".join(f"{k}: {v}" for k, v in ctx["env"].items()))
    pages = fetch_parallel(ctx["main_urls"], 8, "thread")
    ctx["corpus"] = [h for _, h, _ in pages if h]       # tahlil tajribalari uchun korpus
    LOG(f"Korpus: {len(ctx['corpus'])} sahifa, "
        f"{sum(map(len, ctx['corpus'])) / 2 ** 20:.1f} MB")
    for exp in (exp_profiling, exp_baseline, exp_fetch_workers, exp_analyze_workers,
                exp_sync, exp_invoke, exp_reduce, exp_versions):
        exp(ctx)
    charts = write_report(ctx, args.out)
    ok = sum(c["ok"] for c in ctx["checks"])
    LOG(f"\nTo'g'rilik: {ok}/{len(ctx['checks'])} tekshiruv MOS")
    LOG("\nXULOSA:")
    for i, t in enumerate(conclusions(ctx["res"]), 1):
        LOG(f"  {i}. {t}")
    LOG(f"\nNatijalar: {os.path.abspath(args.out)}  (natijalar.md, natijalar.json, "
        f"barcha_natijalar.csv, {len(charts)} ta grafik)")
    return ctx


@muallif(BOTH)
def main(argv=None):
    """Kirish nuqtasi: lokal server yoki real URL'lar bilan tajribani boshlaydi."""
    if sys.stdout is None:                              # exe --windowed rejimi
        sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    args = parse_args(argv)
    if args.mpi_worker:                     # mpiexec -n p python ... --mpi-worker
        return mpi_worker(args)
    if args.gpu_diag:
        txt = gpu_diagnostics()
        with open("gpu_diag.txt", "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        LOG(txt)
        return 0
    if args.task5:
        return run_task5(args)
    if args.test4:
        return run_self_tests4(args)
    if args.task4:
        return run_task4(args)
    if args.test3:
        return run_self_tests3(args)
    if args.task3:
        return run_task3(args)
    if args.test:
        return run_self_tests()
    if args.urls:
        with open(args.urls, encoding="utf-8") as f:
            urls = [u.strip() for u in f if u.strip() and not u.startswith("#")]
        FETCH_CFG["per_domain"] = 4                     # real saytlarga yumshoqroq
        need = max(max(args.loads), args.pages)
        if len(urls) < need + 1:
            sys.exit(f"Kamida {need + 1} ta URL kerak (hozir {len(urls)})")
        run_all(args, urls[:need], urls[need:need + args.batch], f"real saytlar ({args.urls})")
        return 0
    n_pages = max(max(args.loads), args.pages) + args.batch
    LOG(f"Lokal sinov serveri ishga tushmoqda ({n_pages} sahifa, kechikish "
        f"{args.delay} s) ...")
    with LocalTestServer(n_pages, args.page_kb, args.delay) as srv:
        main_n = max(max(args.loads), args.pages)
        run_all(args, srv.urls(main_n), srv.urls(args.batch, start=main_n))
    return 0


if __name__ == "__main__":
    mp.freeze_support()                     # Windows (spawn) uchun majburiy
    sys.exit(main())
