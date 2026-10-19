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
AUTHORS = {"Boltayev Elbek": [], "Soyibjonov Otabek": [], "Ikkalasi": []}


def muallif(name):
    """Funksiya muallifini reyestrga yozadi; funksiyaning o'zini o'zgartirmaydi
    (shuning uchun pickle va ProcessPool bilan muammo yo'q)."""
    def deco(fn):
        AUTHORS[name].append(fn.__qualname__)
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
    return words, weights


@muallif(ELBEK)
def generate_page(page_id, size_kb=40):
    """Berilgan id uchun DETERMINISTIK HTML sahifa yasaydi (har safar bir xil):
    sarlavha, paragraflar, havolalar va tahlilda tashlab yuborilishi kerak
    bo'lgan <script>/<style> bloklari."""
    words, weights = build_vocabulary()
    rng = random.Random(1_000_003 * page_id + 12345)
    title = " ".join(rng.choices(words[:200], k=4))
    head = (f"<!DOCTYPE html><html lang='uz'><head><meta charset='utf-8'>"
            f"<title>Sahifa {page_id}: {title}</title>"
            f"<style>body{{font-family:sans-serif}} .stilsoz{{color:red}}</style>"
            f"<script>var yashirin = 'skriptsoz skriptsoz';</script></head>"
            f"<body><h1>{title}</h1>\n")
    parts, size, target = [head], len(head), size_kb * 1024
    while size < target:
        para = " ".join(rng.choices(words, weights=weights, k=rng.randint(30, 80)))
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
def _server_process_main(port_q, n_pages, size_kb, delay):
    """Server ALOHIDA JARAYONDA ishlaydi: shunda server oqimlari o'lchanayotgan
    mijoz jarayonining GIL iga ta'sir qilmaydi."""
    pages = {i: generate_page(i, size_kb).encode("utf-8") for i in range(n_pages)}
    srv = _QuietServer(("127.0.0.1", 0), make_handler(pages, delay))
    port_q.put(srv.server_address[1])
    srv.serve_forever()


class LocalTestServer:
    """Kontekst menejer: `with LocalTestServer(...) as srv:` - serverni ishga
    tushiradi va oxirida to'xtatadi."""

    @muallif(ELBEK)
    def __init__(self, n_pages, size_kb=40, delay=0.05):
        self.n_pages, self.size_kb, self.delay = n_pages, size_kb, delay
        self.proc, self.port = None, None

    def __enter__(self):
        q = mp.Queue()
        self.proc = mp.Process(target=_server_process_main,
                               args=(q, self.n_pages, self.size_kb, self.delay),
                               daemon=True)
        self.proc.start()
        self.port = q.get(timeout=120)
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
    ax.set_title("1-grafik. Ketma-ket versiya: vaqtning yuklamaga bog'liqligi")
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
    fig.suptitle("2-grafik. Oqimlar va jarayonlar: tezlashish (GIL ta'siri o'ngda)")
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
    ax.set_title("3-grafik. parallel_reduce: so'zlar chastotasi")
    ax.grid(alpha=.3)
    ax.legend()
    save(fig, "3_reduce.png")

    for key, fname, title, label in (
            ("invoke", "4_invoke.png", "4-grafik. parallel_invoke: 3 ta mustaqil vazifa", "mode"),
            ("versions", "5_versiyalar.png", "5-grafik. Loyihaning 4 versiyasi", "version")):
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
    ax.set_title("6-grafik. CPU va devor vaqti: IO-bound vs CPU-bound")
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
        + (f"; foydali jarayonlar soni yadrolar soni ({CPU_COUNT}) bilan cheklanadi, undan "
           f"ko'pida ({wmax} ta: S = {ap_:.2f}) faqat qo'shimcha xarajat ortadi."
           if best_ap["speedup"] > 1 else
           f". Bu yuklamada tahlil hajmi juda kichik - jarayon yaratish va HTML ni pickle "
           f"qilish xarajati yutuqdan katta; --pages yoki --page-kb ni oshirib ko'ring."),
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
    md = [f"# Parallel veb-skreyper va matn tahlili tizimi - 2-topshiriq natijalari",
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
    ap = argparse.ArgumentParser(description="Parallel veb-skreyper - 2-topshiriq")
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
    a = ap.parse_args(argv)
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
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    args = parse_args(argv)
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
