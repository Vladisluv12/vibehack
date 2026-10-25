"""Общий вывод прогресса и фильтр служебных сообщений библиотек."""
import argparse
import atexit
import math
import os
import re
import sys
import threading
import time
import warnings

_quiet = False

def quiet_library_logs():
    """Убирает служебные INFO/WARNING из stderr, сохраняя ошибки и traceback.

    MediaPipe пишет часть сообщений напрямую из C++, поэтому настройки
    Python logging и warnings недостаточно. Фильтр устанавливается до импорта.
    """
    global _quiet
    if _quiet:
        return
    _quiet = True
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["GLOG_minloglevel"] = "2"
    warnings.filterwarnings("ignore")
    original = os.dup(2)
    reader, writer = os.pipe()
    os.dup2(writer, 2)
    os.close(writer)
    noise = re.compile(rb"^(?:[IWD]\d{4,8}\s|INFO:|WARNING:|libEGL warning:|\[\s*(?:WARN|INFO)(?::|\s))")

    def forward_errors():
        with os.fdopen(reader, "rb") as stream:
            for line in stream:
                if not noise.match(line):
                    remaining = memoryview(line)
                    while remaining:
                        remaining = remaining[os.write(original, remaining):]

    thread = threading.Thread(target=forward_errors, daemon=True)
    thread.start()

    def finish():
        sys.stderr.flush()
        os.dup2(original, 2)
        thread.join(timeout=1)
        if not thread.is_alive():
            os.close(original)

    atexit.register(finish)

def duration(seconds):
    if seconds is None:
        return "расчёт…"
    seconds = max(0, math.ceil(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def configure_console():
    if "--verbose-logs" not in sys.argv:
        quiet_library_logs()


def positive_interval(value):
    try:
        interval = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("интервал должен быть числом") from None
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError("интервал должен быть положительным конечным числом")
    return interval


def add_log_arguments(parser):
    parser.add_argument("--log-interval", type=positive_interval, default=2,
                        help="интервал вывода статуса в секундах (по умолчанию 2)")
    parser.add_argument("--verbose-logs", action="store_true",
                        help="показывать служебные сообщения библиотек")


class Progress:
    def __init__(self, label, total, interval=2, unit="клипов"):
        self.label, self.total, self.interval, self.unit = label, total, interval, unit
        self.started = time.monotonic()
        self.last_print = -math.inf
        self.update(0, force=True)

    def update(self, done, detail="", force=False):
        now = time.monotonic()
        if not force and done < self.total and now - self.last_print < self.interval:
            return
        self.last_print = now
        if not self.total:
            print(f"{self.label}: нет данных.", flush=True)
            return
        elapsed = now - self.started
        rate = done / elapsed if elapsed > 0 else 0
        remaining = max(0, self.total - done)
        eta = remaining / rate if rate else (0 if not remaining else None)
        percent = 100 * done / self.total if self.total else 100
        line = (f"{self.label}: {done}/{self.total} ({percent:.1f}%) | "
                f"{rate:.1f} {self.unit}/с | прошло {duration(elapsed)} | осталось ~{duration(eta)}")
        if detail:
            line += " | " + detail
        print(line, flush=True)


class LiveStatus:
    """Статус непрерывного потока: время и FPS, без вымышленного ETA."""
    def __init__(self, interval=2):
        self.interval = interval
        self.started = self.last_print = time.monotonic()
        self.frames = self.last_frames = 0

    def tick(self, detail):
        self.frames += 1
        now = time.monotonic()
        if now - self.last_print >= self.interval:
            fps = (self.frames - self.last_frames) / (now - self.last_print)
            print(f"Камера: {self.frames} кадров | {fps:.1f} FPS | "
                  f"работает {duration(now - self.started)} | {detail}", flush=True)
            self.last_print, self.last_frames = now, self.frames

    def finish(self, reason):
        elapsed = time.monotonic() - self.started
        print(f"Остановлено: {reason}. Обработано {self.frames} кадров за {duration(elapsed)} "
              f"(в среднем {self.frames / max(elapsed, 1e-6):.1f} FPS).", flush=True)
