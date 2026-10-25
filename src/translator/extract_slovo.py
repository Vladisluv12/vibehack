"""Извлекает точки Holistic для выбранного словаря Slovo.

Сохраняет data/slovo_kp/{train,test}/<id>/<user_id>_<attachment_id>.npy.
Готовые файлы пропускаются; прогресс показывает скорость и оставшееся время.

python extract_slovo.py --workers 6
python extract_slovo.py --workers 4 --verbose-logs

Работа на нескольких компьютерах (у всех одинаковые vocab.json, annotations.csv и флаги):
    3 одинаковых ПК:      --shard 0/3   --shard 1/3   --shard 2/3
    мощный ПК + 2 ноута:  мощный --shard 0,1/4 --workers 16,  ноуты --shard 2/4 и --shard 3/4
Видео делятся по хешу attachment_id — части не пересекаются и вместе покрывают всё.
Потом: python merge_kp.py pack (на каждом) -> перенести zip -> python merge_kp.py unpack *.zip -> check.

Ускорение (одинаково на всех ПК!): --max-side 640 --frame-step 2  (~2-3x быстрее)
"""
import argparse
import atexit
import csv
import hashlib
import json
import platform
import math
import multiprocessing as mproc
import os
import re
import sys
import threading
import time
import warnings
from pathlib import Path


def quiet_library_logs():
    """Убирает служебные INFO/WARNING из stderr, сохраняя ошибки и traceback.

    MediaPipe пишет часть сообщений напрямую из C++, поэтому настройки
    Python logging и warnings недостаточно. Фильтр устанавливается до импорта.
    """
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


if __name__ in ("__main__", "__mp_main__") and "--verbose-logs" not in sys.argv:
    quiet_library_logs()

import cv2
import numpy as np

from features import WORDS, SLOVO_DIR, KP_DIR, mp_holistic, process, frame_features

MAX_SIDE = 960


def work(job):
    video, out, max_side, frame_step = job
    if out.exists():
        return "skip", video
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        return "fail", video
    feats = []
    # Отдельный Holistic для каждого видео: трекинг не переходит между роликами.
    try:
        with mp_holistic.Holistic(model_complexity=1, smooth_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5) as h:
            i = 0
            while True:
                if i % frame_step:
                    # пропускаемый кадр: grab() без декодирования в картинку — быстрее read()
                    i += 1
                    if not cap.grab():
                        break
                    continue
                i += 1
                ok, frame = cap.read()
                if not ok:
                    break
                hh, ww = frame.shape[:2]
                k = max_side / max(hh, ww)
                if k < 1:
                    frame = cv2.resize(frame, (int(ww * k), int(hh * k)), interpolation=cv2.INTER_AREA)
                feats.append(frame_features(process(h, frame)))
    finally:
        cap.release()
    if not feats:
        return "fail", video
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, np.stack(feats))
    tmp.rename(out)
    return "ok", video


def duration(seconds):
    if seconds is None:
        return "расчёт…"
    seconds = max(0, math.ceil(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slovo", type=Path, default=SLOVO_DIR)
    ap.add_argument("--out", type=Path, default=KP_DIR)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--log-interval", type=float, default=2,
                    help="интервал обновления прогресса в секундах (по умолчанию 2)")
    ap.add_argument("--shard", default="0/1",
                    help="части работы для этого ПК: I/N или I,J/N, например 1/3 или 0,1/4")
    ap.add_argument("--max-side", type=int, default=MAX_SIDE,
                    help=f"уменьшать кадр до этой стороны (по умолчанию {MAX_SIDE}; 640 — быстрее)")
    ap.add_argument("--frame-step", type=int, default=1,
                    help="обрабатывать каждый N-й кадр (по умолчанию 1; 2 — вдвое быстрее)")
    ap.add_argument("--verbose-logs", action="store_true",
                    help="показывать служебные сообщения MediaPipe")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers должен быть больше нуля")
    if not math.isfinite(args.log_interval) or args.log_interval <= 0:
        ap.error("--log-interval должен быть положительным конечным числом")

    try:
        parts_str, n_str = args.shard.split("/")
        shard_n = int(n_str)
        my_parts = {int(x) for x in parts_str.split(",")}
        assert shard_n >= 1 and all(0 <= x < shard_n for x in my_parts)
    except (ValueError, AssertionError):
        ap.error("--shard должен быть вида I/N или I,J/N, где 0 <= I < N, например 1/3 или 0,1/4")
    if args.max_side < 128 or args.frame_step < 1:
        ap.error("--max-side >= 128, --frame-step >= 1")

    def my_shard(attachment_id):
        # md5, а не hash(): одинаково на всех компьютерах и при любом порядке строк
        return int(hashlib.md5(attachment_id.encode()).hexdigest(), 16) % shard_n in my_parts

    label_to_id = {label: wid for wid, (label, _) in WORDS.items()}
    jobs, found = [], {wid: 0 for wid in WORDS}
    with open(args.slovo / "annotations.csv", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            wid = label_to_id.get(row["text"])
            if wid is None or not my_shard(row["attachment_id"]):
                continue
            split = "train" if row["train"] == "True" else "test"
            video = args.slovo / split / f"{row['attachment_id']}.mp4"
            out = args.out / split / wid / f"{row['user_id']}_{row['attachment_id']}.npy"
            jobs.append((video, out, args.max_side, args.frame_step))
            found[wid] += 1

    missing = [w for w, n in found.items() if n == 0]
    if missing:
        print("Метки без видео:", {w: WORDS[w][0] for w in missing}, flush=True)
    total = len(jobs)
    pending = [job for job in jobs if not job[1].exists()]

    # параметры запуска — merge_kp.py check проверит, что на всех ПК они одинаковые
    meta_dir = args.out / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    parts_name = "-".join(map(str, sorted(my_parts)))
    (meta_dir / f"shard_{parts_name}_of_{shard_n}_{platform.node()}.json").write_text(json.dumps({
        "host": platform.node(), "shard": args.shard, "max_side": args.max_side,
        "frame_step": args.frame_step, "classes": len(WORDS), "videos": len(jobs),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    stats = {"ok": 0, "skip": total - len(pending), "fail": 0}
    print("Извлечение точек Slovo", flush=True)
    print(f"Данные: {args.slovo.resolve()}", flush=True)
    print(f"Результат: {args.out.resolve()}", flush=True)
    print(f"Части: {sorted(my_parts)} из {shard_n}" if shard_n > 1 else "Части: все видео", flush=True)
    print(f"Кадр до {args.max_side}px, каждый {args.frame_step}-й кадр", flush=True)
    print(f"Выбрано видео: {total} | Классов: {sum(n > 0 for n in found.values())} | Процессов: {args.workers}", flush=True)
    print(f"Уже готовы: {stats['skip']} | Осталось обработать: {len(pending)}", flush=True)
    if not total:
        print("Нет видео для выбранного словаря. Проверь разметку и WORDS.", flush=True)
        return 1
    if not pending:
        print(f"Готово: {total}/{total} (100%). Все файлы точек уже существуют.", flush=True)
        return 0

    start = time.monotonic()
    last_print = -math.inf
    finished_pending = 0
    tty = sys.stdout.isatty()

    def progress(force=False):
        nonlocal last_print
        now = time.monotonic()
        if not force and now - last_print < args.log_interval:
            return
        last_print = now
        elapsed = now - start
        # Пропущенные при старте файлы не увеличивают оценку скорости.
        rate = finished_pending / elapsed if elapsed > 0 else 0
        remaining = len(pending) - finished_pending
        eta = remaining / rate if rate else None
        done = sum(stats.values())
        line = (f"{done}/{total} ({100 * done / total:.1f}%) | "
                f"новых: {stats['ok']} | пропущено: {stats['skip']} | ошибок: {stats['fail']} | "
                f"{rate * 60:.1f} видео/мин | прошло {duration(elapsed)} | осталось ~{duration(eta)}")
        if tty:
            line = (f"{done}/{total} ({100 * done / total:.1f}%) | {rate * 60:.0f}/мин | "
                    f"прошло {duration(elapsed)} | осталось ~{duration(eta)} | ошибок {stats['fail']}")
            print("\r" + line + "\033[K", end="", flush=True)
        else:
            print(line, flush=True)

    progress(force=True)
    ctx = mproc.get_context("spawn")
    try:
        with ctx.Pool(args.workers) as pool:
            results = pool.imap_unordered(work, pending, chunksize=1)
            while finished_pending < len(pending):
                try:
                    status, video = results.next(timeout=min(1, args.log_interval))
                except mproc.TimeoutError:
                    progress()
                    continue
                finished_pending += 1
                stats[status] += 1
                if status == "fail":
                    if tty:
                        print(flush=True)
                    print(f"ОШИБКА: не удалось прочитать видео {video}", flush=True)
                progress()
    except KeyboardInterrupt:
        if tty:
            print(flush=True)
        print("Обработка остановлена. Готовые файлы сохранятся и будут пропущены при следующем запуске.", flush=True)
        return 130

    progress(force=True)
    if tty:
        print(flush=True)
    print(f"Завершено за {duration(time.monotonic() - start)}. "
          f"Сохранено: {stats['ok']} | Пропущено: {stats['skip']} | Ошибок: {stats['fail']}", flush=True)
    print(f"Файлы точек: {args.out.resolve()}", flush=True)
    return 1 if stats["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
