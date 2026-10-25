"""Извлекает точки Holistic для выбранного словаря Slovo.

Сохраняет data/slovo_kp/{train,test}/<id>/<user_id>_<attachment_id>.npy.
Готовые файлы пропускаются; прогресс показывает скорость и оставшееся время.

python extract_slovo.py --workers 6
python extract_slovo.py --workers 4 --verbose-logs
"""
import argparse
import csv
import math
import multiprocessing as mproc
import os
import sys
import time
from pathlib import Path

from console import configure_console, duration




if __name__ in ("__main__", "__mp_main__"):
    configure_console()

import cv2
import numpy as np

from features import WORDS, SLOVO_DIR, KP_DIR, FEAT_DIM, mp_holistic, process, frame_features
from extraction_registry import (REGISTRY_DIR, MAX_SIDE, HOLISTIC_OPTIONS, build_catalog,
                                 default_device_id, device_id, digest, ensure_profile,
                                 feature_profile, output_valid, task_words, write_report)



def work(job):
    video, out = job
    if output_valid(out, FEAT_DIM):
        return "skip", video
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        return "fail", video
    feats = []
    # Отдельный Holistic для каждого видео: трекинг не переходит между роликами.
    try:
        with mp_holistic.Holistic(**HOLISTIC_OPTIONS) as h:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                hh, ww = frame.shape[:2]
                k = MAX_SIDE / max(hh, ww)
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




def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slovo", type=Path, default=SLOVO_DIR)
    ap.add_argument("--out", type=Path, default=KP_DIR)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--log-interval", type=float, default=2,
                    help="интервал обновления прогресса в секундах (по умолчанию 2)")
    ap.add_argument("--verbose-logs", action="store_true",
                    help="показывать служебные сообщения MediaPipe")
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument("--words", nargs="+", choices=sorted(WORDS),
                    help="extract only these word IDs; by default use the full vocabulary")
    ap.add_argument("--dry-run", action="store_true", help="show counts without extracting landmarks")
    selection.add_argument("--task", type=device_id, help="задание из общего плана: part-01, part-02")
    ap.add_argument("--plan", type=Path, help="общий план (по умолчанию metadata/extraction/plan.json)")
    ap.add_argument("--metadata-dir", type=Path, default=REGISTRY_DIR)
    ap.add_argument("--device-id", type=device_id, default=default_device_id())
    ap.add_argument("--report-only", action="store_true", help="обновить разметку и отчёт без извлечения")
    ap.add_argument("--artifact-uri", help="ссылка или адрес архива с точками; файлы не загружаются автоматически")
    ap.add_argument("--adopt-existing", action="store_true", help="принять старые .npy без версии, полученные тем же кодом")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers должен быть больше нуля")
    if not math.isfinite(args.log_interval) or args.log_interval <= 0:
        ap.error("--log-interval должен быть положительным конечным числом")

    try:
        annotation_source = args.slovo / "annotations.csv"
        if not annotation_source.is_file():
            annotation_source = args.metadata_dir / "annotations.tsv"
        profile = feature_profile()
        ensure_profile(args.out, profile, args.adopt_existing, write=not args.dry_run)
        catalog, annotation_rows = build_catalog(annotation_source, WORDS, profile, args.metadata_dir,
                                                 write=not args.dry_run)
        ids = task_words(args.plan or args.metadata_dir / "plan.json", args.task, catalog) if args.task else args.words
        selected = {wid: WORDS[wid] for wid in ids} if ids is not None else WORDS
    except (ValueError, FileNotFoundError, KeyError) as exc:
        ap.error(str(exc))
    scope = args.task or ("words-" + digest(sorted(selected))[:10] if args.words else "all")

    def publish_report():
        path = write_report(args.metadata_dir, args.device_id, scope, catalog, annotation_rows,
                            selected, args.out, args.artifact_uri)
        print(f"Отчёт устройства: {path}", flush=True)

    label_to_id = {label: wid for wid, (label, _) in selected.items()}
    jobs, found = [], {wid: 0 for wid in selected}
    profiles = {wid: {"train": 0, "test": 0, "users": set()} for wid in selected}
    with open(annotation_source, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            wid = label_to_id.get(row["text"])
            if wid is None:
                continue
            split = "train" if row["train"] == "True" else "test"
            video = args.slovo / split / f"{row['attachment_id']}.mp4"
            out = args.out / split / wid / f"{row['user_id']}_{row['attachment_id']}.npy"
            jobs.append((video, out))
            found[wid] += 1
            profiles[wid][split] += 1
            profiles[wid]["users"].add(row["user_id"])

    missing = [w for w, n in found.items() if n == 0]
    if missing:
        print("Метки без видео:", {w: WORDS[w][0] for w in missing}, flush=True)
    total = len(jobs)
    pending = [(video, out) for video, out in jobs if not output_valid(out, FEAT_DIM)]
    stats = {"ok": 0, "skip": total - len(pending), "fail": 0}
    print("Извлечение точек Slovo", flush=True)
    print(f"Данные: {args.slovo.resolve()}", flush=True)
    print(f"Результат: {args.out.resolve()}", flush=True)
    print(f"Выбрано видео: {total} | Классов: {sum(n > 0 for n in found.values())} | Процессов: {args.workers}", flush=True)
    print(f"Уже готовы: {stats['skip']} | Осталось обработать: {len(pending)}", flush=True)
    if args.report_only and not args.dry_run:
        publish_report()
        print("Разметка и реестр обновлены. Извлечение не запускалось.", flush=True)
        return 1 if missing else 0
    if not total:
        print("Нет видео для выбранного словаря. Проверь разметку и WORDS.", flush=True)
        return 1
    if args.dry_run:
        for wid, info in profiles.items():
            print(f"{wid:>20} | {selected[wid][0]} | train={info['train']} test={info['test']} "
                  f"people={len(info['users'])}", flush=True)
        absent = sum(not video.is_file() for video, _ in pending)
        print(f"Нужно извлечь: {len(pending)} | Нет исходного видео: {absent}", flush=True)
        return 1 if missing or absent else 0
    if missing:
        print("Исправь словарь: часть меток не найдена в annotations.csv.", flush=True)
        return 1
    if not pending:
        publish_report()
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

    publish_report()
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
    finally:
        if tty:
            print(flush=True)
        publish_report()

    progress(force=True)
    if tty:
        print(flush=True)
    print(f"Завершено за {duration(time.monotonic() - start)}. "
          f"Сохранено: {stats['ok']} | Пропущено: {stats['skip']} | Ошибок: {stats['fail']}", flush=True)
    print(f"Файлы точек: {args.out.resolve()}", flush=True)
    return 1 if stats["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
