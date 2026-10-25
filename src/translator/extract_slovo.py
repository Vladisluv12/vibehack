"""Прогоняет видео Slovo через MediaPipe Holistic и сохраняет точки.

Берёт только слова из WORDS (features.py) и класс no_event.
Результат: data/slovo_kp/{train,test}/<id>/<user_id>_<attachment_id>.npy, форма (кадры, 425).
Уже обработанные видео пропускаются — можно прервать (Ctrl+C) и запустить снова.

python extract_slovo.py              # все ядра / 2
python extract_slovo.py --workers 6
"""
import argparse
import csv
import multiprocessing as mproc
import os
import time
from pathlib import Path

import cv2
import numpy as np

from features import WORDS, SLOVO_DIR, KP_DIR, mp_holistic, process, frame_features

MAX_SIDE = 960  # видео в Slovo 1080x1920 — уменьшаем, точность почти не страдает, скорость x2-3


def work(job):
    video, out = job
    if out.exists():
        return "skip", video
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return "fail", video
    feats = []
    # новый Holistic на каждое видео: иначе трекинг тянется из предыдущего ролика
    with mp_holistic.Holistic(model_complexity=1, smooth_landmarks=True,
                              min_detection_confidence=0.5, min_tracking_confidence=0.5) as h:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            hh, ww = frame.shape[:2]
            k = MAX_SIDE / max(hh, ww)
            if k < 1:
                frame = cv2.resize(frame, (int(ww * k), int(hh * k)), interpolation=cv2.INTER_AREA)
            feats.append(frame_features(process(h, frame)))
    cap.release()
    if not feats:
        return "fail", video
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, np.stack(feats))
    tmp.rename(out)  # атомарно: прерванная обработка не оставит битый файл
    return "ok", video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slovo", type=Path, default=SLOVO_DIR)
    ap.add_argument("--out", type=Path, default=KP_DIR)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    label_to_id = {label: wid for wid, (label, _) in WORDS.items()}
    jobs, found = [], {wid: 0 for wid in WORDS}
    with open(args.slovo / "annotations.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            wid = label_to_id.get(row["text"])
            if wid is None:
                continue
            split = "train" if row["train"] == "True" else "test"
            video = args.slovo / split / f"{row['attachment_id']}.mp4"
            out = args.out / split / wid / f"{row['user_id']}_{row['attachment_id']}.npy"
            jobs.append((video, out))
            found[wid] += 1

    missing = [w for w, n in found.items() if n == 0]
    if missing:
        print("ВНИМАНИЕ: этих меток нет в annotations.csv, проверь написание:",
              {w: WORDS[w][0] for w in missing})
    print(f"видео к обработке: {len(jobs)}, воркеров: {args.workers}")

    t0, done, stats = time.time(), 0, {"ok": 0, "skip": 0, "fail": 0}
    ctx = mproc.get_context("spawn")  # mediapipe не любит fork
    with ctx.Pool(args.workers) as pool:
        for status, video in pool.imap_unordered(work, jobs, chunksize=2):
            done += 1
            stats[status] += 1
            if status == "fail":
                print("не удалось прочитать:", video)
            if done % 20 == 0 or done == len(jobs):
                el = time.time() - t0
                eta = el / done * (len(jobs) - done)
                print(f"{done}/{len(jobs)}  {stats}  прошло {el / 60:.1f} мин, осталось ~{eta / 60:.1f} мин")
    print("готово:", stats, "->", args.out)


if __name__ == "__main__":
    main()
