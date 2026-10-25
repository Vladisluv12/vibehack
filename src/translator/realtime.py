"""Распознавание жестов с вебки в реальном времени.

Клавиши: c — очистить фразу, q / Esc — выход.
Фраза завершается, когда рук нет в кадре ~1.5 с; русский текст печатается в терминал.
"""
import argparse
import time
from collections import deque

from console import LiveStatus, add_log_arguments, configure_console

if __name__ == "__main__":
    configure_console()

import cv2
import numpy as np
import torch

from features import (WORDS, NONE, WINDOW_SEC, MODEL_PATH, FLAG_LH, FLAG_RH, create_holistic,
                      open_camera, process, draw_landmarks, frame_features, make_input, put_lines)
from model import model_from_checkpoint

PHRASE_END_SEC = 1.5


def ru(word_id):
    return WORDS.get(word_id, (word_id, word_id))[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.6, help="минимальная уверенность")
    ap.add_argument("--stride", type=int, default=3, help="запускать модель каждые N кадров")
    ap.add_argument("--votes", type=int, default=3, help="сколько окон подряд должны согласиться")
    add_log_arguments(ap)
    args = ap.parse_args()
    if args.stride < 1 or args.votes < 1 or not 0 <= args.thr <= 1:
        ap.error("--stride и --votes должны быть больше нуля; --thr — от 0 до 1")
    print(f"Загрузка модели: {args.model}…", flush=True)

    ckpt = torch.load(args.model, map_location="cpu")
    classes = ckpt["classes"]
    model = model_from_checkpoint(ckpt)
    model.eval()
    print(f"Модель загружена | Классов: {len(classes)} | Устройство: CPU | "
          f"Порог уверенности: {args.thr:.0%}", flush=True)
    print(f"Окно: {WINDOW_SEC:.1f} с | Подтверждений: {args.votes} | "
          f"Распознавание каждые {args.stride} кадра", flush=True)
    none_id = classes.index(NONE) if NONE in classes else -1

    # окно по ВРЕМЕНИ, а не по числу кадров: FPS вебки и видео Slovo разные
    buf = deque()                     # (timestamp, features)
    votes = deque(maxlen=args.votes)
    phrase, last_word, frame_i = [], None, 0
    last_hands_t = time.time()
    top3 = []

    print(f"Открытие камеры {args.cam}…", flush=True)
    cap = open_camera(args.cam)
    print("Камера открыта. Инициализация Holistic…", flush=True)
    print("Выход: q / Esc или Ctrl+C; очистить фразу: c.", flush=True)
    live = LiveStatus(args.log_interval)
    reason = "выход пользователя"
    recognized, inference_count, inference_ms = 0, 0, 0.0
    try:
        with create_holistic() as holistic:
            print("Holistic готов. Ожидание жестов…", flush=True)
            while True:
                ok, frame = cap.read()
                if not ok:
                    reason = "камера перестала выдавать кадры"
                    print("ОШИБКА: кадр с камеры не получен.", flush=True)
                    break
                now = time.time()
                frame_i += 1
                res = process(holistic, frame)
                draw_landmarks(frame, res)
                feat = frame_features(res)
                buf.append((now, feat))
                while buf and now - buf[0][0] > WINDOW_SEC:
                    buf.popleft()

                hands = feat[FLAG_LH] > 0 or feat[FLAG_RH] > 0
                if hands:
                    last_hands_t = now
                elif phrase and now - last_hands_t > PHRASE_END_SEC:
                    print("Фраза:", " ".join(ru(w) for w in phrase).capitalize(), flush=True)
                    phrase, last_word = [], None

                window_full = len(buf) >= 8 and now - buf[0][0] > 0.8 * WINDOW_SEC
                if window_full and frame_i % args.stride == 0:
                    if not hands:
                        votes.append(none_id)
                        top3 = []
                    else:
                        x = torch.from_numpy(make_input(np.stack([f for _, f in buf]), ckpt["seq_len"]))[None]
                        inference_started = time.perf_counter()
                        with torch.no_grad():
                            probs = model(x).softmax(-1)[0].numpy()
                        inference_count += 1
                        inference_ms += (time.perf_counter() - inference_started) * 1000
                        order = probs.argsort()[::-1][:3]
                        top3 = [(classes[i], probs[i]) for i in order]
                        best = int(order[0])
                        votes.append(best if probs[best] >= args.thr else none_id)

                    if len(votes) == args.votes and len(set(votes)) == 1:
                        cls = votes[0]
                        if cls == none_id:
                            last_word = None            # пауза — то же слово можно повторить
                        elif classes[cls] != last_word:
                            last_word = classes[cls]
                            phrase.append(last_word)
                            recognized += 1
                            print(f"Распознано: {ru(last_word)} ({last_word})", flush=True)
                            buf.clear(); votes.clear()  # следующий жест — с чистого окна

                frame = cv2.flip(frame, 1)
                lines = [f"{c}: {p:.2f}" for c, p in top3] or ["(no hands)"]
                put_lines(frame, lines, (0, 255, 255))
                put_lines(frame, ["phrase: " + " ".join(phrase)], (0, 255, 0), y0=frame.shape[0] - 30)
                cv2.imshow("RSL Translator", frame)
                if not hands:
                    state = "руки не в кадре"
                elif not window_full:
                    filled = now - buf[0][0] if buf else 0
                    state = f"накопление окна: {filled:.1f}/{WINDOW_SEC:.1f} с"
                elif top3:
                    candidate, confidence = top3[0]
                    state = f"кандидат: {ru(candidate) or 'нет жеста'} ({confidence:.0%})"
                else:
                    state = "ожидание жеста"
                avg_ms = inference_ms / inference_count if inference_count else 0
                live.tick(f"{state} | слов: {recognized} | "
                          f"прогнозов: {inference_count} | модель: {avg_ms:.1f} мс")

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    phrase, last_word = [], None
                    print("Фраза очищена.", flush=True)
    except KeyboardInterrupt:
        reason = "остановлено пользователем"
    except BaseException:
        reason = "ошибка обработки"
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        live.finish(reason)
        if phrase:
            print("Текущая фраза:", " ".join(ru(word) for word in phrase), flush=True)
        print(f"Всего распознано слов: {recognized}. Прогнозов модели: {inference_count}.", flush=True)



if __name__ == "__main__":
    main()
