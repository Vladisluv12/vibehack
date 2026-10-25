"""Распознавание жестов с вебки в реальном времени.

Клавиши: c — очистить фразу, q / Esc — выход.
Фраза завершается, когда рук нет в кадре ~1.5 с; русский текст печатается в терминал.
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

from features import (WORDS, NONE, WINDOW_SEC, MODEL_PATH, FLAG_LH, FLAG_RH, create_holistic,
                      open_camera, process, draw_landmarks, frame_features, make_input, put_lines)
from model import SignGRU

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
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    classes = ckpt["classes"]
    model = SignGRU(ckpt["in_dim"], len(classes), ckpt["hidden"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    none_id = classes.index(NONE) if NONE in classes else -1

    # окно по ВРЕМЕНИ, а не по числу кадров: FPS вебки и видео Slovo разные
    buf = deque()                     # (timestamp, features)
    votes = deque(maxlen=args.votes)
    phrase, last_word, frame_i = [], None, 0
    last_hands_t = time.time()
    top3 = []

    cap = open_camera(args.cam)
    with create_holistic() as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
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
                print(">>", " ".join(ru(w) for w in phrase).capitalize())
                phrase, last_word = [], None

            window_full = len(buf) >= 8 and now - buf[0][0] > 0.8 * WINDOW_SEC
            if window_full and frame_i % args.stride == 0:
                if not hands:
                    votes.append(none_id)
                    top3 = []
                else:
                    x = torch.from_numpy(make_input(np.stack([f for _, f in buf])))[None]
                    with torch.no_grad():
                        probs = model(x).softmax(-1)[0].numpy()
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
                        print("  +", last_word, f"({ru(last_word)})")
                        buf.clear(); votes.clear()  # следующий жест — с чистого окна

            frame = cv2.flip(frame, 1)
            lines = [f"{c}: {p:.2f}" for c, p in top3] or ["(no hands)"]
            put_lines(frame, lines, (0, 255, 255))
            put_lines(frame, ["phrase: " + " ".join(phrase)], (0, 255, 0), y0=frame.shape[0] - 30)
            cv2.imshow("RSL Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                phrase, last_word = [], None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
