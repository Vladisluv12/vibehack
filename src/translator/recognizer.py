"""Распознавание жестов РЖЯ в видеофайле. Используется ботом, но можно запускать и отдельно:

python recognizer.py путь/к/видео.mp4

Логика:
  1. Каждый кадр -> MediaPipe -> 425 признаков (как при обучении).
  2. Обрезаем видео по кадрам, где видны руки.
  3. Короткое видео (<= 3 c) считаем одним жестом и классифицируем целиком —
     так же, как устроены клипы в Slovo.
  4. Длинное — скользящее окно 1.6 c, как в realtime.py: слово засчитывается,
     если несколько окон подряд уверенно согласны.
"""
import sys
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from features import (WORDS, NONE, WINDOW_SEC, MODEL_PATH, FLAG_LH, FLAG_RH, FEAT_DIM,
                      mp_holistic, process, frame_features, make_input)
from model import SignGRU

MAX_SIDE = 960          # уменьшаем большие видео, как в extract_slovo.py
STEP_SEC = 0.1          # шаг скользящего окна
SINGLE_MAX_SEC = 3.0    # видео (по кадрам с руками) короче этого — один жест
MARGIN_SEC = 0.2        # запас по краям при обрезке по рукам


def ru(word_id):
    return WORDS.get(word_id, (word_id, word_id))[1] or "нет жеста"


@dataclass
class Result:
    words: list = field(default_factory=list)       # распознанные слова [(id, вероятность)]
    candidates: list = field(default_factory=list)  # топ-3 по всему видео [(id, вероятность)]
    duration: float = 0.0
    hands_ratio: float = 0.0                        # доля кадров, где видна хотя бы одна рука

    @property
    def text(self):
        return " ".join(ru(w) for w, _ in self.words).capitalize()


class Recognizer:
    def __init__(self, model_path=MODEL_PATH, thr=0.6, votes=3):
        ckpt = torch.load(model_path, map_location="cpu")
        self.classes = ckpt["classes"]
        self.model = SignGRU(ckpt["in_dim"], len(self.classes), ckpt["hidden"])
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        self.none_id = self.classes.index(NONE) if NONE in self.classes else -1
        self.thr, self.votes = thr, votes

    @torch.no_grad()
    def predict(self, feats):
        x = torch.from_numpy(make_input(feats))[None]
        return self.model(x).softmax(-1)[0].numpy()

    def top(self, probs, k=3):
        return [(self.classes[i], float(probs[i])) for i in probs.argsort()[::-1][:k]]

    @staticmethod
    def extract(path):
        """Видео -> (признаки (N, 425), fps)."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"не удалось открыть видео {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1 or fps > 120:
            fps = 30.0
        feats = []
        # новый Holistic на каждое видео — трекинг не должен тянуться из прошлого файла
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
        return (np.stack(feats) if feats else np.zeros((0, FEAT_DIM), np.float32)), fps

    def recognize(self, path):
        feats, fps = self.extract(path)
        return self.recognize_features(feats, fps)

    def recognize_features(self, feats, fps):
        n = len(feats)
        res = Result(duration=n / fps)
        if n < 4:
            return res
        hands = (feats[:, FLAG_LH] > 0) | (feats[:, FLAG_RH] > 0)
        res.hands_ratio = float(hands.mean())
        if hands.sum() < 4:
            return res

        # обрезаем по первому/последнему кадру с руками
        idx = np.flatnonzero(hands)
        m = int(MARGIN_SEC * fps)
        a, b = max(0, idx[0] - m), min(n, idx[-1] + 1 + m)
        clip = feats[a:b]

        probs = self.predict(clip)
        res.candidates = self.top(probs)

        if len(clip) / fps <= SINGLE_MAX_SEC:
            best = int(probs.argmax())
            if best != self.none_id and probs[best] >= self.thr:
                res.words = [(self.classes[best], float(probs[best]))]
            return res

        res.words = self._sliding(clip, hands[a:b], fps)
        return res

    def _sliding(self, feats, hands, fps):
        """То же голосование, что в realtime.py, только по кадрам файла."""
        n = len(feats)
        win = max(4, int(round(WINDOW_SEC * fps)))
        step = max(1, int(round(STEP_SEC * fps)))
        votes = deque(maxlen=self.votes)
        words, last_word, buf_start = [], None, 0

        for i in range(n):
            w_start = max(buf_start, i + 1 - win)
            if (i + 1 - w_start) < 0.8 * win or i % step:
                continue
            if not hands[i]:
                votes.append((self.none_id, 0.0))
            else:
                probs = self.predict(feats[w_start:i + 1])
                best = int(probs.argmax())
                votes.append((best, float(probs[best])) if probs[best] >= self.thr else (self.none_id, 0.0))

            ids = {c for c, _ in votes}
            if len(votes) == self.votes and len(ids) == 1:
                cls = votes[0][0]
                if cls == self.none_id:
                    last_word = None
                elif self.classes[cls] != last_word:
                    last_word = self.classes[cls]
                    words.append((last_word, max(p for _, p in votes)))
                    buf_start = i + 1      # следующий жест — с чистого окна
                    votes.clear()
        return words


def main():
    if len(sys.argv) < 2:
        raise SystemExit("python recognizer.py видео.mp4 [ещё видео ...]")
    rec = Recognizer()
    for path in sys.argv[1:]:
        r = rec.recognize(path)
        print(f"\n{path}  ({r.duration:.1f} c, руки видны в {r.hands_ratio:.0%} кадров)")
        print("  текст:", r.text or "(ничего не распознано)")
        print("  слова:", [(ru(w), round(p, 2)) for w, p in r.words])
        print("  кандидаты:", [(ru(w), round(p, 2)) for w, p in r.candidates])


if __name__ == "__main__":
    main()
