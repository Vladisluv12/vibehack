"""Собирает словарь для обучения (vocab.json) из annotations.csv Slovo.

В словарь всегда входят no_event и текущие базовые слова из features.py,
остальные добираются до --size: из --words-file (по одной метке на строку, как в колонке text)
или случайно (фиксированный seed — у всех получится одинаково).

python make_vocab.py --words-file words_basic.txt   # базовые 23 + все слова из файла (~210)
python make_vocab.py --size 200                      # базовые + случайные до 200
python make_vocab.py --all                 # все 1000 жестов
python make_vocab.py --remove              # удалить vocab.json и вернуться к словарю из features.py

vocab.json нужно закоммитить: на всех компьютерах словарь должен быть одинаковым.
"""
import argparse
import csv
import json
import random
import re
from collections import Counter

from features import SLOVO_DIR, VOCAB_FILE

# базовый словарь — тот же, что захардкожен в features.py (порядок и id сохраняем)
BASE = {
    "_none": ("no_event", ""),
    "privet": ("Привет!", "привет"), "poka": ("Пока", "пока"),
    "dobro_pozhalovat": ("Добро пожаловать!", "добро пожаловать"), "kto": ("кто?", "кто"),
    "chto": ("что", "что"), "da": ("да", "да"), "ya": ("я", "я"), "ty": ("ты", "ты"),
    "my": ("мы", "мы"), "vy": ("вы", "вы"), "menya": ("меня", "меня"), "tebya": ("тебя", "тебя"),
    "imya": ("имя", "имя"), "horosho": ("хорошо", "хорошо"), "ploho": ("Плохо", "плохо"),
    "drug": ("друг", "друг"), "hotet": ("хотеть", "хотеть"), "znat": ("знать", "знать"),
    "ponyat": ("понять", "понять"), "pomoch": ("помочь", "помочь"), "rad": ("рад", "рад"),
    "ochen": ("очень", "очень"), "tozhe": ("тоже", "тоже"),
}

TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
              ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p",
               "r", "s", "t", "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"]))


def make_id(label, used):
    s = "".join(TR.get(c, c) for c in label.lower())
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_") or "w"
    wid, k = s, 2
    while wid in used:
        wid, k = f"{s}_{k}", k + 1
    return wid


def display(label):
    text = label.strip().rstrip("!?").strip()
    return text[:1].lower() + text[1:] if len(text) > 1 else text


def is_letter(label):
    return len(label) == 1 and label.isupper()   # дактиль: А, Б, ... Я


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=None,
                    help="сколько слов без no_event (по умолчанию: все из --words-file, иначе 200)")
    ap.add_argument("--all", action="store_true", help="взять все жесты Slovo")
    ap.add_argument("--words-file", help="свой список меток, по одной на строку")
    ap.add_argument("--with-letters", action="store_true", help="включать буквы дактиля")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()

    if args.remove:
        VOCAB_FILE.unlink(missing_ok=True)
        print("vocab.json удалён, используется словарь из features.py")
        return

    with open(SLOVO_DIR / "annotations.csv", newline="", encoding="utf-8-sig") as f:
        counts = Counter(row["text"] for row in csv.DictReader(f, delimiter="\t"))
    available = [l for l in sorted(counts) if l != "no_event" and (args.with_letters or not is_letter(l))]

    words = dict(BASE)
    used_labels = {label for label, _ in words.values()}
    used_ids = set(words)

    wanted = []
    if args.words_file:
        lines = [l.strip() for l in open(args.words_file, encoding="utf-8") if l.strip()]
        bad = [l for l in lines if l not in counts]
        if bad:
            print("ВНИМАНИЕ: этих меток нет в annotations.csv, пропускаю:", bad)
        wanted = [l for l in lines if l in counts]
    rest = [l for l in available if l not in wanted]
    random.Random(args.seed).shuffle(rest)
    candidates = wanted + rest

    if args.all:
        target = len(available)
    elif args.size:
        target = args.size
    else:
        target = len(set(wanted) | {label for label, _ in BASE.values()}) - 1 if wanted else 200
    for label in candidates:
        if len(words) - 1 >= target:
            break
        if label in used_labels:
            continue
        wid = make_id(label, used_ids)
        words[wid] = (label, display(label))
        used_ids.add(wid)
        used_labels.add(label)

    VOCAB_FILE.write_text(json.dumps(words, ensure_ascii=False, indent=1), encoding="utf-8")
    n_videos = sum(counts[label] for label, _ in words.values())
    print(f"vocab.json: {len(words) - 1} слов + no_event, видео: {n_videos}")
    print("Примеры:", ", ".join(ru for _, (_, ru) in list(words.items())[len(BASE):len(BASE) + 15]), "...")


if __name__ == "__main__":
    main()
