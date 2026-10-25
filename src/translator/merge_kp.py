"""Сбор точек с нескольких ПК в одну папку data/slovo_kp.

На КАЖДОМ ПК после extract_slovo.py:
    python merge_kp.py pack                  -> data/kp_<имя ПК>.zip

Перенести zip-файлы на ПК, где будет обучение (флешка, scp, Telegram «Избранное» и т.п.), затем:
    python merge_kp.py unpack data/kp_*.zip  -> всё в data/slovo_kp (существующие файлы не трогает)
    python merge_kp.py check                 -> всё ли на месте и одинаковые ли были настройки
"""
import argparse
import csv
import json
import platform
import sys
import zipfile
from collections import Counter
from pathlib import Path

from features import WORDS, SLOVO_DIR, KP_DIR


def pack(args):
    if not KP_DIR.exists():
        raise SystemExit(f"Нет папки {KP_DIR} — сначала запусти extract_slovo.py")
    out = Path(args.out) if args.out else KP_DIR.parent / f"kp_{platform.node()}.zip"
    files = [f for f in KP_DIR.rglob("*")
             if f.is_file() and not f.name.endswith(".tmp.npy") and (f.suffix in (".npy", ".json"))]
    # npy с float32 почти не жмутся — ZIP_STORED быстрее и размер почти тот же
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, f.relative_to(KP_DIR.parent))      # пути вида slovo_kp/train/<id>/<file>.npy
    n_npy = sum(f.suffix == ".npy" for f in files)
    print(f"Упаковано {n_npy} файлов точек -> {out} ({out.stat().st_size / 1e6:.0f} МБ)")


def unpack(args):
    added = skipped = 0
    for zpath in args.zips:
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if not name.startswith("slovo_kp/") or name.endswith("/"):
                    continue
                dst = KP_DIR.parent / name
                if dst.exists():
                    skipped += 1
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".part")
                tmp.write_bytes(z.read(name))
                tmp.rename(dst)
                added += 1
        print(f"{zpath}: готово")
    print(f"Добавлено файлов: {added}, уже были: {skipped} -> {KP_DIR}")


def check(args):
    # 1. одинаковые ли настройки на всех ПК
    metas = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((KP_DIR / "_meta").glob("*.json"))]
    if metas:
        print("Запуски извлечения:")
        for m in metas:
            print(f"  {m['host']:<20} shard {m['shard']:<8} max_side {m['max_side']:<5} "
                  f"frame_step {m['frame_step']}  классов {m['classes']}  видео {m['videos']}")
        for key in ("max_side", "frame_step", "classes"):
            if len({m[key] for m in metas}) > 1:
                print(f"  ВНИМАНИЕ: разный {key} на разных ПК — проверь, что запускали с одинаковыми флагами/vocab.json")
        shards = {m["shard"].split("/")[1] for m in metas}
        if len(shards) > 1:
            print("  ВНИМАНИЕ: разное число частей (N в --shard) — части могут пересекаться или не покрыть всё")

    # 2. сколько видео ожидалось и сколько есть
    label_to_id = {label: wid for wid, (label, _) in WORDS.items()}
    expected = Counter()
    ann = SLOVO_DIR / "annotations.csv"
    if not ann.exists():
        print(f"\nНет {ann} — не могу сравнить с разметкой, показываю только то, что есть")
    else:
        with open(ann, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                wid = label_to_id.get(row["text"])
                if wid:
                    expected[("train" if row["train"] == "True" else "test", wid)] += 1

    have = Counter()
    for split in ("train", "test"):
        for d in (KP_DIR / split).glob("*"):
            if d.is_dir():
                have[(split, d.name)] = sum(1 for f in d.glob("*.npy") if not f.name.endswith(".tmp.npy"))

    tot_e, tot_h = sum(expected.values()), sum(have[k] for k in expected) if expected else sum(have.values())
    print(f"\nФайлов точек: {sum(have.values())}" + (f" из {tot_e} ({100 * tot_h / max(tot_e, 1):.1f}%)" if expected else ""))
    if expected:
        missing = {k: expected[k] - have[k] for k in expected if have[k] < expected[k]}
        empty_train = sorted({w for (s, w) in expected if s == "train" and have[(s, w)] == 0})
        if not missing:
            print("Всё на месте, можно обучать: python train.py --device cuda")
        else:
            print(f"Не хватает {sum(missing.values())} файлов в {len(missing)} папках (класс/сплит).")
            if empty_train:
                print(f"Классы без единого train-файла ({len(empty_train)}): {', '.join(empty_train[:20])}"
                      + (" ..." if len(empty_train) > 20 else ""))
            print("Скорее всего, не все zip распакованы или какой-то ПК не дошёл до конца.")
            print("Дозапустить недостающее на этом ПК: python extract_slovo.py   (готовое пропустится)")
            sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="упаковать data/slovo_kp этого ПК в zip")
    p.add_argument("--out", help="куда сохранить zip (по умолчанию data/kp_<имя ПК>.zip)")
    u = sub.add_parser("unpack", help="распаковать zip-файлы с других ПК в data/slovo_kp")
    u.add_argument("zips", nargs="+")
    sub.add_parser("check", help="проверить, что все видео словаря обработаны")
    args = ap.parse_args()
    {"pack": pack, "unpack": unpack, "check": check}[args.cmd](args)


if __name__ == "__main__":
    main()
