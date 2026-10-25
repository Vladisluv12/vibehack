"""Общая разметка, задания по словам и отдельные отчёты устройств."""
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import io
import json
import re
import socket
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = PROJECT / "metadata" / "extraction"
MAX_SIDE = 960
HOLISTIC_OPTIONS = dict(model_complexity=1, smooth_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def feature_profile():
    source = (PROJECT / "src/translator/features.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {"POSE_IDX", "LIPS", "BROWS", "FACE_IDX", "FEAT_DIM"}
    code = []
    dimension = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"lm_to_np", "frame_features"}:
            code.append(ast.dump(node, include_attributes=False))
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            code.append(ast.dump(node, include_attributes=False))
            if any(isinstance(t, ast.Name) and t.id == "FEAT_DIM" for t in node.targets):
                dimension = ast.literal_eval(node.value)
    profile = {"mediapipe": importlib.metadata.version("mediapipe"), "feature_dim": dimension,
               "max_side": MAX_SIDE, "holistic": HOLISTIC_OPTIONS, "feature_code_sha256": digest(code)}
    return profile


def build_catalog(annotation_path, vocabulary, profile, directory, write=True):
    label_to_id = {value[0]: key for key, value in vocabulary.items()}
    if len(label_to_id) != len(vocabulary):
        raise ValueError("У двух word_id одинаковая метка Slovo")
    with Path(annotation_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [dict(row, word_id=label_to_id[row["text"]])
                for row in csv.DictReader(stream, delimiter="\t") if row["text"] in label_to_id]
    rows.sort(key=lambda row: (row["word_id"], row["attachment_id"]))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["word_id"]].append(row)
    words = {}
    for wid, (label, translation) in sorted(vocabulary.items()):
        clips = grouped[wid]
        words[wid] = {"label": label, "translation": translation,
                      "train": sum(row["train"] == "True" for row in clips),
                      "test": sum(row["train"] != "True" for row in clips),
                      "annotation_digest": digest(clips),
                      "attachment_ids": sorted(row["attachment_id"] for row in clips)}
    catalog = {"schema_version": 1, "dataset": "Slovo", "preprocessing": profile,
               "feature_signature": digest(profile), "words": words}
    catalog["catalog_id"] = digest(catalog)
    if write:
        directory = Path(directory)
        save_json(directory / "catalog.json", catalog)
        fields = ["word_id", "attachment_id", "text", "user_id", "height", "width", "length", "train"]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        target = directory / "annotations.tsv"
        if not target.exists() or target.read_text(encoding="utf-8") != buffer.getvalue():
            temporary = target.with_suffix(".tsv.tmp")
            temporary.write_text(buffer.getvalue(), encoding="utf-8")
            temporary.replace(target)
    return catalog, rows


def ensure_profile(output_dir, profile, adopt_existing=False, write=True):
    output_dir = Path(output_dir)
    marker = output_dir / ".extraction-profile.json"
    signature = digest(profile)
    if marker.exists():
        if load_json(marker).get("feature_signature") != signature:
            raise ValueError("В папке точек другой формат/версия MediaPipe. Укажи отдельную --out.")
    else:
        existing = next(output_dir.rglob("*.npy"), None) if output_dir.exists() else None
        if existing is not None and not adopt_existing:
            raise ValueError("У существующих .npy нет версии обработки. Если они получены этим же "
                             "кодом и MediaPipe, один раз добавь --adopt-existing; иначе укажи новую --out.")
        if write:
            save_json(marker, {"feature_signature": signature, "preprocessing": profile})


def output_valid(path, dimension=425):
    import numpy as np
    try:
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        return (data.ndim == 2 and data.shape[0] > 0 and data.shape[1] == dimension
                and data.dtype == np.float32 and bool(np.isfinite(data).all()))
    except (OSError, ValueError, EOFError):
        return False


def device_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise argparse.ArgumentTypeError("device-id: 1–64 латинских букв, цифр, точек, '_' или '-'")
    return value


def default_device_id():
    name = re.sub(r"[^A-Za-z0-9_.-]", "-", socket.gethostname()).strip("._-")
    return (name or "device")[:64]


def write_report(directory, device, scope, catalog, rows, selected, output_dir, artifact_uri=None):
    device_id(device)
    device_id(scope)
    words = {}
    grouped = defaultdict(list)
    for row in rows:
        if row["word_id"] in selected:
            grouped[row["word_id"]].append(row)
    for wid in sorted(selected):
        completed, invalid = [], []
        clips = grouped[wid]
        for row in clips:
            split = "train" if row["train"] == "True" else "test"
            path = Path(output_dir) / split / wid / f"{row['user_id']}_{row['attachment_id']}.npy"
            if output_valid(path, catalog["preprocessing"]["feature_dim"]):
                completed.append(row["attachment_id"])
            elif path.exists():
                invalid.append(row["attachment_id"])
        state = ("missing_label" if not clips else "complete" if len(completed) == len(clips)
                 else "partial" if completed else "not_started")
        words[wid] = {"status": state, "expected": len(clips), "completed_ids": sorted(completed),
                      "invalid_ids": sorted(invalid),
                      "annotation_digest": catalog["words"][wid]["annotation_digest"]}
    report = {"schema_version": 1, "device_id": device, "task": scope,
              "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "feature_signature": catalog["feature_signature"], "artifact_uri": artifact_uri,
              "words": words}
    path = Path(directory) / "devices" / f"{device}__{scope}.json"
    save_json(path, report)
    return path


def registry_status(catalog, directory):
    completed = defaultdict(set)
    owners = defaultdict(set)
    for path in sorted((Path(directory) / "devices").glob("*.json")):
        report = load_json(path)
        if report.get("feature_signature") != catalog["feature_signature"]:
            continue
        for wid, entry in report.get("words", {}).items():
            spec = catalog["words"].get(wid)
            if not spec or entry.get("annotation_digest") != spec["annotation_digest"]:
                continue
            ids = set(entry.get("completed_ids", [])) & set(spec["attachment_ids"])
            completed[wid].update(ids)
            if ids:
                owners[wid].add(report["device_id"])
    done = {wid for wid, spec in catalog["words"].items()
            if spec["train"] + spec["test"] > 0
            and len(completed[wid]) >= spec["train"] + spec["test"]}
    return done, completed, owners


def make_plan(directory, parts, output=None):
    if parts < 1:
        raise ValueError("Число частей должно быть больше нуля")
    directory = Path(directory)
    catalog = load_json(directory / "catalog.json")
    done, completed, _ = registry_status(catalog, directory)
    tasks = [{"id": f"part-{i + 1:02d}", "words": [], "video_count": 0} for i in range(parts)]
    pending = [(wid, spec["train"] + spec["test"]) for wid, spec in catalog["words"].items()
               if wid not in done and spec["train"] + spec["test"] > 0]
    for wid, size in sorted(pending, key=lambda item: (-item[1], item[0])):
        task = min(tasks, key=lambda item: (item["video_count"], item["id"]))
        task["words"].append(wid)
        task["video_count"] += size
    for task in tasks:
        task["words"].sort()
    plan = {"schema_version": 1, "catalog_id": catalog["catalog_id"],
            "feature_signature": catalog["feature_signature"], "completed_words": sorted(done),
            "unavailable_words": sorted(wid for wid, spec in catalog["words"].items()
                                        if spec["train"] + spec["test"] == 0),
            "tasks": tasks}
    target = Path(output) if output else directory / "plan.json"
    if target.exists():
        raise FileExistsError(f"План уже существует: {target}. Для нового плана укажи --out.")
    save_json(target, plan)
    return target, plan


def task_words(plan_path, task_id, catalog):
    plan = load_json(plan_path)
    if plan["catalog_id"] != catalog["catalog_id"]:
        raise ValueError("Словарь, разметка или обработка изменились: нужен новый общий план")
    for task in plan["tasks"]:
        if task["id"] == task_id:
            return task["words"]
    raise ValueError(f"Задание {task_id!r} отсутствует в плане")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=REGISTRY_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="общий статус по опубликованным отчётам")
    plan_parser = sub.add_parser("plan", help="разделить ещё не готовые слова")
    plan_parser.add_argument("--parts", type=int, default=2)
    plan_parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            path, plan = make_plan(args.metadata_dir, args.parts, args.out)
            print(f"План: {path}", flush=True)
            print(f"Уже готовы и исключены из заданий: {len(plan['completed_words'])} слов", flush=True)
            for task in plan["tasks"]:
                print(f"{task['id']}: {len(task['words'])} слов, {task['video_count']} видео", flush=True)
        else:
            catalog = load_json(args.metadata_dir / "catalog.json")
            done, completed, owners = registry_status(catalog, args.metadata_dir)
            total = sum(spec["train"] + spec["test"] for spec in catalog["words"].values())
            print(f"По отчётам устройств: {len(done)}/{len(catalog['words'])} слов полностью; "
                  f"{sum(len(ids) for ids in completed.values())}/{total} видео", flush=True)
            for wid, spec in sorted(catalog["words"].items()):
                expected = spec["train"] + spec["test"]
                print(f"{wid:>20} | {spec['label']} | {len(completed[wid])}/{expected} | "
                      f"{'готово' if wid in done else 'ожидает'} | {', '.join(sorted(owners[wid])) or '—'}")
            print("Это реестр обработки. Наличие отчёта не означает, что .npy скачаны на это устройство.")
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
