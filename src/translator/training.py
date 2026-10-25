"""Shared data, augmentation, grouped splits, training and reporting."""
import csv
import json
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from features import (FEAT_DIM, FLAG_LH, FLAG_RH, N_XYZ, POSE_PAIRS, PT_LH,
                      PT_RH, PT_LH_LOCAL, PT_RH_LOCAL, SL_LH, SL_RH,
                      SL_LH_LOCAL, SL_RH_LOCAL, WORDS, make_input)
from model import ARCHITECTURES, build_model


@dataclass
class Config:
    arch: str = "gru"
    hidden: int = 192
    layers: int = 2
    dropout: float = 0.3
    heads: int = 4
    seq_len: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 32
    label_smoothing: float = 0.1
    balance: str = "sqrt"
    loss: str = "ce"
    focal_gamma: float = 2.0
    mixup: float = 0.0
    augmentation: bool = True
    mirror: bool = True
    scheduler: str = "cosine"

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        config = cls(**values)
        for name in ("hidden", "layers", "heads", "seq_len", "batch_size"):
            if type(getattr(config, name)) is not int or getattr(config, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if config.hidden < 2 or config.seq_len < 2:
            raise ValueError("hidden and seq_len must be >= 2")
        for name in ("augmentation", "mirror"):
            if type(getattr(config, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if config.arch not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {config.arch}")
        if config.arch == "transformer" and config.hidden % config.heads:
            raise ValueError("Transformer hidden must be divisible by heads")
        for name in ("lr", "weight_decay", "dropout", "label_smoothing", "focal_gamma", "mixup"):
            if not isinstance(getattr(config, name), (int, float)) or not np.isfinite(getattr(config, name)):
                raise ValueError(f"{name} must be finite")
        if config.lr <= 0 or min(config.weight_decay, config.focal_gamma, config.mixup) < 0:
            raise ValueError("lr must be > 0; weight_decay, focal_gamma, mixup must be >= 0")
        if not 0 <= config.dropout < 1 or not 0 <= config.label_smoothing < 1:
            raise ValueError("dropout and label_smoothing must be in [0, 1)")
        if config.balance not in ("none", "sqrt", "sampler"):
            raise ValueError("balance must be none, sqrt or sampler")
        if config.loss not in ("ce", "focal") or config.scheduler not in ("cosine", "onecycle"):
            raise ValueError("Invalid loss or scheduler")
        return config

    def to_dict(self):
        return asdict(self)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна. Установи PyTorch с CUDA или укажи --device cpu.")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                        else "cpu" if requested == "auto" else requested)


def files_for(split_dir, word):
    return sorted(f for f in (split_dir / word).glob("*.npy")
                  if not f.name.endswith(".tmp.npy"))


def available_classes(data):
    classes = [w for w in WORDS if files_for(data / "train", w)]
    if len(classes) < 2:
        raise ValueError(f"В {data / 'train'} меньше 2 классов. Сначала запусти extract_slovo.py.")
    missing = [w for w in WORDS if w not in classes]
    if missing:
        print(f"WARNING: ещё нет признаков для {len(missing)} слов: {', '.join(missing)}", flush=True)
    return classes


def load_data(split_dir, classes):
    sequences, labels, groups, paths = [], [], [], []
    for ci, word in enumerate(classes):
        for path in files_for(split_dir, word):
            group, separator, attachment = path.stem.partition("_")
            if not separator or not group or not attachment:
                raise ValueError(f"Ожидается имя user_id_attachment_id.npy: {path}")
            seq = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
            if seq.ndim != 2 or seq.shape[1] != FEAT_DIM or len(seq) == 0 or not np.isfinite(seq).all():
                raise ValueError(f"Повреждённые признаки: {path}, shape={seq.shape}")
            sequences.append(seq)
            labels.append(ci)
            groups.append(group)
            paths.append(path)
    if not paths:
        raise ValueError(f"Нет признаков в {split_dir}")
    return sequences, np.asarray(labels, dtype=np.int64), np.asarray(groups), paths


def grouped_splits(labels, groups, n_splits, seed):
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:
        raise ValueError("Установи зависимости: pip install -r requirements-training.txt") from exc
    if n_splits < 2 or len(np.unique(groups)) < n_splits:
        raise ValueError("Для CV нужны >= 2 фолдов и не меньше уникальных людей, чем фолдов")
    classes = np.unique(labels)
    for label in classes:
        if len(np.unique(groups[labels == label])) < 2:
            raise ValueError(f"Класс {label}: нужен минимум 2 разных человека для групповой валидации")
    # A grouped stratifier is approximate. Reject a split missing a training class.
    # Deterministically try several seeds; selection uses labels/groups, never model scores.
    for offset in range(100):
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + offset)
        splits = list(splitter.split(np.zeros(len(labels)), labels, groups))
        if all(np.array_equal(np.unique(labels[train]), classes) for train, _ in splits):
            return splits
    raise ValueError("Не удалось сохранить все классы в обучении. Уменьши --folds или добавь людей.")


def mirror_sequence(seq):
    seq = seq.copy()
    xyz = seq[:, :N_XYZ].reshape(len(seq), -1, 3).copy()
    xyz[..., 0] *= -1
    for a, b in POSE_PAIRS:
        xyz[:, [a, b]] = xyz[:, [b, a]]
    for left, right in ((PT_LH, PT_RH), (PT_LH_LOCAL, PT_RH_LOCAL)):
        tmp = xyz[:, left].copy()
        xyz[:, left] = xyz[:, right]
        xyz[:, right] = tmp
    seq[:, :N_XYZ] = xyz.reshape(len(seq), -1)
    seq[:, [FLAG_LH, FLAG_RH]] = seq[:, [FLAG_RH, FLAG_LH]]
    return seq


def augment(seq, allow_mirror=True):
    seq = seq.copy()
    if allow_mirror and random.random() < 0.5:
        seq = mirror_sequence(seq)
    n = len(seq)
    length = random.randint(max(1, int(n * 0.8)), n)
    start = random.randint(0, n - length)
    seq = seq[start:start + length].copy()
    xyz = seq[:, :N_XYZ].reshape(len(seq), -1, 3)
    present = (xyz != 0).any(-1, keepdims=True)
    angle = np.deg2rad(random.uniform(-12, 12))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)
    xyz[..., :2] = xyz[..., :2] @ rotation.T
    xyz *= random.uniform(0.9, 1.1)
    xyz += np.random.normal(0, 0.02, xyz.shape).astype(np.float32) * present
    seq[:, :N_XYZ] = xyz.reshape(len(seq), -1)
    if random.random() < 0.3:
        mask = np.random.rand(len(seq)) < 0.2
        blocks = (SL_LH, SL_LH_LOCAL, FLAG_LH) if random.random() < 0.5 else (SL_RH, SL_RH_LOCAL, FLAG_RH)
        for block in blocks:
            seq[mask, block] = 0
    return seq


class Clips(Dataset):
    def __init__(self, sequences, labels, indices, config, training):
        self.sequences, self.labels, self.indices = sequences, labels, indices
        self.config, self.training = config, training

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        seq = self.sequences[i]
        if self.training and self.config.augmentation:
            seq = augment(seq, self.config.mirror)
        return torch.from_numpy(make_input(seq, self.config.seq_len)), int(self.labels[i])


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=precision + recall > 0)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        # Fixed full vocabulary: an absent validation class contributes zero.
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()) if support.any() else 0.0,
        "per_class": [{"precision": float(p), "recall": float(r), "f1": float(f), "support": int(n)}
                      for p, r, f, n in zip(precision, recall, f1, support)],
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, n_classes, device):
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for x, y in loader:
        logits = model(x.to(device, non_blocking=device.type == "cuda"))
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite validation logits")
        predictions = logits.argmax(1).cpu().numpy()
        np.add.at(cm, (y.numpy(), predictions), 1)
    return metrics_from_cm(cm)


def classification_loss(logits, targets, config, weights=None):
    losses = F.cross_entropy(logits, targets, weight=weights, reduction="none",
                             label_smoothing=config.label_smoothing)
    if config.loss == "focal":
        probability = logits.softmax(-1).gather(1, targets[:, None]).squeeze(1)
        losses = losses * (1 - probability).pow(config.focal_gamma)
    return losses.mean()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, model, config, classes, epoch, seed):
    checkpoint = {
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "classes": list(classes), "in_dim": FEAT_DIM * 2,
        "hidden": config.hidden, "seq_len": config.seq_len, "model_config": config.to_dict(),
        "epoch": epoch, "seed": seed,
    }
    temporary = path.with_suffix(".tmp.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def fit(sequences, labels, train_indices, val_indices, classes, config, run_dir,
        device, epochs=100, patience=15, seed=0, num_workers=0, amp=False, min_delta=1e-4):
    """Without validation, train for a fixed epoch budget (final refit only)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    use_cuda = device.type == "cuda"
    training = Clips(sequences, labels, train_indices, config, True)
    counts = np.bincount(labels[train_indices], minlength=len(classes))
    if (counts == 0).any():
        raise ValueError("В обучающем фолде отсутствуют классы")
    sampler = None
    weights = None
    if config.balance == "sqrt":
        weights = torch.tensor(np.sqrt(counts.sum() / counts), dtype=torch.float32, device=device)
        weights /= weights.mean()
    elif config.balance == "sampler":
        sample_weights = 1.0 / counts[labels[train_indices]]
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(training))
    train_loader = DataLoader(training, batch_size=config.batch_size, shuffle=sampler is None,
                              sampler=sampler, pin_memory=use_cuda, num_workers=num_workers)
    val_loader = None
    if val_indices is not None:
        val_loader = DataLoader(Clips(sequences, labels, val_indices, config, False),
                                batch_size=config.batch_size * 2, pin_memory=use_cuda, num_workers=num_workers)
    model = build_model(config.to_dict(), FEAT_DIM * 2, len(classes)).to(device)
    print(f"{run_dir.name}: {config.arch}, parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr, total_steps=epochs * len(train_loader), pct_start=0.3)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = amp and use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score, best_epoch, stale, best_metrics = -1.0, 0, 0, None
    stopping_reference = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, total = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=use_cuda), y.to(device, non_blocking=use_cuda)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if config.mixup > 0:
                    lam = float(np.random.beta(config.mixup, config.mixup))
                    permutation = torch.randperm(len(x), device=device)
                    logits = model(lam * x + (1 - lam) * x[permutation])
                    loss = (lam * classification_loss(logits, y, config, weights)
                            + (1 - lam) * classification_loss(logits, y[permutation], config, weights))
                else:
                    loss = classification_loss(model(x), y, config, weights)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; lower lr or disable --amp")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not use_amp)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if config.scheduler == "onecycle" and scaler.get_scale() >= previous_scale:
                scheduler.step()
            loss_sum += float(loss.detach()) * len(y)
            total += len(y)
        if config.scheduler == "cosine":
            scheduler.step()
        row = {"epoch": epoch, "train_loss": loss_sum / total, "lr": optimizer.param_groups[0]["lr"]}
        if val_loader is not None:
            metrics = evaluate(model, val_loader, len(classes), device)
            row.update({k: metrics[k] for k in ("accuracy", "macro_f1", "balanced_accuracy")})
            score = metrics["macro_f1"]
            if score > best_score:
                best_score, best_epoch, best_metrics = score, epoch, metrics
                save_checkpoint(run_dir / "best.pt", model, config, classes, epoch, seed)
            if score > stopping_reference + min_delta:
                stopping_reference, stale = score, 0
            else:
                stale += 1
            print(f"epoch {epoch:3d} loss={row['train_loss']:.4f} "
                  f"val_f1={score:.4f} val_acc={metrics['accuracy']:.4f} best_f1={best_score:.4f}", flush=True)
        else:
            best_epoch = epoch
            print(f"epoch {epoch:3d} loss={row['train_loss']:.4f}", flush=True)
        history.append(row)
        save_json(run_dir / "history.json", history)
        if val_loader is not None and patience and stale >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break
    if val_loader is None:
        save_checkpoint(run_dir / "best.pt", model, config, classes, best_epoch, seed)
    result = {"best_epoch": best_epoch, "epochs_completed": epoch, "metrics": best_metrics,
              "config": config.to_dict(), "seed": seed}
    save_json(run_dir / "result.json", result)
    return result


def split_manifest(paths, labels, groups, splits, classes):
    return {
        "classes": list(classes),
        "samples": [{"path": str(p.resolve()), "label": int(y), "user_id": str(g)}
                    for p, y, g in zip(paths, labels, groups)],
        "folds": [{"train": train.tolist(), "validation": val.tolist(),
                   "missing_validation_classes": [classes[i] for i in range(len(classes))
                                                   if i not in set(labels[val].tolist())]}
                  for train, val in splits],
    }


def test_checkpoint(checkpoint_path, data_dir, output_path, device, batch_size=64):
    """Explicit final test; never invoked during hyperparameter selection."""
    from model import model_from_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = Config.from_dict(checkpoint.get("model_config", {"hidden": checkpoint["hidden"]}))
    classes = checkpoint["classes"]
    sequences, labels, groups, _ = load_data(data_dir / "test", classes)
    _, _, train_groups, _ = load_data(data_dir / "train", classes)
    counts = np.bincount(labels, minlength=len(classes))
    if (counts == 0).any():
        raise ValueError("В test отсутствуют классы модели; итоговая оценка будет неполной")
    loader = DataLoader(Clips(sequences, labels, np.arange(len(labels)), config, False), batch_size=batch_size)
    model = model_from_checkpoint(checkpoint).to(device)
    result = evaluate(model, loader, len(classes), device)
    result["classes"] = list(classes)
    result["train_test_user_overlap"] = len(set(groups) & set(train_groups))
    result["checkpoint"] = str(Path(checkpoint_path).resolve())
    save_json(output_path, result)
    print(f"TEST accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f}; "
          f"overlapping people={result['train_test_user_overlap']}", flush=True)
    return result


def write_leaderboard(path, trials):
    fields = ["trial", "arch", "mean_macro_f1", "std_macro_f1", "mean_accuracy", "config"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trial in sorted(trials, key=lambda item: item["mean_macro_f1"], reverse=True):
            writer.writerow({k: json.dumps(trial[k], ensure_ascii=False) if k == "config" else trial[k]
                             for k in fields})
