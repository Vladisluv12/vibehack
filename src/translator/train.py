"""Обучение классификатора жестов на точках Slovo (сначала запусти extract_slovo.py).

train — data/slovo_kp/train, валидация — data/slovo_kp/test (официальное разбиение Slovo).

python train.py
python train.py --epochs 120 --hidden 256
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from features import (WORDS, KP_DIR, MODEL_PATH, make_input, N_XYZ, SL_LH, SL_RH, SL_LH_LOCAL,
                      SL_RH_LOCAL, FLAG_LH, FLAG_RH, PT_LH, PT_RH, PT_LH_LOCAL, PT_RH_LOCAL,
                      POSE_PAIRS, SEQ_LEN, FEAT_DIM)
from model import SignGRU


def load(split_dir, classes):
    X, y = [], []
    for ci, w in enumerate(classes):
        for f in sorted((split_dir / w).glob("*.npy")):
            if f.name.endswith(".tmp.npy"):
                continue
            X.append(np.load(f))
            y.append(ci)
    return X, np.array(y)


def mirror(seq):
    """Отражение по горизонтали: левша <-> правша, зеркальная фронталка телефона."""
    seq = seq.copy()
    xyz = seq[:, :N_XYZ].reshape(len(seq), -1, 3).copy()
    xyz[..., 0] *= -1
    for a, b in POSE_PAIRS:
        xyz[:, [a, b]] = xyz[:, [b, a]]
    for A, B in ((PT_LH, PT_RH), (PT_LH_LOCAL, PT_RH_LOCAL)):
        tmp = xyz[:, A].copy()
        xyz[:, A] = xyz[:, B]
        xyz[:, B] = tmp
    seq[:, :N_XYZ] = xyz.reshape(len(seq), -1)
    seq[:, [FLAG_LH, FLAG_RH]] = seq[:, [FLAG_RH, FLAG_LH]]
    return seq


def augment(seq):
    seq = seq.copy()
    if random.random() < 0.5:
        seq = mirror(seq)

    # время: случайный кусок 80–100% клипа (жест чуть раньше/позже, быстрее/медленнее)
    n = len(seq)
    L = random.randint(max(2, int(n * 0.8)), n)
    s = random.randint(0, n - L)
    seq = seq[s:s + L]

    xyz = seq[:, :N_XYZ].reshape(len(seq), -1, 3)
    present = (xyz != 0).any(-1, keepdims=True)          # не трогаем отсутствующие точки
    a = np.deg2rad(random.uniform(-12, 12))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    xyz[..., :2] = xyz[..., :2] @ R.T                    # наклон камеры / человека
    xyz *= random.uniform(0.9, 1.1)                      # другой масштаб
    xyz += np.random.normal(0, 0.02, xyz.shape).astype(np.float32) * present  # дрожание трекинга
    seq[:, :N_XYZ] = xyz.reshape(len(seq), -1)

    # MediaPipe иногда теряет руку на нескольких кадрах — имитируем
    if random.random() < 0.3:
        m = np.random.rand(len(seq)) < 0.2
        if random.random() < 0.5:
            seq[m, SL_LH], seq[m, SL_LH_LOCAL], seq[m, FLAG_LH] = 0, 0, 0
        else:
            seq[m, SL_RH], seq[m, SL_RH_LOCAL], seq[m, FLAG_RH] = 0, 0, 0
    return seq


class Clips(Dataset):
    def __init__(self, X, y, aug):
        self.X, self.y, self.aug = X, y, aug

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        seq = augment(self.X[i]) if self.aug else self.X[i]
        return torch.from_numpy(make_input(seq)), int(self.y[i])


@torch.no_grad()
def evaluate(model, loader, n_cls):
    model.eval()
    cm = np.zeros((n_cls, n_cls), int)
    for x, y in loader:
        pred = model(x).argmax(1)
        for t, p in zip(y.tolist(), pred.tolist()):
            cm[t, p] += 1
    return cm.trace() / max(cm.sum(), 1), cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=KP_DIR)
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=192)
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    classes = [w for w in WORDS if any((args.data / "train" / w).glob("*.npy"))]
    if len(classes) < 2:
        raise SystemExit(f"В {args.data / 'train'} меньше 2 классов — сначала запусти extract_slovo.py")
    Xtr, ytr = load(args.data / "train", classes)
    Xva, yva = load(args.data / "test", classes)
    print("клипов в train по классам:", {c: int((ytr == i).sum()) for i, c in enumerate(classes)})
    print(f"train: {len(ytr)}  val (test): {len(yva)}")

    train_dl = DataLoader(Clips(Xtr, ytr, True), batch_size=32, shuffle=True)
    val_dl = DataLoader(Clips(Xva, yva, False), batch_size=64)

    # no_event в Slovo в 20 раз больше, чем любого слова — уравновешиваем вес классов
    counts = np.bincount(ytr, minlength=len(classes)).astype(np.float32)
    weights = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float32)
    weights /= weights.mean()

    in_dim = FEAT_DIM * 2
    model = SignGRU(in_dim, len(classes), args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, t in train_dl:
            logits = model(x)
            loss = loss_fn(logits, t)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(t)
            correct += (logits.argmax(1) == t).sum().item(); total += len(t)
        sched.step()

        val_acc, _ = evaluate(model, val_dl, len(classes)) if len(yva) else (correct / total, None)
        if val_acc >= best:
            best = val_acc
            torch.save({"state": model.state_dict(), "classes": classes, "in_dim": in_dim,
                        "hidden": args.hidden, "seq_len": SEQ_LEN}, args.out)
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:3d}  loss {loss_sum / total:.3f}  train_acc {correct / total:.3f}  "
                  f"val_acc {val_acc:.3f}  best {best:.3f}")

    if len(yva):
        ckpt = torch.load(args.out, map_location="cpu")
        model.load_state_dict(ckpt["state"])
        acc, cm = evaluate(model, val_dl, len(classes))
        print(f"\nлучшая val accuracy: {acc:.3f}")
        print("точность по классам:")
        for i, (c, row) in enumerate(zip(classes, cm)):
            errors = row.copy(); errors[i] = 0
            confused = f"   чаще всего путает с: {classes[int(errors.argmax())]}" if errors.any() else ""
            print(f"  {c:>18}: {row[i]}/{row.sum()}{confused}")
    print(f"\nмодель сохранена в {args.out}")


if __name__ == "__main__":
    main()
