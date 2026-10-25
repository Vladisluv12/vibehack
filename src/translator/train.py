"""Train one configuration with signer-disjoint validation, or refit on all train data."""
import argparse
import json
import shutil
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np

from features import KP_DIR, PROJECT
from training import (Config, available_classes, fit, grouped_splits, load_data,
                      save_json, select_device, split_manifest, test_checkpoint)


def add_config_arguments(parser):
    for field in fields(Config):
        flag = "--" + field.name.replace("_", "-")
        if field.type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(flag, type=field.type, default=None)


def config_from_args(args):
    values = {}
    if args.config is not None:
        values = json.loads(args.config.read_text(encoding="utf-8"))
    values.update({f.name: getattr(args, f.name) for f in fields(Config)
                   if getattr(args, f.name) is not None})
    return Config.from_dict(values)


def validate_run_options(args):
    if args.epochs < 1 or args.patience < 0 or args.num_workers < 0 or args.seed < 0:
        raise ValueError("epochs >= 1; patience, num_workers, seed >= 0")
    if not np.isfinite(args.min_delta) or args.min_delta < 0:
        raise ValueError("min_delta must be >= 0")


def add_run_arguments(parser):
    parser.add_argument("--data", type=Path, default=KP_DIR)
    parser.add_argument("--config", type=Path, help="JSON configuration, e.g. best_config.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15, help="0 disables early stopping")
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="CUDA mixed precision")
    parser.add_argument("--refit", action="store_true", help="Train on ALL train for a fixed epoch budget")
    parser.add_argument("--test", action="store_true", help="Explicitly evaluate official test once afterwards")


def create_run_dir(requested, prefix):
    directory = requested or PROJECT / "runs" / (prefix + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    add_config_arguments(parser)
    parser.add_argument("--fold", type=int, default=0, help="Validation fold for a single run")
    parser.add_argument("--out", type=Path, help="Optional extra copy of the checkpoint; existing files are protected")
    args = parser.parse_args()
    try:
        validate_run_options(args)
        config = config_from_args(args)
        device = select_device(args.device)
        if args.out is not None and args.out.exists():
            raise ValueError(f"--out already exists: {args.out}. Use a new filename.")
        classes = available_classes(args.data)
        sequences, labels, groups, paths = load_data(args.data / "train", classes)
        splits = [] if args.refit else grouped_splits(labels, groups, args.folds, args.seed)
        if args.refit:
            train_indices, val_indices = np.arange(len(labels)), None
        else:
            if not 0 <= args.fold < len(splits):
                raise ValueError("--fold must be in [0, folds)")
            train_indices, val_indices = splits[args.fold]
        run_dir = create_run_dir(args.run_dir, config.arch)
        save_json(run_dir / "run.json", {**vars_as_json(args), "config": config.to_dict()})
        manifest = split_manifest(paths, labels, groups, splits, classes)
        save_json(run_dir / "splits.json", manifest)
        if splits:
            missing = manifest["folds"][args.fold]["missing_validation_classes"]
            print(f"Validation: {len(val_indices)} clips, missing classes: {missing}", flush=True)
        print(f"Run: {run_dir}\nDevice: {device}; train clips: {len(train_indices)}; classes: {len(classes)}", flush=True)
        result = fit(sequences, labels, train_indices, val_indices, classes, config, run_dir, device,
                     epochs=args.epochs, patience=args.patience, seed=args.seed,
                     num_workers=args.num_workers, amp=args.amp, min_delta=args.min_delta)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / "best.pt", args.out)
        if args.test:
            test_checkpoint(run_dir / "best.pt", args.data, run_dir / "test_metrics.json", device)
        print(f"Checkpoint: {run_dir / 'best.pt'}; best epoch: {result['best_epoch']}", flush=True)
    except (ValueError, FileExistsError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


def vars_as_json(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


if __name__ == "__main__":
    main()
