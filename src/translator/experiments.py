"""Compare configurations on identical signer-disjoint CV folds; optionally refit the winner."""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from train import add_run_arguments, create_run_dir, validate_run_options, vars_as_json
from training import (Config, available_classes, fit, grouped_splits, load_data, save_json,
                      select_device, split_manifest, test_checkpoint, write_leaderboard)

DEFAULT_SPACE = {
    "arch": ["gru", "attn_gru", "transformer"],
    "hidden": [128, 192, 256], "layers": [1, 2, 3], "dropout": [0.2, 0.35, 0.5],
    "lr": [0.0003, 0.001, 0.003], "weight_decay": [0.0001, 0.01],
    "batch_size": [16, 32], "seq_len": [32, 64], "balance": ["sqrt", "sampler", "none"],
    "label_smoothing": [0.0, 0.1], "loss": ["ce", "focal"], "mixup": [0.0, 0.2],
    "scheduler": ["cosine", "onecycle"],
}


def candidate_configs(space, method, trials, seed, base=None):
    try:
        from sklearn.model_selection import ParameterSampler
    except ImportError as exc:
        raise ValueError("Установи зависимости: pip install -r requirements-training.txt") from exc
    if not isinstance(space, dict) or not space:
        raise ValueError("--space must contain a non-empty JSON object of lists")
    for key, values in space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Space entry {key} must be a non-empty list")
        for value in values:
            Config.from_dict({**(base or {}), key: value})
    keys = list(space)
    total = int(np.prod([len(space[key]) for key in keys], dtype=object))
    if method == "grid":
        if total > trials:
            raise ValueError(f"Grid has {total} combinations; increase --trials to >= {total}, or narrow --space")
        candidates = [dict(zip(keys, values)) for values in product(*(space[key] for key in keys))]
    else:
        candidates = list(ParameterSampler(space, n_iter=min(trials, total), random_state=seed))
    configs = [Config.from_dict({**(base or {}), **candidate}) for candidate in candidates]
    # With the built-in space, compare all architectures under equal starting settings first.
    if space == DEFAULT_SPACE and method == "random":
        baselines = [Config.from_dict({**(base or {}), "arch": arch}) for arch in DEFAULT_SPACE["arch"]]
        configs = baselines[:trials] + configs
    unique = []
    seen = set()
    for config in configs:
        key = json.dumps(config.to_dict(), sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(config)
        if len(unique) == trials:
            break
    return unique


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--space", type=Path, help="JSON object of parameter lists; overrides built-in space")
    parser.add_argument("--method", choices=("random", "grid"), default="random")
    parser.add_argument("--trials", type=int, default=12, help="Number of configs; also a safety cap for a full grid")
    parser.add_argument("--dry-run", action="store_true", help="Save plan/splits without training")
    args = parser.parse_args()
    try:
        validate_run_options(args)
        if args.trials < 1:
            raise ValueError("--trials must be >= 1")
        if args.test and not args.refit:
            raise ValueError("--test requires --refit; choose the winner using CV first")
        device = select_device(args.device)
        base = {} if args.config is None else json.loads(args.config.read_text(encoding="utf-8"))
        Config.from_dict(base)
        space = DEFAULT_SPACE if args.space is None else json.loads(args.space.read_text(encoding="utf-8"))
        configs = candidate_configs(space, args.method, args.trials, args.seed, base)
        classes = available_classes(args.data)
        sequences, labels, groups, paths = load_data(args.data / "train", classes)
        splits = grouped_splits(labels, groups, args.folds, args.seed)
        run_dir = create_run_dir(args.run_dir, "search")
        manifest = split_manifest(paths, labels, groups, splits, classes)
        save_json(run_dir / "splits.json", manifest)
        save_json(run_dir / "plan.json", {
            **vars_as_json(args), "configs": [c.to_dict() for c in configs],
            "training_jobs": len(configs) * len(splits), "classes": classes,
        })
        print(f"Run: {run_dir}; configs={len(configs)}, folds={len(splits)}, "
              f"training jobs={len(configs) * len(splits)}; device={device}", flush=True)
        for i, split in enumerate(manifest["folds"]):
            print(f"Fold {i}: train={len(split['train'])}, validation={len(split['validation'])}, "
                  f"missing validation classes={split['missing_validation_classes']}", flush=True)
        if args.dry_run:
            print("Plan saved; no models trained and official test not opened.", flush=True)
            return
        trials = []
        for trial_index, config in enumerate(configs):
            trial_dir = run_dir / f"trial_{trial_index:03d}"
            results = []
            for fold, (train_indices, val_indices) in enumerate(splits):
                print(f"\nTrial {trial_index + 1}/{len(configs)}, fold {fold + 1}/{len(splits)}: "
                      f"{json.dumps(config.to_dict())}", flush=True)
                result = fit(
                    sequences, labels, train_indices, val_indices, classes, config,
                    trial_dir / f"fold_{fold:02d}", device, epochs=args.epochs, patience=args.patience,
                    # Match fold seeds across candidates; reset model and optimizer in every fit.
                    seed=args.seed + fold, num_workers=args.num_workers, amp=args.amp,
                    min_delta=args.min_delta)
                results.append(result)
                save_json(trial_dir / "fold_results.json", results)
            scores = [r["metrics"]["macro_f1"] for r in results]
            # Pooled out-of-fold confusion matrix also exposes absent-class fold effects.
            pooled = np.sum([r["metrics"]["confusion_matrix"] for r in results], axis=0)
            from training import metrics_from_cm
            trial = {
                "trial": trial_index, "arch": config.arch, "config": config.to_dict(),
                "mean_macro_f1": float(np.mean(scores)), "std_macro_f1": float(np.std(scores)),
                "mean_accuracy": float(np.mean([r["metrics"]["accuracy"] for r in results])),
                "folds": results, "out_of_fold_metrics": metrics_from_cm(pooled),
                "refit_epochs": max(1, int(np.median([r["best_epoch"] for r in results]))),
            }
            trials.append(trial)
            save_json(run_dir / "results.json", trials)
            write_leaderboard(run_dir / "leaderboard.csv", trials)
            winner = max(trials, key=lambda item: item["mean_macro_f1"])
            save_json(run_dir / "best_config.json", winner["config"])
            save_json(run_dir / "winner.json", winner)
            print(f"CV macro-F1={trial['mean_macro_f1']:.4f} +/- {trial['std_macro_f1']:.4f}", flush=True)
        winner = max(trials, key=lambda item: item["mean_macro_f1"])
        print(f"Best: trial {winner['trial']}, macro-F1={winner['mean_macro_f1']:.4f}", flush=True)
        if args.refit:
            final_dir = run_dir / "final"
            fit(sequences, labels, np.arange(len(labels)), None, classes,
                Config.from_dict(winner["config"]), final_dir, device, epochs=winner["refit_epochs"],
                seed=args.seed, num_workers=args.num_workers, amp=args.amp)
            if args.test:
                test_checkpoint(final_dir / "best.pt", args.data, final_dir / "test_metrics.json", device)
            print(f"Final checkpoint: {final_dir / 'best.pt'}", flush=True)
        print(f"Results: {run_dir / 'leaderboard.csv'}", flush=True)
    except (ValueError, FileExistsError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
