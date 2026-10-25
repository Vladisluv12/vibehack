"""Run with: python -m unittest discover -s tests -v"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "translator"))
from features import FEAT_DIM, N_XYZ, PROJECT, make_input
from model import ARCHITECTURES, SignGRU, build_model, model_from_checkpoint
from training import (Clips, Config, augment, classification_loss, evaluate, fit,
                      grouped_splits, load_data, metrics_from_cm, mirror_sequence)
from experiments import candidate_configs


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_models_backward_and_checkpoint_roundtrip(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                config = Config(arch=architecture, hidden=16, layers=2, heads=4)
                model = build_model(config.to_dict(), FEAT_DIM * 2, 3)
                x = torch.randn(2, 8, FEAT_DIM * 2)
                output = model(x)
                self.assertEqual(output.shape, (2, 3))
                classification_loss(output, torch.tensor([0, 2]), config).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters() if p.requires_grad))
                checkpoint = {"state": model.state_dict(), "in_dim": FEAT_DIM * 2,
                              "hidden": 16, "classes": ["a", "b", "c"], "model_config": config.to_dict()}
                model.eval()
                restored = model_from_checkpoint(checkpoint).eval()
                torch.testing.assert_close(model(x), restored(x))

    def test_transformer_layers_have_independent_initial_weights(self):
        model = build_model(Config(arch="transformer", hidden=16).to_dict(), FEAT_DIM * 2, 3)
        self.assertFalse(torch.equal(model.encoder.layers[0].linear1.weight,
                                     model.encoder.layers[1].linear1.weight))

    def test_old_checkpoint_format(self):
        original = SignGRU(FEAT_DIM * 2, 3, hidden=16).eval()
        checkpoint = {"state": original.state_dict(), "classes": ["a", "b", "c"],
                      "in_dim": FEAT_DIM * 2, "hidden": 16, "seq_len": 32}
        restored = model_from_checkpoint(checkpoint).eval()
        x = torch.randn(2, 32, FEAT_DIM * 2)
        torch.testing.assert_close(original(x), restored(x))
        real = PROJECT / "models" / "sign_gru.pt"
        if real.exists():
            model_from_checkpoint(torch.load(real, map_location="cpu", weights_only=True))

    def test_group_splits_are_disjoint_and_repeatable(self):
        y = np.tile(np.arange(3), 12)
        groups = np.repeat(np.arange(6).astype(str), 6)
        splits = grouped_splits(y, groups, 3, 17)
        again = grouped_splits(y, groups, 3, 17)
        self.assertEqual(sorted(np.concatenate([v for _, v in splits]).tolist()), list(range(len(y))))
        for (train, val), (train2, val2) in zip(splits, again):
            self.assertFalse(set(groups[train]) & set(groups[val]))
            self.assertEqual(set(y[train]), {0, 1, 2})
            np.testing.assert_array_equal(train, train2)
            np.testing.assert_array_equal(val, val2)
        with self.assertRaises(ValueError):
            grouped_splits(np.array([0, 0, 1, 1]), np.array(["a", "a", "b", "c"]), 2, 0)

    def test_single_frame_augmentation_and_mirror(self):
        seq = np.random.default_rng(0).normal(size=(1, FEAT_DIM)).astype(np.float32)
        seq[:, 75:138] = 0
        seq[:, 201:264] = 0
        np.testing.assert_allclose(mirror_sequence(mirror_sequence(seq)), seq)
        for _ in range(20):
            result = augment(seq, allow_mirror=False)
            self.assertTrue(np.isfinite(result).all())
            np.testing.assert_array_equal(result[:, 75:138], 0)
            self.assertEqual(make_input(result, 64).shape, (64, FEAT_DIM * 2))
        np.testing.assert_array_equal(make_input(seq)[:, FEAT_DIM:], 0)

    def test_metrics_penalize_majority_only_classifier(self):
        metrics = metrics_from_cm(np.array([[100, 0], [10, 0]]))
        self.assertGreater(metrics["accuracy"], 0.9)
        self.assertLess(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)

    def test_focal_gamma_zero_matches_cross_entropy(self):
        logits = torch.randn(5, 3)
        targets = torch.tensor([0, 1, 2, 0, 1])
        torch.testing.assert_close(classification_loss(logits, targets, Config(loss="ce")),
                                   classification_loss(logits, targets, Config(loss="focal", focal_gamma=0)))

    def test_temporary_and_invalid_features(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "word"
            folder.mkdir()
            np.save(folder / "user_clip.npy", np.zeros((1, FEAT_DIM), np.float32))
            np.save(folder / "user_bad.tmp.npy", np.zeros(2))
            sequences, labels, groups, paths = load_data(Path(directory), ["word"])
            self.assertEqual(len(sequences), 1)
            self.assertEqual(groups.tolist(), ["user"])
            np.save(folder / "user_clip.npy", np.zeros((0, FEAT_DIM), np.float32))
            with self.assertRaises(ValueError):
                load_data(Path(directory), ["word"])

    def test_small_end_to_end_cv_refit_all_approaches(self):
        rng = np.random.default_rng(0)
        sequences = [rng.normal(size=(5, FEAT_DIM)).astype(np.float32) for _ in range(12)]
        labels = np.tile([0, 1], 6)
        groups = np.repeat(np.arange(6).astype(str), 2)
        train, val = grouped_splits(labels, groups, 2, 0)[0]
        with tempfile.TemporaryDirectory() as directory:
            for arch, balance, loss, scheduler in (
                    ("gru", "sqrt", "ce", "cosine"),
                    ("attn_gru", "sampler", "focal", "onecycle"),
                    ("transformer", "none", "ce", "cosine")):
                config = Config(arch=arch, hidden=16, layers=1, seq_len=8,
                                batch_size=4, balance=balance, loss=loss, scheduler=scheduler, mixup=0.2)
                run = Path(directory) / arch
                result = fit(sequences, labels, train, val, ["a", "b"], config,
                             run, torch.device("cpu"), epochs=2, patience=1)
                self.assertGreaterEqual(result["metrics"]["macro_f1"], 0)
                checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
                loader = DataLoader(Clips(sequences, labels, val, config, False), batch_size=4)
                evaluated = evaluate(model_from_checkpoint(checkpoint), loader, 2, torch.device("cpu"))
                self.assertEqual(evaluated, result["metrics"])
                self.assertTrue(json.loads((run / "history.json").read_text()))
            fit(sequences, labels, np.arange(12), None, ["a", "b"], config,
                Path(directory) / "final", torch.device("cpu"), epochs=1)
            self.assertTrue((Path(directory) / "final" / "best.pt").is_file())

    def test_search_is_reproducible_and_grid_is_bounded(self):
        space = {"arch": ["gru", "transformer"], "hidden": [16, 32]}
        a = candidate_configs(space, "random", 3, 0)
        b = candidate_configs(space, "random", 3, 0)
        self.assertEqual(a, b)
        self.assertEqual(len(candidate_configs(space, "grid", 4, 0)), 4)
        with self.assertRaises(ValueError):
            candidate_configs(space, "grid", 3, 0)

    def test_search_and_evaluation_cli_on_synthetic_files(self):
        import os
        import subprocess
        rng = np.random.default_rng(42)
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in sys.path if p),
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            for word in ("privet", "poka"):
                (data / "train" / word).mkdir(parents=True)
                (data / "test" / word).mkdir(parents=True)
                for user in range(6):
                    np.save(data / "train" / word / f"user{user}_clip.npy",
                            rng.normal(size=(4, FEAT_DIM)).astype(np.float32))
                # A corrupt test file proves that search/refit never open official test.
                (data / "test" / word / "heldout_clip.npy").write_bytes(b"must not be read")
            space = root / "space.json"
            space.write_text(json.dumps({"arch": list(ARCHITECTURES), "hidden": [16],
                                         "layers": [1], "seq_len": [8], "batch_size": [4]}))
            search = root / "search"
            def run(script, *arguments):
                process = subprocess.run(
                    [sys.executable, str(PROJECT / "src" / "translator" / script), *map(str, arguments)],
                    env=environment, text=True, capture_output=True)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                return process
            run("experiments.py", "--space", space, "--method", "grid", "--trials", 3,
                "--folds", 2, "--epochs", 1, "--patience", 0, "--device", "cpu",
                "--data", data, "--run-dir", search, "--refit")
            results = json.loads((search / "results.json").read_text())
            self.assertEqual(len(results), 3)
            self.assertTrue(all(len(trial["folds"]) == 2 for trial in results))
            self.assertTrue((search / "final" / "best.pt").exists())
            self.assertFalse((search / "final" / "test_metrics.json").exists())
            for word in ("privet", "poka"):
                np.save(data / "test" / word / "heldout_clip.npy", np.zeros((4, FEAT_DIM), np.float32))
            run("evaluate.py", "--model", search / "final" / "best.pt", "--data", data, "--device", "cpu")
            metrics = json.loads((search / "final" / "test_metrics.json").read_text())
            self.assertEqual(np.sum(metrics["confusion_matrix"]), 2)
            self.assertEqual(metrics["train_test_user_overlap"], 0)
            run("train.py", "--config", search / "best_config.json", "--data", data,
                "--device", "cpu", "--epochs", 1, "--folds", 2, "--run-dir", root / "single")
            self.assertTrue((root / "single" / "best.pt").is_file())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_mixed_precision_on_synthetic_clips(self):
        rng = np.random.default_rng(42)
        sequences = [rng.normal(size=(4, FEAT_DIM)).astype(np.float32) for _ in range(8)]
        labels = np.tile([0, 1], 4)
        with tempfile.TemporaryDirectory() as directory:
            for arch in ARCHITECTURES:
                config = Config(arch=arch, hidden=16, layers=1, batch_size=4, seq_len=8)
                result = fit(sequences, labels, np.arange(4), np.arange(4, 8),
                             ["a", "b"], config, Path(directory) / arch,
                             torch.device("cuda"), epochs=1, amp=True)
                self.assertTrue(np.isfinite(result["metrics"]["macro_f1"]))


if __name__ == "__main__":
    unittest.main()
