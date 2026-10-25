"""Explicit final evaluation of a saved model on the official Slovo test split."""
import argparse
from pathlib import Path

from features import KP_DIR
from training import select_device, test_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=KP_DIR)
    parser.add_argument("--out", type=Path, help="JSON report (default: beside checkpoint)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    try:
        destination = args.out or args.model.parent / "test_metrics.json"
        test_checkpoint(args.model, args.data, destination, select_device(args.device))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
