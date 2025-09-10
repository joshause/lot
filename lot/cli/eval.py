#!/usr/bin/env python3
"""
CLI entry-point for stand-alone evaluation.
Usage:
    lot-eval <checkpoint.pt> --config experiments/xxx.yaml
"""
import argparse
import pathlib

from torch.utils.data import DataLoader
from lot.engine.eval import evaluate
from lot.tasks.copy_reverse import CopyReverseDataset
import yaml

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a LoT checkpoint")
    parser.add_argument("checkpoint", type=pathlib.Path, help="Lightning .pt checkpoint")
    parser.add_argument("--config", type=pathlib.Path, required=True, help="Training YAML")
    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ds = CopyReverseDataset(**cfg["data"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # run evaluation
    evaluate(args.checkpoint, dl, args.checkpoint.parent)

if __name__ == "__main__":
    main()