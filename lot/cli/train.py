#!/usr/bin/env python3
import pathlib, pytorch_lightning as pl, torch
import yaml
from lot.engine.train import LotModule
from lot.model.transformer import Transformer
from lot.tasks.copy_reverse import CopyReverseDataset
from torch.utils.data import DataLoader
from jsonargparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    pl.seed_everything(cfg["seed"])
    model = Transformer(**cfg["model"])
    ds = CopyReverseDataset(**cfg["data"])
    train_loader = DataLoader(ds, batch_size=cfg["train_batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(ds, batch_size=cfg["val_batch_size"], shuffle=False)

    module = LotModule(model, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                       lambda_sync=cfg["lambda_sync"], lambda_diversity=cfg["lambda_diversity"])

    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=pathlib.Path("runs") / args.config.stem,
        logger=pl.loggers.TensorBoardLogger("runs/", name=args.config.stem),
    )
    trainer.fit(module, train_loader, val_loader)

if __name__ == "__main__":
    main()