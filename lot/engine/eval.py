# lot/engine/eval.py
import json
import pathlib
import torch
from torch.utils.data import DataLoader
from lot.model.transformer import Transformer


def evaluate(checkpoint_path: pathlib.Path,
              data: DataLoader,
              output_dir: pathlib.Path) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # 2. model already exists in checkpoint
    model = ckpt["hyper_parameters"]["model"]   # Transformer object
    state = {k.partition("model.")[2]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    # 4. evaluate
    total, em_total = 0, 0
    syncs = []
    ce_sum, ce_count = 0.0, 0
    for x, y in data:
        x, y = x.to(device), y.to(device)
        logits, info = model(x)
        preds = logits.argmax(-1)
        mask = y.ne(-100)
        em_total += ((preds.eq(y) & mask).all(dim=-1)).sum().item()
        total += x.size(0)

        # cross-entropy (only on valid tokens)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)),
                            y.view(-1),
                            ignore_index=-100,
                            reduction='sum')
        ce_sum += ce.item()
        ce_count += mask.sum().item()

        # sync order (only for lattice runs)
        if info["layers"][0].get("phase") is not None:
            phases = torch.stack([info["layers"][l]["phase"] for l in range(len(info["layers"]))])
            syncs.append(torch.abs(torch.exp(1j * phases).mean(1)).mean().item())
        else:
            syncs.append(0.0)

    metrics = {"exact_match": em_total / total,
               "mean_sync_order": float(torch.tensor(syncs).mean()),
               "ce_loss": ce_sum / max(ce_count, 1)    # per-token average
    }
    print(json.dumps(metrics, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics))
