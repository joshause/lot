# Lattice Oscillator Transformer (LOT)

## 1. Introduction
**Lattice Oscillator Transformer (LOT)** is both a framework for investigating **synchronized phase attention** and a Kuramato phase attention Transformer proof of concept It is a fusion of ideas from condensed matter physics, neuroscience, and machine learning that incorporates the collective dynamics of **coupled oscillators** and **phase coherence** in a Transformer neural network architecture.

Standard Transformer heads are independent computational units.
LOT arranges heads on a 2-D lattice whose dynamics follow a [**Kuramoto oscillator model**](https://en.wikipedia.org/wiki/Kuramoto_model):

```
phaseᵢ(t+Δt) = phaseᵢ(t) + Δt [ ωᵢ + κᵢ Σⱼ wᵢⱼ sin(θⱼ - θᵢ) ]
attention score += α cos θᵢ
```
Each attention head acts as a coupled oscillator with **nearest-neighbor interactions**, *potentially* creating emergent collective behaviors that enhance the transformer's ability to capture long-range dependencies and hierarchical patterns.

In LOT, information travels as **coherent phase waves**, yielding:

- **Interpretable dynamics** - phases & sync-order are trivial to visualise.

- **Built-in relational inductive bias** - nearby heads coordinate, distant ones decouple.

- **Regularisation for free** - sync & diversity losses fall out of the physics.

The framework provides **parameter-matched vanilla baselines**, **statistical reporting**, and **zero-effort scaling** so you can rigorously test whether the oscillator mechanism **improves generalization** on your task.

## 2. Repository Map
```
lot/
├── lot/
│   ├── model/           # attention modules & transformer trunk
│   ├── tasks/           # canonical seq-to-seq tasks
│   ├── engine/          # Lightning trainers & evaluators
├── experiments/         # YAML configs (seed, model, training schedule)
├── tests/               # pytest unit & regression tests
├── scripts/             # CLI entry points
├── csrc/                # optional CUDA kernels (build with python setup.py build_ext)
```

## 3. Setup
Requirements: Python ≥ 3.9, PyTorch ≥ 2.1, pytorch-lightning ≥ 2.1
(works on CPU, but GPU recommended)

```
git clone https://github.com/joshause/lot.git
cd lot
python -m venv venv && source venv/bin/activate
pip install -e .[dev,test]          # editable install + dev tools
pytest tests/                       # sanity check (≈ 5 s)
```

Ensure that cli components are locally executable:

```
chmod +x lot/cli/train.py
chmod +x lot/cli/eval.py
chmod +x lot/cli/export.py
```

Build CUDA kernels (**OPTIONAL+UNTESTED, requires nvcc**):

```
python setup.py build_ext --inplace
```

## 4. Train & Evaluate & Export

### 4.1 Copy-Reverse benchmark (included)

#### Train:

Config: experiments/copy_reverse.yaml

Task: copy first half, emit delimiter, reverse first half.

Metric: **exact-match accuracy** (% perfectly recovered sequences).

```
# lattice run
lot-train --config experiments/copy_reverse.yaml

# vanilla baseline (same param count)
# edit yaml -> attn_type: vanilla
lot-train --config experiments/copy_reverse.yaml

# example of hyper-parameter overrides
lot-train --config experiments/copy_reverse.yaml \
          --lr 3e-4 \
          --lambda_sync 0.05
```

Logs & checkpoints land in runs/<config-name>/version_* (TensorBoard).
Best checkpoint is automatically symlinked to .../checkpoint.pt.

#### Evaluate:

```
lot-eval runs/version_{n}/checkpoints/{epoch+step}.ckpt \
        --config experiments/copy_reverse.yaml
```

Output (metrics.json):

```
{
  "exact_match": 0.987,
  "mean_sync_order": 0.843,
  "ce_loss": 0.052
}
```

#### Export

```
lot-export runs/cr_lattice/version_0/checkpoint.pt \
        --config experiments/copy_reverse.yaml \
        --format onnx \
        --output my_model.onnx
```

Produces my_model.onnx (or my_model.ts.pt) ready for deployment / inference servers.

### 4.2 Character-level language modelling (example/suggestion)
lot-train --config experiments/enwik8_char.yaml   # 90 M params, 1 epoch ≈ 4 h on 1×A100

Report bits-per-character (bpc) & sync dynamics.

## 5. CLI Reference
| Command | Purpose |
| -------- | ------- |
| lot-train --config path/to.yaml | Train from scratch (resumes if run-dir exists) |
| lot-eval checkpoint.pt --config yaml | Compute metrics on full test set |
| lot-export checkpoint.pt --onnx model.onnx | Export for deployment |

## 6. Configuration System

YAML files are self-contained experiments:

```
seed: 42
model:
  attn_type: lattice          # lattice | vanilla
  vocab_size: 1000
  max_seq_len: 128
  embed_dim: 256
  num_heads: 8
  num_layers: 6
  lattice_shape: [2, 4]       # must multiply to num_heads
  init_freq_range: [0.9, 1.1]
  dropout: 0.1

data:
  _target_: lot.tasks.copy_reverse.CopyReverseDataset
  vocab_size: 1000
  seq_len: 128
  n_samples: 50000

train_batch_size: 128
lr: 3e-4
weight_decay: 0.01
lambda_sync: 0.02          # κ_sync loss weight
lambda_diversity: 0.01     # frequency diversity loss
max_epochs: 50
```

Override at runtime:

```
lot-train --config experiments/copy_reverse.yaml \
          --lambda_sync 0.0   # ablate sync loss
```

## 7. Adding a New Task

1. Implement a torch.utils.data.Dataset (return (input_ids, target_ids)).

```
# lot/tasks/my_task.py
class MyDataset(Dataset):
    def __init__(self, ...): ...
    def __getitem__(self, idx): ...
```

2. Create YAML:

```
data:
  _target_: lot.tasks.my_task.MyDataset
  my_param: 42
```

3. Train:

```
lot-train --config experiments/my_task.yaml
```

No code change required.

## 8. Extending the Framework

### 8.1 New attention mechanism

Inherit from BaseMultiHeadAttention (see model/vanilla.py), register:

```
ATTENTION_REGISTRY = {
    "vanilla": VanillaMultiHeadAttention,
    "lattice": LatticeMultiHeadAttention,
    "my_attn": MyAttention,
}
```

Use in YAML: attn_type: my_attn.

### 8.2 Custom CUDA kernel

Place my_kernel.cu in csrc/, include header:

```
torch::Tensor my_kernel(torch::Tensor phase, ...);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_kernel", &my_kernel, "...");
}
```

Python side:

```
from torch.utils.cpp_extension import load
my_ext = load(name="my_ext", sources=["csrc/my_kernel.cu"], verbose=True)
```

Call inside components.py - fallback to PyTorch if not compiled.

## 9. Monitoring & Visualisation

Lightning logs **automatically**:

- train/val_loss, ce_loss, exact_match, sync_order

- Attention entropy (per head)

- Phase snapshots every N steps (enable with --phase_snap_interval 500)

TensorBoard:

```
tensorboard --logdir runs/
```

Plot delimiter-wave (copy-reverse):

```
from lot.viz import plot_delimiter_wave
plot_delimiter_wave(logdir="runs/cr_lattice/version_0/", step=1000)
```

## 10. Reproducibility Checklist

[x] Deterministic data loader seeds

[x] pl.seed_everything called before any random op

[x] Exact YAML & code commit hash logged to hparams.yaml

[x] Checkpoint contains full trainer state (optimizer, RNG)

[x] Evaluation script uses same preprocessing & vocab


## 11. Performance Notes (example/template)
| Model | params | {dataset} 1 epoch | {n}×{gpu|cpu:name} | sync overhead |
| -------- | ------- | ------- |  ------- |  ------- |
| LOT-{dataset} (256d, 8 heads) | 10M | 0.97 bpc |  3 h 50 m | +7 % vs vanilla |
| vanilla-{dataset} (256d, 8 heads) | 10M | 0.98 bpc |  3 h 35 m | - |

(Example run; exact numbers depend on CUDA kernel or CPU usage.)

## 12. Contributing

Project git-flow model:

1. Fork & create feature branch feat/<name>

2. Add unit tests (pytest tests/) & update docs

3. Open PR -> CI must pass (lint, type-check, GPU tests on copy-reverse ≤ 10 min)

4. Tag maintainer for review

See CONTRIBUTING.md for coding style (black, isort, mypy).

## 13. Citing

If you use this framework, please cite:

```
@software{lot2025,
  title = {Lattice Oscillator Transformer},
  url  = {https://github.com/joshause/lot},
  year = {2025}
}
```

## 14. Licence

This project is open source and available under the [MIT License](LICENSE)
