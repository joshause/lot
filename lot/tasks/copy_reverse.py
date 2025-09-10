"""
Deterministic copy-reverse dataset.
Guarantees perfect length, single delimiter, exact reversal.
"""
import torch
from torch.utils.data import Dataset


class CopyReverseDataset(Dataset):
    """
    Sequence:  [x0 … xL/2-1]  [delimiter]  [xL/2-1 … x0]
    Target:    [x1 … xL/2-1]  [delimiter]  [x0 … xL/2-1]  [pad]
    Exact-match accuracy is well-defined.
    """

    def __init__(self, vocab_size: int = 1000, seq_len: int = 128, n_samples: int = 50_000):
        assert seq_len % 2 == 0
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.samples = []
        half = seq_len // 2
        delim = 2
        pad = 0
        gen = torch.Generator().manual_seed(42)
        for _ in range(n_samples):
            prefix = torch.randint(3, vocab_size, (half,), generator=gen)
            suffix = prefix.flip(0)
            seq = torch.cat([prefix, torch.tensor([delim]), suffix])
            tgt = torch.cat([seq[1:], torch.tensor([pad])])
            self.samples.append((seq, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]