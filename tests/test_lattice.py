import torch, pytest
from lot.model.lattice import LatticeMultiHeadAttention

@pytest.mark.parametrize("n_heads", [4, 6, 8])
def test_kuramoto_no_nan(n_heads):
    H = n_heads
    phase = torch.rand(H)
    freq = torch.ones(H)
    coupling = torch.ones(H) * 0.1
    nbr_idx = torch.randint(0, H, (H, 8))
    nbr_w = torch.rand(H, 8)
    from lot.model.components import kuramoto_step
    for _ in range(100):
        phase = kuramoto_step(phase, freq, coupling, nbr_idx, nbr_w)
        assert phase.isfinite().all()

def test_param_count_same_as_vanilla():
    from lot.model.vanilla import VanillaMultiHeadAttention
    from lot.model.lattice import LatticeMultiHeadAttention
    C, H = 256, 8
    vanilla = VanillaMultiHeadAttention(C, H)
    lattice = LatticeMultiHeadAttention(C, H, lattice_shape=(2, 4))
    # lattice has 2 extra buffers (phase, freq) + 1 learnable (coupling)  →  H+2 parameters
    diff = sum(p.numel() for p in lattice.parameters() if p.requires_grad) - \
           sum(p.numel() for p in vanilla.parameters() if p.requires_grad)
    assert diff == H
