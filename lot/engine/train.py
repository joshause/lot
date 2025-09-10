"""
Lightning-based training loop.
Logs:  ce_loss, exact_match%, sync_order, grad_norm, lr
"""
import torch, torch.nn.functional as F, pytorch_lightning as pl

class LotModule(pl.LightningModule):
    def __init__(self, model, lr: float = 1e-3, weight_decay: float = 0.01,
                 lambda_sync: float = 0.01, lambda_diversity: float = 0.01):
        super().__init__()
        self.model = model
        self.lr = float(lr)                       # defensive cast
        self.wd = float(weight_decay)
        self.lam_sync = lambda_sync
        self.lam_div = lambda_diversity
        self.save_hyperparameters(logger=False)   # ensures lr/wd are logged as scalars
        self.save_hyperparameters(ignore=["model"]) # ensures model is not saved as a hyperparameter

    def _common_step(self, batch, stage: str):
        x, y = batch
        logits, info = self.model(x)
        loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

        # regularisers
        sync_loss, div_loss = torch.tensor(0.0), torch.tensor(0.0)
        if self.lam_sync > 0 or self.lam_div > 0:
            phases = torch.stack([info["layers"][l]["phase"] for l in range(len(info["layers"]))])  # (L,H)
            complex = torch.exp(1j * phases)
            order = torch.abs(complex.mean(1))  # (L,)
            sync_loss = (1 - order).mean()
            freq = self.model.blocks[0].attn.intrinsic_freq  # (H,)
            div_loss = torch.exp(-torch.var(freq))
        loss = loss_ce + self.lam_sync * sync_loss + self.lam_div * div_loss

        # exact match
        preds = logits.argmax(-1)
        mask = y.ne(-100)
        em = (preds.eq(y) & mask).all(dim=-1).float().mean()

        self.log_dict({f"{stage}_loss": loss, f"{stage}_ce": loss_ce,
                       f"{stage}_em": em, f"{stage}_sync": 1 - sync_loss}, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._common_step(batch, "train")

    def validation_step(self, batch, _):
        self._common_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)