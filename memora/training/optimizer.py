"""Optimizer builder for Memora training.

build_optimizer(model, training_cfg) -> AdamW
set_lr(optimizer, lr) -> None
"""

from __future__ import annotations

import torch
import torch.optim as optim

from memora.config import TrainingConfig
from memora.model import MemoraModel


def build_optimizer(model: MemoraModel, cfg: TrainingConfig) -> optim.AdamW:
    """Build an AdamW optimizer using model.weight_decay_groups().

    Parameters
    ----------
    model:
        MemoraModel instance. weight_decay_groups() provides the param groups.
    cfg:
        TrainingConfig with lr, betas, eps, weight_decay, and fused flag.

    Returns
    -------
    torch.optim.AdamW with two param groups: decayed and non-decayed.
    """
    groups = model.weight_decay_groups(cfg.weight_decay)

    # fused AdamW is only available when CUDA is present and explicitly enabled.
    use_fused = (
        getattr(cfg, "fused_optimizer", False)
        and torch.cuda.is_available()
    )
    kwargs: dict = {
        "lr": cfg.lr,
        "betas": (cfg.beta1, cfg.beta2),
        "eps": cfg.eps,
    }
    if use_fused:
        kwargs["fused"] = True  # pragma: no cover

    return optim.AdamW(groups, **kwargs)


def set_lr(optimizer: optim.Optimizer, lr: float) -> None:
    """Set the learning rate for every param group in the optimizer."""
    for group in optimizer.param_groups:
        group["lr"] = lr
