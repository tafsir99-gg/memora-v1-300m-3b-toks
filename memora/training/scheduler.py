"""Learning-rate scheduler — pure, stateless function.

lr_at(tokens_seen, cfg) -> float

No state is stored here: the LR at any token count is computed from
(tokens_seen, config) alone, so resume is automatically consistent as long as
the config's must-match fields (lr, min_lr, warmup_tokens, max_tokens,
lr_schedule, wsd_decay_tokens, wsd_decay_start_tokens) are unchanged.

Schedules
---------
cosine:
    Linear warmup 0 -> lr over [0, warmup_tokens].
    Cosine decay lr -> min_lr over [warmup_tokens, max_tokens].
    Hold min_lr for tokens_seen > max_tokens.

wsd (warmup-stable-decay):
    Linear warmup 0 -> lr over [0, warmup_tokens].
    Stable at lr over [warmup_tokens, decay_start].
    Linear decay lr -> min_lr over [decay_start, decay_start + wsd_decay_tokens].
    Hold min_lr for tokens_seen > decay_start + wsd_decay_tokens.

    decay_start = wsd_decay_start_tokens if set, else max_tokens - wsd_decay_tokens.
    Constraints validated by TrainingConfig: decay_start >= warmup_tokens and
    decay_start + wsd_decay_tokens <= max_tokens.
"""

from __future__ import annotations

import math

from memora.config import TrainingConfig


def lr_at(tokens_seen: int, cfg: TrainingConfig) -> float:
    """Return the learning rate at ``tokens_seen`` tokens.

    Parameters
    ----------
    tokens_seen:
        Number of tokens processed so far (0-indexed: 0 = before first step).
    cfg:
        TrainingConfig with lr, min_lr, warmup_tokens, lr_schedule, etc.

    Returns
    -------
    float learning rate in [min_lr, lr].
    """
    lr = cfg.lr
    min_lr = cfg.min_lr

    # --- Warmup ---
    if cfg.warmup_tokens > 0 and tokens_seen < cfg.warmup_tokens:
        return lr * tokens_seen / cfg.warmup_tokens

    if cfg.lr_schedule == "cosine":
        return _cosine_lr(tokens_seen, lr, min_lr, cfg.warmup_tokens, cfg.max_tokens)

    if cfg.lr_schedule == "wsd":
        decay_start = (
            cfg.wsd_decay_start_tokens
            if cfg.wsd_decay_start_tokens is not None
            else cfg.max_tokens - cfg.wsd_decay_tokens
        )
        return _wsd_lr(
            tokens_seen,
            lr,
            min_lr,
            cfg.warmup_tokens,
            decay_start,
            cfg.wsd_decay_tokens,
        )

    raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cosine_lr(
    tokens_seen: int,
    lr: float,
    min_lr: float,
    warmup_tokens: int,
    max_tokens: int,
) -> float:
    """Cosine decay from lr to min_lr over [warmup_tokens, max_tokens], then hold."""
    if tokens_seen >= max_tokens:
        return min_lr
    # Linear interpolation in cosine space
    progress = (tokens_seen - warmup_tokens) / max(max_tokens - warmup_tokens, 1)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine_factor * (lr - min_lr)


def _wsd_lr(
    tokens_seen: int,
    lr: float,
    min_lr: float,
    warmup_tokens: int,
    decay_start: int,
    decay_tokens: int,
) -> float:
    """WSD: stable at lr, then linear decay to min_lr, then hold."""
    decay_end = decay_start + decay_tokens
    if tokens_seen < decay_start:
        # Stable phase (post-warmup)
        return lr
    if tokens_seen >= decay_end:
        return min_lr
    # Linear decay
    progress = (tokens_seen - decay_start) / decay_tokens
    return lr + progress * (min_lr - lr)
