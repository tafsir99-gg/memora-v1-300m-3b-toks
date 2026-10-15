"""Tests for memora.training.scheduler (Stage 7a)."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from memora.config import MemoraConfig, TrainingConfig
from memora.training.scheduler import lr_at

# ---------------------------------------------------------------------------

# Helpers
# ---------------------------------------------------------------------------


def _cfg(**kwargs) -> TrainingConfig:
    """Build a minimal TrainingConfig, overriding defaults as needed."""
    defaults = {
        "global_batch_tokens": 1024,
        "micro_batch_size": 2,
        "seq_len": 128,
        "lr": 6e-4,
        "min_lr": 6e-5,
        "warmup_tokens": 1000,
        "max_tokens": 10000,
        "lr_schedule": "cosine",
        "wsd_decay_tokens": 2000,
    }
    defaults.update(kwargs)
    return TrainingConfig.model_validate(defaults)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def test_warmup_at_zero():
    """At token 0, LR = 0 (linear warmup starts at 0)."""
    cfg = _cfg()
    assert lr_at(0, cfg) == pytest.approx(0.0)


def test_warmup_at_end():
    """At warmup_tokens, LR = lr (warmup complete)."""
    cfg = _cfg()
    assert lr_at(cfg.warmup_tokens, cfg) == pytest.approx(cfg.lr, rel=1e-9)


def test_warmup_midpoint():
    """Halfway through warmup, LR = lr/2."""
    cfg = _cfg(warmup_tokens=2000)
    assert lr_at(1000, cfg) == pytest.approx(cfg.lr / 2, rel=1e-9)


def test_warmup_zero_tokens_skips_warmup():
    """warmup_tokens=0 means skip warmup; LR at token 0 = lr."""
    cfg = _cfg(warmup_tokens=0)
    assert lr_at(0, cfg) == pytest.approx(cfg.lr, rel=1e-5)


# ---------------------------------------------------------------------------
# Cosine schedule
# ---------------------------------------------------------------------------


def test_cosine_at_warmup_end_equals_lr():
    cfg = _cfg()
    assert lr_at(cfg.warmup_tokens, cfg) == pytest.approx(cfg.lr, rel=1e-9)


def test_cosine_at_max_tokens_equals_min_lr():
    cfg = _cfg()
    assert lr_at(cfg.max_tokens, cfg) == pytest.approx(cfg.min_lr, rel=1e-9)


def test_cosine_beyond_max_tokens_holds_min_lr():
    cfg = _cfg()
    assert lr_at(cfg.max_tokens + 99999, cfg) == pytest.approx(cfg.min_lr, rel=1e-9)


def test_cosine_midpoint_closed_form():
    """At (warmup + max_tokens)/2, cosine factor = 0.5.
    LR = min_lr + 0.5 * (lr - min_lr).
    """
    cfg = _cfg()
    mid = (cfg.warmup_tokens + cfg.max_tokens) // 2
    expected = cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr)
    assert lr_at(mid, cfg) == pytest.approx(expected, rel=1e-5)


def test_cosine_monotone_decreasing():
    """LR is non-increasing after warmup."""
    cfg = _cfg()
    tokens = list(range(cfg.warmup_tokens, cfg.max_tokens + 1, 500))
    lrs = [lr_at(t, cfg) for t in tokens]
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1] - 1e-12, f"LR increased at step {i}"


def test_cosine_continuity_at_warmup_boundary():
    """LR must be continuous at warmup_tokens.

    Derivation:
      At t = warmup_tokens - 1 (still in warmup):
        LR_left  = lr * (warmup_tokens - 1) / warmup_tokens
      At t = warmup_tokens (cosine, progress = 0):
        LR_right = min_lr + 1.0 * (lr - min_lr) = lr
      Exact gap = lr - lr * (W-1)/W = lr / warmup_tokens

    This equals exactly one warmup-step increment.  It is not a
    discontinuity — it is the final linear step of the ramp.  Any
    gap larger than (1 + ε_float) * lr / warmup_tokens would indicate
    a real schedule discontinuity.

    Threshold: 1.001 * lr / warmup_tokens (allows 0.1% float slack;
    float rounding in lr*(W-1)/W is < 1 ULP ≈ lr * 1e-15 << 0.001 * lr/W).
    """
    cfg = _cfg(warmup_tokens=1000)
    lhs = lr_at(cfg.warmup_tokens - 1, cfg)  # last warmup step
    rhs = lr_at(cfg.warmup_tokens, cfg)       # first post-warmup value
    # Expected exact gap = lr / warmup_tokens
    expected_gap = cfg.lr / cfg.warmup_tokens
    assert abs(lhs - rhs) == pytest.approx(expected_gap, rel=1e-9)
    # And it must be below the tight threshold
    threshold = 1.001 * expected_gap
    assert abs(lhs - rhs) < threshold


def test_cosine_1b_tokens_300m_config():
    """1B tokens under 300m cosine config: arithmetic check.

    From IMPLEMENTATION_PLAN:
      warmup = 131,072,000 tokens
      max    = 3,000,000,000 tokens
      lr     = 6e-4, min_lr = 6e-5

    At 1B tokens, progress = (1e9 - 131e6) / (3e9 - 131e6) ≈ 0.2907
    cosine_factor = 0.5 * (1 + cos(π * 0.2907)) ≈ 0.5 * (1 + 0.5777) ≈ 0.7889

    Arithmetic:
      LR ≈ 6e-5 + 0.7889 * (6e-4 - 6e-5) ≈ 6e-5 + 0.7889 * 5.4e-4 ≈ 6e-5 + 4.26e-4 ≈ 4.86e-4

    That is ~0.79 of the (lr - min_lr) range above min_lr, as the plan states.
    """
    cfg_300m = MemoraConfig.from_yaml("configs/300m.yaml")
    tcfg = cfg_300m.training
    tokens_1b = 1_000_000_000
    lr = lr_at(tokens_1b, tcfg)

    warmup = tcfg.warmup_tokens
    max_t = tcfg.max_tokens
    progress = (tokens_1b - warmup) / (max_t - warmup)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    expected = tcfg.min_lr + cosine_factor * (tcfg.lr - tcfg.min_lr)

    assert lr == pytest.approx(expected, rel=1e-9)
    # confirm ~79% of the lr-min_lr range above min_lr
    fraction_above_min = (lr - tcfg.min_lr) / (tcfg.lr - tcfg.min_lr)
    assert 0.75 <= fraction_above_min <= 0.83, f"Expected ~79%, got {fraction_above_min:.4f}"


# ---------------------------------------------------------------------------
# WSD schedule
# ---------------------------------------------------------------------------


def _wsd_cfg(**kwargs) -> TrainingConfig:
    defaults = {
        "global_batch_tokens": 1024,
        "micro_batch_size": 2,
        "seq_len": 128,
        "lr": 6e-4,
        "min_lr": 6e-5,
        "warmup_tokens": 1000,
        "max_tokens": 10000,
        "lr_schedule": "wsd",
        "wsd_decay_tokens": 2000,
    }
    defaults.update(kwargs)
    return TrainingConfig.model_validate(defaults)


def test_wsd_stable_phase():
    """After warmup and before decay, LR = lr."""
    cfg = _wsd_cfg()
    decay_start = cfg.max_tokens - cfg.wsd_decay_tokens  # 8000
    assert lr_at(cfg.warmup_tokens, cfg) == pytest.approx(cfg.lr, rel=1e-9)
    assert lr_at(5000, cfg) == pytest.approx(cfg.lr, rel=1e-9)
    assert lr_at(decay_start - 1, cfg) == pytest.approx(cfg.lr, rel=1e-9)


def test_wsd_decay_start():
    """At decay_start, LR is still at peak (linear decay not yet applied)."""
    cfg = _wsd_cfg()
    decay_start = cfg.max_tokens - cfg.wsd_decay_tokens  # 8000
    assert lr_at(decay_start, cfg) == pytest.approx(cfg.lr, rel=1e-9)


def test_wsd_decay_midpoint():
    """At decay_start + decay_tokens/2, LR = (lr + min_lr) / 2 (linear)."""
    cfg = _wsd_cfg()
    decay_start = cfg.max_tokens - cfg.wsd_decay_tokens
    mid = decay_start + cfg.wsd_decay_tokens // 2
    expected = cfg.lr + 0.5 * (cfg.min_lr - cfg.lr)
    assert lr_at(mid, cfg) == pytest.approx(expected, rel=1e-9)


def test_wsd_decay_end():
    """At decay_start + decay_tokens = max_tokens, LR = min_lr."""
    cfg = _wsd_cfg()
    assert lr_at(cfg.max_tokens, cfg) == pytest.approx(cfg.min_lr, rel=1e-9)


def test_wsd_hold_after_end():
    """Beyond max_tokens, LR stays at min_lr."""
    cfg = _wsd_cfg()
    assert lr_at(cfg.max_tokens + 5000, cfg) == pytest.approx(cfg.min_lr, rel=1e-9)


def test_wsd_explicit_decay_start():
    """wsd_decay_start_tokens overrides default."""
    cfg = _wsd_cfg(wsd_decay_start_tokens=3000)
    # Stable until 3000
    assert lr_at(2999, cfg) == pytest.approx(cfg.lr, rel=1e-9)
    # Midpoint of decay [3000, 5000]: progress=0.5
    assert lr_at(4000, cfg) == pytest.approx(
        cfg.lr + 0.5 * (cfg.min_lr - cfg.lr), rel=1e-9
    )
    # After decay end (5000)
    assert lr_at(5001, cfg) == pytest.approx(cfg.min_lr, rel=1e-9)


def test_wsd_continuity_at_warmup_boundary():
    """LR must be continuous at warmup_tokens for wsd schedule.

    Same derivation as cosine: the gap equals exactly one warmup-step
    increment (lr / warmup_tokens).  The stable phase returns lr exactly,
    so the gap is lr - lr*(W-1)/W = lr/W.
    """
    cfg = _wsd_cfg()
    lhs = lr_at(cfg.warmup_tokens - 1, cfg)
    rhs = lr_at(cfg.warmup_tokens, cfg)
    expected_gap = cfg.lr / cfg.warmup_tokens
    assert abs(lhs - rhs) == pytest.approx(expected_gap, rel=1e-9)
    threshold = 1.001 * expected_gap
    assert abs(lhs - rhs) < threshold


def test_wsd_continuity_at_decay_boundary():
    """LR is continuous at decay_start."""
    cfg = _wsd_cfg()
    decay_start = cfg.max_tokens - cfg.wsd_decay_tokens
    lhs = lr_at(decay_start - 1, cfg)
    rhs = lr_at(decay_start, cfg)
    assert abs(lhs - rhs) < 1e-9


# ---------------------------------------------------------------------------
# wsd_decay_start_tokens validation (already in test_config.py; here for schedule)
# ---------------------------------------------------------------------------


def test_wsd_start_before_warmup_raises():
    with pytest.raises(ValidationError):
        _wsd_cfg(warmup_tokens=5000, wsd_decay_start_tokens=1000)


def test_wsd_start_plus_decay_exceeds_max_raises():
    with pytest.raises(ValidationError):
        _wsd_cfg(wsd_decay_start_tokens=9000, wsd_decay_tokens=2000, max_tokens=10000)


# ---------------------------------------------------------------------------
# Discontinuous-schedule rejection: proves the threshold is tight
# ---------------------------------------------------------------------------


def test_continuity_threshold_rejects_discontinuous_schedule():
    """The 1.001 * lr/W threshold must FAIL for a schedule that jumps by 10% of lr.

    This test builds a deliberately bad schedule function that jumps by
    10% * lr at the warmup boundary, applies the same continuity check,
    and asserts the check raises AssertionError.  If this test fails (i.e.
    the AssertionError is NOT raised), the threshold is too loose.

    Derivation:
      Bad gap = 0.10 * lr  (10% of peak LR — obviously a real discontinuity)
      Threshold = 1.001 * lr / warmup_tokens = 1.001 * 6e-4 / 1000 = 6.006e-7
      Bad gap (6e-5) >> threshold (6.006e-7)  =>  check must fail.
    """
    cfg = _cfg(warmup_tokens=1000)

    # Simulate a bad schedule: warmup ramp that is fine, but at t=warmup
    # it starts 10% above where the ramp left off instead of smoothly
    # continuing to lr.
    def bad_schedule(t: int) -> float:
        """Same warmup, but adds a 10%-lr jump exactly at warmup_tokens."""
        if t < cfg.warmup_tokens:
            return float(cfg.lr * t / cfg.warmup_tokens)
        # Discontinuous jump: stable at lr + 10% jump
        return float(cfg.lr + 0.10 * cfg.lr)

    lhs = bad_schedule(cfg.warmup_tokens - 1)
    rhs = bad_schedule(cfg.warmup_tokens)
    bad_gap = abs(lhs - rhs)

    # Confirm the bad gap is large (for test comprehensibility)
    assert bad_gap == pytest.approx(0.10 * cfg.lr + cfg.lr / cfg.warmup_tokens, rel=1e-9)

    # The threshold that the real tests use
    threshold = 1.001 * cfg.lr / cfg.warmup_tokens

    # The bad gap must EXCEED the threshold
    assert bad_gap > threshold, (
        f"Bad gap {bad_gap:.6e} should exceed threshold {threshold:.6e}; "
        "threshold is too loose!"
    )
    # Confirm the continuity check on the bad schedule would fail:
    with pytest.raises(AssertionError):
        assert bad_gap < threshold
