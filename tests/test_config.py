"""Tests for memora.config — configuration loading and validation."""

import pytest
from pydantic import ValidationError

from memora.config import (
    DataConfig,
    MemoraConfig,
    ModelConfig,
    TrainingConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny() -> MemoraConfig:
    """Load the tiny config as a base for most tests."""
    return MemoraConfig.from_yaml("configs/tiny.yaml")


def _300m() -> MemoraConfig:
    """Load the 300m config."""
    return MemoraConfig.from_yaml("configs/300m.yaml")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_300m_config_loads():
    cfg = _300m()
    assert cfg.model.d_model == 960
    assert cfg.model.n_heads == 15
    assert cfg.model.n_layers == 24


def test_300m_warmup_tokens_matches_plan():
    """Approved plan value: ~131M tokens."""
    cfg = _300m()
    assert cfg.training.warmup_tokens == 131_072_000


def test_300m_global_batch_matches_plan():
    """Approved plan value: 524,288 tokens. Never silently change this."""
    cfg = _300m()
    assert cfg.training.global_batch_tokens == 524_288


def test_tiny_config_loads():
    cfg = _tiny()
    assert cfg.model.d_model == 128
    assert cfg.model.n_heads == 2
    assert cfg.training.global_batch_tokens == 1024


# ---------------------------------------------------------------------------
# grad_accum_steps — arithmetic verified against plan
# ---------------------------------------------------------------------------


def test_grad_accum_1gpu():
    """300m config: 524288 / (16 * 1024 * 1 GPU) = 32 accumulation steps."""
    cfg = _300m()
    assert cfg.training.grad_accum_steps(world_size=1) == 32


def test_grad_accum_2gpu():
    """300m config: 524288 / (16 * 1024 * 2 GPU) = 16 accumulation steps."""
    cfg = _300m()
    assert cfg.training.grad_accum_steps(world_size=2) == 16


def test_grad_accum_tiny():
    """Tiny config: 1024 / (2 * 128 * 1 GPU) = 4 accumulation steps."""
    cfg = _tiny()
    assert cfg.training.grad_accum_steps(world_size=1) == 4


def test_grad_accum_tiny_2gpu():
    """Tiny config: 1024 / (2 * 128 * 2 GPU) = 2 accumulation steps."""
    cfg = _tiny()
    assert cfg.training.grad_accum_steps(world_size=2) == 2


def test_grad_accum_bad_world_size():
    cfg = _tiny()
    with pytest.raises(ValueError, match="world_size"):
        cfg.training.grad_accum_steps(world_size=0)


def test_grad_accum_indivisible():
    """global_batch_tokens not divisible by micro*seq*world_size raises."""
    t = TrainingConfig.model_validate(
        {
            "global_batch_tokens": 1024,
            "micro_batch_size": 2,
            "seq_len": 128,
            "warmup_tokens": 100,
            "max_tokens": 10000,
        }
    )
    # 1024 / (2*128*3) = 1024/768 — not integer
    with pytest.raises(ValueError, match="not divisible"):
        t.grad_accum_steps(world_size=3)


# ---------------------------------------------------------------------------
# model_config = frozen=True — mutation must use model_copy(update=...)
# ---------------------------------------------------------------------------


def test_frozen_prevents_direct_mutation():
    cfg = _tiny()
    with pytest.raises(ValidationError):
        cfg.training = cfg.training.model_copy(update={"lr": 1e-3})  # type: ignore[misc]


def test_model_copy_update_works():
    """model_copy(update=...) is the supported way to derive a modified config."""
    cfg = _tiny()
    new_training = cfg.training.model_copy(update={"lr": 1e-3})
    assert new_training.lr == 1e-3
    # original unchanged
    assert cfg.training.lr == 6e-4


# ---------------------------------------------------------------------------
# extra="forbid" — unknown keys must fail loudly
# ---------------------------------------------------------------------------


def test_unknown_key_in_model_section():
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        ModelConfig.model_validate({"d_model": 128, "nonexistent_key": 99})


def test_unknown_key_in_training_section():
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        TrainingConfig.model_validate(
            {
                "global_batch_tokens": 1024,
                "micro_batch_size": 2,
                "seq_len": 128,
                "warmup_tokens": 100,
                "max_tokens": 10000,
                "typo_lr_rate": 1e-4,  # misspelled field
            }
        )


def test_unknown_key_in_top_level():
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        MemoraConfig.model_validate({"not_a_section": {}})


# ---------------------------------------------------------------------------
# ModelConfig validation
# ---------------------------------------------------------------------------


def test_d_model_not_divisible_by_n_heads():
    with pytest.raises(ValidationError, match="divisible"):
        ModelConfig.model_validate({"d_model": 100, "n_heads": 3})


def test_head_dim_must_be_even():
    """d_model=6, n_heads=3 → head_dim=2 (even, ok)."""
    m = ModelConfig.model_validate({"d_model": 6, "n_heads": 3})
    assert m.d_model // m.n_heads == 2

    """d_model=6, n_heads=2 → head_dim=3 (odd, fails)."""
    with pytest.raises(ValidationError, match="even"):
        ModelConfig.model_validate({"d_model": 6, "n_heads": 2})


# ---------------------------------------------------------------------------
# TrainingConfig validation
# ---------------------------------------------------------------------------


def test_min_lr_less_than_lr():
    with pytest.raises(ValidationError, match="min_lr"):
        TrainingConfig.model_validate(
            {
                "lr": 1e-4,
                "min_lr": 1e-3,
                "global_batch_tokens": 1024,
                "micro_batch_size": 2,
                "seq_len": 128,
                "warmup_tokens": 100,
                "max_tokens": 10000,
            }
        )


def test_warmup_tokens_must_be_less_than_max_tokens():
    with pytest.raises(ValidationError, match="warmup_tokens"):
        TrainingConfig.model_validate(
            {
                "global_batch_tokens": 1024,
                "micro_batch_size": 2,
                "seq_len": 128,
                "warmup_tokens": 10000,
                "max_tokens": 5000,  # warmup >= max → error
            }
        )


def test_wsd_budget_check():
    """For wsd: warmup + decay must not exceed max_tokens."""
    with pytest.raises(ValidationError, match="wsd"):
        TrainingConfig.model_validate(
            {
                "global_batch_tokens": 1024,
                "micro_batch_size": 2,
                "seq_len": 128,
                "warmup_tokens": 5000,
                "wsd_decay_tokens": 6000,
                "max_tokens": 10000,  # 5000 + 6000 > 10000
                "lr_schedule": "wsd",
            }
        )


def test_wsd_budget_ok():
    t = TrainingConfig.model_validate(
        {
            "global_batch_tokens": 1024,
            "micro_batch_size": 2,
            "seq_len": 128,
            "warmup_tokens": 2000,
            "wsd_decay_tokens": 3000,
            "max_tokens": 10000,
            "lr_schedule": "wsd",
        }
    )
    assert t.lr_schedule == "wsd"


def test_global_batch_multiple_of_micro():
    with pytest.raises(ValidationError, match="multiple"):
        TrainingConfig.model_validate(
            {
                "global_batch_tokens": 1000,
                "micro_batch_size": 2,
                "seq_len": 128,
                "warmup_tokens": 100,
                "max_tokens": 10000,
            }
        )


# ---------------------------------------------------------------------------
# DataConfig validation
# ---------------------------------------------------------------------------


def test_val_ratio_must_be_positive():
    """val_ratio=0.0 must be rejected (gt=0, not ge=0)."""
    with pytest.raises(ValidationError):
        DataConfig.model_validate({"val_ratio": 0.0})


def test_val_ratio_must_be_less_than_one():
    with pytest.raises(ValidationError):
        DataConfig.model_validate({"val_ratio": 1.5})


# ---------------------------------------------------------------------------
# Cross-section validation
# ---------------------------------------------------------------------------


def test_seq_len_must_match_max_seq_len():
    cfg = _tiny()
    # Derive a new config where seq_len != max_seq_len
    bad = {
        "model": cfg.model.model_dump() | {"max_seq_len": 64},
        "training": cfg.training.model_dump() | {"seq_len": 128},
        "checkpoint": cfg.checkpoint.model_dump(),
        "session": cfg.session.model_dump(),
        "data": cfg.data.model_dump(),
        "evaluation": cfg.evaluation.model_dump(),
        "logging": cfg.logging.model_dump(),
        "distributed": cfg.distributed.model_dump(),
    }
    with pytest.raises(ValidationError, match="seq_len"):
        MemoraConfig.model_validate(bad)
