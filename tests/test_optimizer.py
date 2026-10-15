"""Tests for memora.training.optimizer (Stage 7a)."""

from __future__ import annotations

import pytest
import torch

from memora.config import MemoraConfig
from memora.model import MemoraModel
from memora.training.optimizer import build_optimizer, set_lr


@pytest.fixture
def tiny_config() -> MemoraConfig:
    return MemoraConfig.from_yaml("configs/tiny.yaml")


@pytest.fixture
def tiny_model(tiny_config: MemoraConfig) -> MemoraModel:
    return MemoraModel(tiny_config.model, seed=0)


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------


def test_optimizer_type(tiny_model: MemoraModel, tiny_config: MemoraConfig):
    opt = build_optimizer(tiny_model, tiny_config.training)
    assert isinstance(opt, torch.optim.AdamW)


def test_optimizer_two_param_groups(tiny_model: MemoraModel, tiny_config: MemoraConfig):
    """There must be exactly two param groups: decay and no-decay."""
    opt = build_optimizer(tiny_model, tiny_config.training)
    assert len(opt.param_groups) == 2


def test_optimizer_weight_decay_values(tiny_model: MemoraModel, tiny_config: MemoraConfig):
    """First group has weight_decay=cfg.weight_decay; second group has weight_decay=0."""
    opt = build_optimizer(tiny_model, tiny_config.training)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(tiny_config.training.weight_decay)
    assert opt.param_groups[1]["weight_decay"] == pytest.approx(0.0)


def test_optimizer_lr(tiny_model: MemoraModel, tiny_config: MemoraConfig):
    """All param groups start with lr from config."""
    opt = build_optimizer(tiny_model, tiny_config.training)
    for group in opt.param_groups:
        assert group["lr"] == pytest.approx(tiny_config.training.lr)


def test_optimizer_betas(tiny_model: MemoraModel, tiny_config: MemoraConfig):
    """Betas match config."""
    opt = build_optimizer(tiny_model, tiny_config.training)
    for group in opt.param_groups:
        assert group["betas"] == (tiny_config.training.beta1, tiny_config.training.beta2)


def test_optimizer_eps(tiny_model: MemoraModel, tiny_config: MemoraConfig):
    """Eps matches config."""
    opt = build_optimizer(tiny_model, tiny_config.training)
    for group in opt.param_groups:
        assert group["eps"] == pytest.approx(tiny_config.training.eps)


def test_optimizer_all_params_covered(tiny_model: MemoraModel, tiny_config: MemoraConfig):
    """Every parameter appears in exactly one group (same as weight_decay_groups contract)."""
    opt = build_optimizer(tiny_model, tiny_config.training)
    all_ids_in_groups = {id(p) for g in opt.param_groups for p in g["params"]}
    all_model_ids = {id(p) for p in tiny_model.parameters()}
    assert all_ids_in_groups == all_model_ids


# ---------------------------------------------------------------------------
# set_lr
# ---------------------------------------------------------------------------


def test_set_lr_changes_every_group(tiny_model: MemoraModel, tiny_config: MemoraConfig):
    """set_lr must update lr in every param group."""
    opt = build_optimizer(tiny_model, tiny_config.training)
    new_lr = 1e-5
    set_lr(opt, new_lr)
    for group in opt.param_groups:
        assert group["lr"] == pytest.approx(new_lr)


def test_set_lr_does_not_affect_weight_decay(tiny_model: MemoraModel, tiny_config: MemoraConfig):
    """set_lr must not touch weight_decay or other hyperparams."""
    opt = build_optimizer(tiny_model, tiny_config.training)
    original_wds = [g["weight_decay"] for g in opt.param_groups]
    set_lr(opt, 1e-5)
    for wd_orig, g in zip(original_wds, opt.param_groups, strict=True):
        assert g["weight_decay"] == wd_orig
