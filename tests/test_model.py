"""Tests for the Memora v1 model architecture."""

import math

import pytest
import torch

from memora.config import MemoraConfig
from memora.model import MemoraModel, RotaryPositionalEmbedding


@pytest.fixture
def tiny_config() -> MemoraConfig:
    return MemoraConfig.from_yaml("configs/tiny.yaml")


@pytest.fixture
def config_300m() -> MemoraConfig:
    return MemoraConfig.from_yaml("configs/300m.yaml")


def test_model_shapes(tiny_config: MemoraConfig):
    model = MemoraModel(tiny_config.model)
    B, T = 2, 16
    idx = torch.randint(0, tiny_config.model.vocab_size, (B, T))

    logits, loss = model(idx)
    assert logits.shape == (B, T, tiny_config.model.vocab_size)
    assert loss is None

    targets = torch.randint(0, tiny_config.model.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_loss_non_contiguous(tiny_config: MemoraConfig):
    model = MemoraModel(tiny_config.model)
    B, T_plus_1 = 2, 17
    batch = torch.randint(0, tiny_config.model.vocab_size, (B, T_plus_1))

    # Non-contiguous slices
    idx = batch[:, :-1]
    targets = batch[:, 1:]
    assert not targets.is_contiguous()

    _, loss1 = model(idx, targets)

    # Contiguous copies
    _, loss2 = model(idx.contiguous(), targets.contiguous())

    assert loss1 is not None and loss2 is not None
    assert torch.allclose(loss1, loss2)


def test_causal_masking(tiny_config: MemoraConfig):
    model = MemoraModel(tiny_config.model)
    model.eval()  # Disable dropout

    B, T = 1, 8
    idx1 = torch.randint(0, tiny_config.model.vocab_size, (B, T))
    idx2 = idx1.clone()

    # Change the LAST token
    idx2[0, -1] = (idx2[0, -1] + 1) % tiny_config.model.vocab_size

    with torch.no_grad():
        logits1, _ = model(idx1)
        logits2, _ = model(idx2)

    if idx1.device.type == "cuda":
        assert torch.allclose(logits1[:, :-1, :], logits2[:, :-1, :], atol=1e-6)
    else:
        assert torch.equal(logits1[:, :-1, :], logits2[:, :-1, :])
    assert not torch.equal(logits1[:, -1:, :], logits2[:, -1:, :])


def test_rope_properties(tiny_config: MemoraConfig):
    dim = 16
    max_seq_len = 32
    rope = RotaryPositionalEmbedding(dim, max_seq_len)

    # 1. head_dim must be even (already enforced implicitly by dim // 2, but let's test)
    assert dim % 2 == 0

    # 2. position 0 is identity
    x = torch.randn(1, 1, 1, dim)
    x_rot = rope(x)
    assert torch.allclose(x, x_rot, atol=1e-6)

    # 3. vector norm is preserved
    x = torch.randn(1, max_seq_len, 1, dim)
    x_rot = rope(x)
    assert torch.allclose(torch.norm(x, dim=-1), torch.norm(x_rot, dim=-1), atol=1e-5)

    # 4. Naive reference comparison
    x_ref = torch.randn(1, max_seq_len, 1, dim)
    x_rot_ref = rope(x_ref)
    for pos in [0, 1, 7, 31]:
        for i in range(dim // 2):
            angle = pos * (10000.0 ** (-2.0 * i / dim))
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            orig_1 = x_ref[0, pos, 0, i].item()
            orig_2 = x_ref[0, pos, 0, i + dim // 2].item()
            expected_1 = orig_1 * cos_a - orig_2 * sin_a
            expected_2 = orig_2 * cos_a + orig_1 * sin_a
            assert math.isclose(x_rot_ref[0, pos, 0, i].item(), expected_1, abs_tol=1e-5)
            assert math.isclose(x_rot_ref[0, pos, 0, i + dim // 2].item(), expected_2, abs_tol=1e-5)

    # 5. Spy test proving Attention calls rope exactly once on q and k
    import unittest.mock
    model = MemoraModel(tiny_config.model)
    with unittest.mock.patch.object(model.rope, 'forward', wraps=model.rope.forward) as mock_rope:
        idx = torch.randint(0, tiny_config.model.vocab_size, (1, 8))
        model(idx)
        # called twice per layer (once for q, once for k)
        assert mock_rope.call_count == 2 * tiny_config.model.n_layers


def test_tied_embeddings(tiny_config: MemoraConfig):
    # Tied
    cfg_tied = tiny_config.model.model_copy(update={"tie_embeddings": True})
    model_tied = MemoraModel(cfg_tied)
    assert model_tied.lm_head.weight is model_tied.embedding.weight

    # Untied
    cfg_untied = tiny_config.model.model_copy(update={"tie_embeddings": False})
    model_untied = MemoraModel(cfg_untied)
    assert model_untied.lm_head.weight is not model_untied.embedding.weight


def test_parameter_count(tiny_config: MemoraConfig, config_300m: MemoraConfig):
    # Test 1: Tiny matches analytic formula
    model_tiny = MemoraModel(tiny_config.model)
    actual_tiny = sum(p.numel() for p in model_tiny.parameters())

    # We will compute the analytic count manually to verify
    m_cfg = tiny_config.model
    embedding = m_cfg.vocab_size * m_cfg.d_model
    per_layer = 4 * m_cfg.d_model**2 + 2 * m_cfg.d_model * m_cfg.d_ff + 4 * m_cfg.d_model
    all_layers = m_cfg.n_layers * per_layer
    final_norm = 2 * m_cfg.d_model
    analytic_tiny = embedding + all_layers + final_norm

    assert actual_tiny == analytic_tiny

    # Test 2: 300m matches exactly 296,972,160 on meta device (tied) and 328,429,440 (untied)
    with torch.device("meta"):
        cfg_tied = config_300m.model.model_copy(update={"tie_embeddings": True})
        model_300m_tied = MemoraModel(cfg_tied)
        actual_300m_tied = sum(p.numel() for p in model_300m_tied.parameters())
        assert actual_300m_tied == 296972160

        cfg_untied = config_300m.model.model_copy(update={"tie_embeddings": False})
        model_300m_untied = MemoraModel(cfg_untied)
        actual_300m_untied = sum(p.numel() for p in model_300m_untied.parameters())
        assert actual_300m_untied == 328429440


def test_initialization_stats(tiny_config: MemoraConfig):
    model = MemoraModel(tiny_config.model, seed=42)

    # Weight std close to init_std
    embedding_std = model.embedding.weight.std().item()
    assert math.isclose(embedding_std, tiny_config.model.init_std, rel_tol=0.1)

    # Residual-scaled projections are smaller
    scale = 1.0 / math.sqrt(2.0 * tiny_config.model.n_layers)
    expected_proj_std = tiny_config.model.init_std * scale

    import typing

    from memora.model import Block
    block = typing.cast(Block, model.layers[0])
    attn_proj_std = block.attn.proj.weight.std().item()
    mlp_fc2_std = block.mlp.fc2.weight.std().item()
    assert math.isclose(attn_proj_std, expected_proj_std, rel_tol=0.1)
    assert math.isclose(mlp_fc2_std, expected_proj_std, rel_tol=0.1)

    attn_qkv_std = block.attn.qkv.weight.std().item()
    mlp_fc1_std = block.mlp.fc1.weight.std().item()
    assert math.isclose(attn_qkv_std, tiny_config.model.init_std, rel_tol=0.1)
    assert math.isclose(mlp_fc1_std, tiny_config.model.init_std, rel_tol=0.1)

    # Initial loss is close to -ln(1/vocab) = ln(vocab)
    model.eval()
    idx = torch.randint(0, tiny_config.model.vocab_size, (2, 128))
    targets = torch.randint(0, tiny_config.model.vocab_size, (2, 128))
    _, loss = model(idx, targets)

    expected_loss = math.log(tiny_config.model.vocab_size)
    assert loss is not None
    assert math.isclose(loss.item(), expected_loss, rel_tol=0.1)


def test_forward_backward_gradients(tiny_config: MemoraConfig):
    model = MemoraModel(tiny_config.model, seed=42)

    idx = torch.randint(0, tiny_config.model.vocab_size, (1, 16))
    targets = torch.randint(0, tiny_config.model.vocab_size, (1, 16))

    _, loss = model(idx, targets)
    assert loss is not None
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


def test_determinism(tiny_config: MemoraConfig):
    m1 = MemoraModel(tiny_config.model, seed=123)
    m2 = MemoraModel(tiny_config.model, seed=123)
    m3 = MemoraModel(tiny_config.model, seed=456)

    # 1. Init determinism
    assert torch.equal(m1.embedding.weight, m2.embedding.weight)
    assert not torch.equal(m1.embedding.weight, m3.embedding.weight)

    # 2. Forward output determinism
    idx = torch.randint(0, tiny_config.model.vocab_size, (1, 16), generator=torch.Generator().manual_seed(42))

    m1.eval()
    m2.eval()
    with torch.no_grad():
        out1, _ = m1(idx)
        out2, _ = m2(idx)

    assert torch.equal(out1, out2)


def test_weight_decay_groups(tiny_config: MemoraConfig):
    model1 = MemoraModel(tiny_config.model, seed=1)
    model2 = MemoraModel(tiny_config.model, seed=2)

    groups1 = model1.weight_decay_groups(weight_decay=0.1)
    groups2 = model2.weight_decay_groups(weight_decay=0.1)

    assert len(groups1) == 2
    assert len(groups1) == len(groups2)

    for g1, g2 in zip(groups1, groups2, strict=False):
        assert len(g1["params"]) == len(g2["params"])
        for p1, p2 in zip(g1["params"], g2["params"], strict=False):
            assert p1.shape == p2.shape

    decay_params = groups1[0]["params"]
    no_decay_params = groups1[1]["params"]

    all_params_in_groups = {id(p) for p in decay_params + no_decay_params}
    all_model_params = {id(p) for p in model1.parameters()}

    # Every parameter is in exactly one group
    assert all_params_in_groups == all_model_params

    # Embedding is in no-decay
    assert id(model1.embedding.weight) in {id(p) for p in no_decay_params}

    # Norms are in no-decay
    assert id(model1.ln_f.weight) in {id(p) for p in no_decay_params}


def test_tiny_overfit(tiny_config: MemoraConfig):
    """The tiny model memorizes a fixed small batch (loss falls below 0.1)."""
    torch.manual_seed(42)
    model = MemoraModel(tiny_config.model, seed=42)

    # Single batch
    idx = torch.randint(0, tiny_config.model.vocab_size, (2, 32))
    # Targets shifted by 1
    targets = torch.cat([idx[:, 1:], torch.randint(0, tiny_config.model.vocab_size, (2, 1))], dim=1)

    optimizer = torch.optim.AdamW(
        model.weight_decay_groups(tiny_config.training.weight_decay),
        lr=tiny_config.training.lr,
        betas=(tiny_config.training.beta1, tiny_config.training.beta2),
        eps=tiny_config.training.eps,
    )

    for _ in range(200):
        optimizer.zero_grad()
        _, loss = model(idx, targets)
        assert loss is not None
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1


@pytest.mark.gpu
def test_cuda_fp16_autocast(tiny_config: MemoraConfig):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = MemoraModel(tiny_config.model, seed=42).cuda()
    idx = torch.randint(0, tiny_config.model.vocab_size, (1, 16)).cuda()
    targets = torch.randint(0, tiny_config.model.vocab_size, (1, 16)).cuda()

    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits, loss = model(idx, targets)

    assert logits.dtype == torch.float16
    assert loss is not None and loss.dtype == torch.float32  # loss upcasted in model

    assert torch.isfinite(loss)
    assert torch.isfinite(logits).all()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
