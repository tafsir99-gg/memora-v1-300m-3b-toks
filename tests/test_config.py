import pytest
from pydantic import ValidationError

from memora.config import MemoraConfig


def test_300m_config_loads():
    config = MemoraConfig.from_yaml("configs/300m.yaml")
    assert config.model.d_model == 960
    assert config.model.n_heads == 15
    assert config.training.global_batch_tokens == 262144


def test_tiny_config_loads():
    config = MemoraConfig.from_yaml("configs/tiny.yaml")
    assert config.model.d_model == 128
    assert config.model.n_heads == 2
    assert config.training.global_batch_tokens == 1024


def test_d_model_divisible_by_n_heads():
    # Should work
    config = MemoraConfig.from_yaml("configs/tiny.yaml")
    config.model.d_model = 128
    config.model.n_heads = 4
    # Manually re-validate
    from memora.config import ModelConfig

    ModelConfig.model_validate(config.model.model_dump())

    # Should fail
    config.model.d_model = 128
    config.model.n_heads = 3
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(config.model.model_dump())


def test_global_batch_multiple_of_micro():
    config = MemoraConfig.from_yaml("configs/tiny.yaml")
    # Tiny config has global=1024, micro=2, seq_len=128
    # micro * seq_len = 256
    # 1024 % 256 == 0 -> OK
    from memora.config import TrainingConfig

    TrainingConfig.model_validate(config.training.model_dump())

    # Set invalid
    config.training.global_batch_tokens = 1000
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(config.training.model_dump())


def test_seq_len_matches():
    config = MemoraConfig.from_yaml("configs/tiny.yaml")

    # Change training seq len so it doesn't match model max seq len
    config.training.seq_len = 256
    with pytest.raises(ValidationError):
        MemoraConfig.model_validate(config.model_dump())


def test_min_lr_less_than_lr():
    config = MemoraConfig.from_yaml("configs/tiny.yaml")
    config.training.min_lr = 1e-3
    config.training.lr = 1e-4

    from memora.config import TrainingConfig

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(config.training.model_dump())
