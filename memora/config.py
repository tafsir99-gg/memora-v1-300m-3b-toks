import pathlib
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    d_model: int = Field(default=960, gt=0)
    n_layers: int = Field(default=24, gt=0)
    n_heads: int = Field(default=15, gt=0)
    d_ff: int = Field(default=3840, gt=0)
    vocab_size: int = Field(default=32768, ge=256)
    max_seq_len: int = Field(default=1024, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    use_bias: bool = False
    tie_embeddings: bool = True
    norm_eps: float = Field(default=1e-5, gt=0)
    rope_base: float = Field(default=10000.0, gt=0)
    init_std: float = Field(default=0.02, gt=0)

    @model_validator(mode="after")
    def validate_dimensions(self) -> "ModelConfig":
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self


class TrainingConfig(BaseModel):
    optimizer: Literal["adamw"] = "adamw"
    lr: float = Field(default=6e-4, gt=0)
    min_lr: float = Field(default=6e-5, gt=0)
    warmup_tokens: int = Field(default=131_072_000, ge=0)
    lr_schedule: Literal["cosine", "wsd"] = "cosine"
    wsd_decay_tokens: int = Field(default=100_000_000, gt=0)
    max_tokens: int = Field(default=3_000_000_000, gt=0)
    weight_decay: float = Field(default=0.1, ge=0.0)
    beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    beta2: float = Field(default=0.95, gt=0.0, lt=1.0)
    eps: float = Field(default=1e-8, gt=0)
    grad_clip: float = Field(default=1.0, gt=0)
    global_batch_tokens: int = Field(default=524_288, gt=0)
    micro_batch_size: int = Field(default=16, ge=1)
    seq_len: int = Field(default=1024, ge=1)
    precision: Literal["fp16", "fp32"] = "fp16"
    compile: bool = False
    seed: int = 42

    @model_validator(mode="after")
    def validate_training(self) -> "TrainingConfig":
        if self.min_lr > self.lr:
            raise ValueError(f"min_lr ({self.min_lr}) cannot be greater than lr ({self.lr})")
        micro_tokens = self.micro_batch_size * self.seq_len
        if self.global_batch_tokens % micro_tokens != 0:
            raise ValueError(
                f"global_batch_tokens ({self.global_batch_tokens}) must be a multiple of "
                f"micro_batch_size * seq_len ({micro_tokens})"
            )
        return self


class CheckpointConfig(BaseModel):
    dir: str = "checkpoints/"
    save_interval_steps: int = Field(default=500, ge=1)
    keep_last_n: int = Field(default=3, ge=1)
    save_optimizer: bool = True
    lightweight_interval_steps: int = Field(default=100, ge=1)


class DataConfig(BaseModel):
    dataset_name: str = "fineweb-edu-sample"
    tokenizer_path: str = "tokenizer/"
    shard_dir: str = "data/shards/"
    val_ratio: float = Field(default=0.005, ge=0.0, lt=1.0)
    num_workers: int = Field(default=4, ge=0)
    seed: int = 42


class EvalConfig(BaseModel):
    val_interval_steps: int = Field(default=500, ge=1)
    val_tokens: int = Field(default=10_000_000, gt=0)
    sample_interval_steps: int = Field(default=1000, ge=1)
    sample_prompts: list[str] = Field(
        default_factory=lambda: [
            "The meaning of life is",
            "In a groundbreaking discovery, scientists",
            "Once upon a time in a land far away",
        ]
    )
    sample_temperature: float = Field(default=0.8, gt=0.0)
    sample_top_k: int = Field(default=50, ge=1)
    sample_max_tokens: int = Field(default=128, ge=1)


class LoggingConfig(BaseModel):
    log_interval_steps: int = Field(default=10, ge=1)
    log_dir: str = "runs/"
    format: Literal["jsonl"] = "jsonl"


class DistributedConfig(BaseModel):
    backend: Literal["nccl", "gloo"] = "nccl"
    strategy: Literal["ddp"] = "ddp"


class MemoraConfig(BaseModel):
    """Top-level config combining all sections."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)

    @model_validator(mode="after")
    def validate_cross_section(self) -> "MemoraConfig":
        if self.training.seq_len != self.model.max_seq_len:
            raise ValueError(
                f"training.seq_len ({self.training.seq_len}) must match "
                f"model.max_seq_len ({self.model.max_seq_len})"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> "MemoraConfig":
        """Load and validate config from a YAML file."""
        with open(path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            config_dict = {}

        return cls.model_validate(config_dict)
