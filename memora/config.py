import pathlib
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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
        head_dim = self.d_model // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be even (required for RoPE sin/cos split)"
            )
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    optimizer: Literal["adamw"] = "adamw"
    lr: float = Field(default=6e-4, gt=0)
    min_lr: float = Field(default=6e-5, gt=0)
    warmup_tokens: int = Field(default=131_072_000, ge=0)
    lr_schedule: Literal["cosine", "wsd"] = "cosine"
    wsd_decay_tokens: int = Field(default=100_000_000, gt=0)
    wsd_decay_start_tokens: int | None = Field(
        default=None,
        description=(
            "Token count at which WSD decay begins. "
            "None means max_tokens - wsd_decay_tokens. "
            "Must satisfy warmup_tokens <= start and start + wsd_decay_tokens <= max_tokens."
        ),
    )
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
        if self.warmup_tokens >= self.max_tokens:
            raise ValueError(
                f"warmup_tokens ({self.warmup_tokens}) must be < max_tokens ({self.max_tokens})"
            )
        if self.lr_schedule == "wsd":
            start = (
                self.wsd_decay_start_tokens
                if self.wsd_decay_start_tokens is not None
                else self.max_tokens - self.wsd_decay_tokens
            )
            if start < self.warmup_tokens:
                raise ValueError(
                    f"For wsd schedule: wsd_decay_start_tokens ({start}) must be "
                    f">= warmup_tokens ({self.warmup_tokens})"
                )
            if start + self.wsd_decay_tokens > self.max_tokens:
                raise ValueError(
                    f"For wsd schedule: wsd_decay_start_tokens ({start}) + "
                    f"wsd_decay_tokens ({self.wsd_decay_tokens}) = {start + self.wsd_decay_tokens} "
                    f"exceeds max_tokens ({self.max_tokens})"
                )
        return self

    def grad_accum_steps(self, world_size: int) -> int:
        """Return gradient accumulation steps for a given DDP world size.

        Raises ValueError if global_batch_tokens is not divisible by
        micro_batch_size * seq_len * world_size (would cause uneven gradient accumulation).
        """
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        per_gpu_tokens = self.micro_batch_size * self.seq_len
        total_micro_tokens = per_gpu_tokens * world_size
        if self.global_batch_tokens % total_micro_tokens != 0:
            raise ValueError(
                f"global_batch_tokens ({self.global_batch_tokens}) is not divisible by "
                f"micro_batch_size * seq_len * world_size "
                f"({self.micro_batch_size} * {self.seq_len} * {world_size} = {total_micro_tokens})"
            )
        return self.global_batch_tokens // total_micro_tokens


class CheckpointConfig(BaseModel):
    """Checkpoint retention and scheduling.

    Retention policy (per the plan):
      - keep_full_n: resumable checkpoints (model + optimizer + scaler + RNG state)
      - keep_lightweight_n: weights-only snapshots for quick inspection
      - keep_best: keep the checkpoint with best val loss regardless of age

    Full checkpoints ALWAYS include optimizer, scaler, and RNG state.
    A checkpoint without these cannot be resumed and counts as invalid.

    Scheduling (wall-clock based, not step-based):
      At 524,288 tokens/step and ~15–30% MFU on one T4 (ESTIMATE from plan throughput tables):
        tokens/s ≈ 15–30% × 65e12 FLOPS / (6 × 297e6 params) ≈ 5,500–11,000 tok/s
        step time ≈ 524,288 / (5,500–11,000) ≈ 48–95 seconds per step
      Therefore a step-based interval of 500 steps = 6–13 hours between checkpoints
      (contradicts R6: "worst case lose ~5 min of training").

      Wall-clock intervals used instead:
        save_full_interval_minutes=30  → full checkpoint every ~30 min
        save_lightweight_interval_minutes=10  → lightweight every ~10 min
        hf_upload_interval_minutes=120 → upload to HF Hub every ~120 min
          Each upload writes ~3.6 GB of LFS history; at 120 min: ~5-6 uploads/session.
          squash_history (HfApi.super_squash_history) must be run after each session's
          final upload to prevent unbounded LFS growth.

      Session budget:
        session.max_wall_clock_minutes=690  → 11.5h of a 12h Kaggle session
        session.shutdown_grace_minutes=10   → reserved for orderly checkpoint + upload

      Disk worst-case peak during save (2 full + 1 new + 3 lightweight + 1 best):
        2 full × 3.6 GB + 1 new 3.6 GB + 3 × 1.2 GB lightweight + 1 best 3.6 GB ≈ 18.0 GB
        Kaggle output quota: 20 GB. Check shutil.disk_usage before writing.

      These are ESTIMATES. Actual step rate measured in Phase 6 benchmark.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dir: str = "checkpoints/"
    save_full_interval_minutes: int = Field(
        default=30,
        ge=1,
        description=(
            "Save a full (resumable) checkpoint every N minutes. "
            "ESTIMATE: at ~48-95s/step on 1xT4, this is roughly every 19-37 steps."
        ),
    )
    save_lightweight_interval_minutes: int = Field(
        default=10,
        ge=1,
        description="Save weights-only snapshot every N minutes.",
    )
    hf_upload_interval_minutes: int = Field(
        default=120,
        ge=1,
        description=(
            "Upload latest full checkpoint to HF Hub every N minutes. "
            "Each upload writes ~3.6 GB of LFS history; at 120 min this is ~5-6 per session. "
            "squash_history (HfApi.super_squash_history) must run after each session's final "
            "upload to prevent unbounded LFS history growth."
        ),
    )
    keep_full_n: int = Field(
        default=2,
        ge=1,
        description="Number of full (resumable) checkpoints to retain on disk.",
    )
    keep_lightweight_n: int = Field(
        default=3,
        ge=1,
        description="Number of lightweight (weights-only) snapshots to retain on disk.",
    )
    keep_best: bool = Field(
        default=True,
        description="Always retain the checkpoint with the best validation loss.",
    )


class SessionConfig(BaseModel):
    """Kaggle session time budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_wall_clock_minutes: int = Field(
        default=690,
        ge=1,
        description=(
            "Soft limit on total session run time (minutes). "
            "Kaggle sessions cap at 12h = 720 min; leave 30 min for setup + teardown."
        ),
    )
    shutdown_grace_minutes: int = Field(
        default=10,
        ge=1,
        description=(
            "Minutes reserved at end of session for checkpointing and HF Hub upload "
            "before the session expires."
        ),
    )


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str = "fineweb-edu-sample"
    tokenizer_path: str = "tokenizer/"
    shard_dir: str = "data/shards/"
    val_ratio: float = Field(default=0.005, gt=0.0, lt=1.0)
    num_workers: int = Field(default=4, ge=0)
    seed: int = 42


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    val_interval_minutes: int = Field(
        default=60,
        ge=1,
        description=(
            "Run validation every N minutes. "
            "Arithmetic (ESTIMATE at 15-30% MFU, 1xT4): "
            "  Forward pass: ~N FLOPs/token (vs 6N for forward+backward). "
            "  val_tokens=2M, tokens/s≈5,500-11,000 → val takes ~3-6 min. "
            "  At 60-min interval that is ~5-10% of GPU time (acceptable). "
            "Full-set validation is for milestone checkpoints only."
        ),
    )
    val_tokens: int = Field(
        default=2_000_000,
        gt=0,
        description=(
            "Tokens evaluated per validation run. "
            "2M tokens at ~5,500-11,000 tok/s ≈ 3-6 min on one T4 (ESTIMATE)."
        ),
    )
    sample_interval_minutes: int = Field(
        default=60,
        ge=1,
        description="Generate text samples for qualitative tracking every N minutes.",
    )
    sample_prompts: list[str] = Field(
        default_factory=lambda: [
            "The meaning of life is",
            "In a groundbreaking discovery, scientists",
            "Once upon a time in a land far away",
        ]
    )
    sample_temperature: float = Field(default=0.8, gt=0.0)
    sample_top_k: int = Field(default=50, ge=1)
    sample_top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    sample_max_tokens: int = Field(default=128, ge=1)

    @model_validator(mode="after")
    def validate_eval(self) -> "EvalConfig":
        if self.sample_top_p <= 0.0:
            raise ValueError("sample_top_p must be > 0")
        return self


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    log_interval_steps: int = Field(
        default=1,
        ge=1,
        description=(
            "Log training metrics every N steps. "
            "At ~48-95s/step on 1xT4, 10 steps = 8-16 min blind spot; default=1."
        ),
    )
    log_dir: str = "runs/"
    format: Literal["jsonl"] = "jsonl"


class DistributedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["nccl", "gloo"] = "nccl"
    strategy: Literal["ddp"] = "ddp"


class MemoraConfig(BaseModel):
    """Top-level config combining all sections."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
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

    @classmethod
    def with_overrides(cls, base: "MemoraConfig", overrides: dict) -> "MemoraConfig":
        """Return a new MemoraConfig with nested dict overrides applied, running all validators.

        ``overrides`` is a shallow or nested dict in the same shape as the YAML config.
        Validators run exactly as they do on initial load, so an invalid override
        (e.g. micro_batch_size=7) raises ValidationError.

        Use this instead of model_copy(update=...) for top-level changes because
        model_copy bypasses cross-section validators.
        """
        base_dict = base.model_dump()
        # Deep-merge overrides into base_dict
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(base_dict.get(key), dict):
                base_dict[key] = {**base_dict[key], **value}
            else:
                base_dict[key] = value
        return cls.model_validate(base_dict)

    def check_resume_compat(self, current: "MemoraConfig") -> list[str]:
        """Check compatibility between a saved config (self) and the current config.

        Returns a list of human-readable difference strings.
        Raises ValueError for must-match fields that differ.

        Must-match fields (changing breaks reproducibility):
          model config, global_batch_tokens, seq_len, lr, min_lr, warmup_tokens,
          betas, weight_decay, data.seed, data.val_ratio, data.dataset_name.
          world_size differences are detected at runtime by comparing loader state.

        May differ (logged but not fatal):
          max_tokens, lr_schedule, wsd_*, micro_batch_size, checkpoint.*, evaluation.*,
          logging.*, session.*, distributed.*, training.seed, training.compile.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Must match: model config (all fields)
        saved_model = self.model.model_dump()
        curr_model = current.model.model_dump()
        for field, saved_val in saved_model.items():
            curr_val = curr_model[field]
            if saved_val != curr_val:
                errors.append(f"model.{field}: saved={saved_val!r}, current={curr_val!r}")

        # Must match: training hyperparameters
        must_match_training = [
            "global_batch_tokens", "seq_len", "lr", "min_lr",
            "warmup_tokens", "beta1", "beta2", "weight_decay",
        ]
        for field in must_match_training:
            sv = getattr(self.training, field)
            cv = getattr(current.training, field)
            if sv != cv:
                errors.append(f"training.{field}: saved={sv!r}, current={cv!r}")

        # Must match: data settings that affect train/val split
        if self.data.seed != current.data.seed:
            errors.append(
                f"data.seed: saved={self.data.seed!r}, current={current.data.seed!r}"
            )
        if self.data.val_ratio != current.data.val_ratio:
            errors.append(
                f"data.val_ratio: saved={self.data.val_ratio!r}, current={current.data.val_ratio!r}"
            )
        if self.data.dataset_name != current.data.dataset_name:
            errors.append(
                f"data.dataset_name: saved={self.data.dataset_name!r}, "
                f"current={current.data.dataset_name!r}"
            )

        if errors:
            raise ValueError(
                "Resume incompatible — the following must-match fields differ:\n"
                + "\n".join(f"  {e}" for e in errors)
            )

        # May differ: log differences only
        may_differ_training = [
            "max_tokens", "lr_schedule", "wsd_decay_tokens", "wsd_decay_start_tokens",
            "micro_batch_size", "compile", "seed", "grad_clip", "precision",
        ]
        for field in may_differ_training:
            sv = getattr(self.training, field)
            cv = getattr(current.training, field)
            if sv != cv:
                warnings.append(f"training.{field}: saved={sv!r}, current={cv!r} (allowed)")

        return warnings

