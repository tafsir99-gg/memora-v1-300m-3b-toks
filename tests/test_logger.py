"""Tests for memora.training.logger (Stage 7a)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memora.training.logger import TrainingLogger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_path(tmp_path: Path, name: str = "train.jsonl") -> Path:
    return tmp_path / name


def _parse_lines(path: Path) -> list[dict]:
    """Read all lines from a JSONL file and parse as strict JSON."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            lines.append(json.loads(line))  # strict JSON — raises on NaN
    return lines


# ---------------------------------------------------------------------------
# Basic logging
# ---------------------------------------------------------------------------


def test_logger_creates_file(tmp_path: Path):
    path = _log_path(tmp_path)
    with TrainingLogger(path) as logger:
        logger.log(
            "train",
            step=1,
            tokens_seen=1024,
            train_loss=3.5,
            lr=6e-4,
            grad_norm=0.5,
            step_time=60.0,
            tokens_per_sec=8000.0,
        )
    assert path.exists()


def test_logger_valid_json(tmp_path: Path):
    path = _log_path(tmp_path)
    with TrainingLogger(path) as logger:
        logger.log(
            "train",
            step=1,
            tokens_seen=1024,
            train_loss=3.5,
            lr=6e-4,
            grad_norm=0.5,
            step_time=60.0,
            tokens_per_sec=8000.0,
        )
    records = _parse_lines(path)
    assert len(records) == 1
    assert records[0]["event"] == "train"
    assert records[0]["step"] == 1


def test_logger_all_event_types(tmp_path: Path):
    """All supported event types log without error and parse as valid JSON."""
    path = _log_path(tmp_path)
    with TrainingLogger(path) as logger:
        logger.log(
            "session_start",
            step=0,
            tokens_seen=0,
            session_id="abc123",
            wall_clock_seconds=0.0,
        )
        logger.log(
            "train",
            step=1,
            tokens_seen=524288,
            train_loss=4.2,
            lr=3e-4,
            grad_norm=1.0,
            step_time=72.0,
            tokens_per_sec=7280.0,
        )
        logger.log(
            "val",
            step=1,
            tokens_seen=524288,
            val_loss=4.0,
            perplexity=54.6,
        )
        logger.log(
            "ckpt",
            step=1,
            tokens_seen=524288,
            checkpoint_seconds=12.5,
            checkpoint_path="checkpoints/ckpt_0001.pt",
        )
        logger.log(
            "sample",
            step=1,
            tokens_seen=524288,
            prompt="Once upon",
            text="Once upon a time",
        )
        logger.log(
            "session_end",
            step=1,
            tokens_seen=524288,
            session_id="abc123",
            wall_clock_seconds=100.0,
            cumulative_compute_seconds=99.0,
        )
    records = _parse_lines(path)
    assert len(records) == 6
    event_types = [r["event"] for r in records]
    assert event_types == [
        "session_start", "train", "val", "ckpt", "sample", "session_end"
    ]


def test_logger_missing_field_raises(tmp_path: Path):
    path = _log_path(tmp_path)
    with TrainingLogger(path) as logger, pytest.raises(ValueError, match="missing required fields"):
        logger.log(
            "train",
            step=1,
            tokens_seen=1024,
            # train_loss missing
            lr=6e-4,
            grad_norm=0.5,
            step_time=60.0,
            tokens_per_sec=8000.0,
        )


def test_logger_unknown_event_raises(tmp_path: Path):
    path = _log_path(tmp_path)
    with TrainingLogger(path) as logger, pytest.raises(ValueError, match="Unknown event type"):
        logger.log("bogus_event", step=1, tokens_seen=1)


# ---------------------------------------------------------------------------
# NaN / Inf handling
# ---------------------------------------------------------------------------


def test_logger_nan_converted_to_null(tmp_path: Path):
    """NaN values are replaced with null and recorded in 'nonfinite'."""
    path = _log_path(tmp_path)
    with TrainingLogger(path) as logger:
        logger.log(
            "train",
            step=1,
            tokens_seen=1024,
            train_loss=float("nan"),
            lr=6e-4,
            grad_norm=0.5,
            step_time=60.0,
            tokens_per_sec=8000.0,
        )
    records = _parse_lines(path)
    assert records[0]["train_loss"] is None
    assert "nonfinite" in records[0]
    assert "train_loss" in records[0]["nonfinite"]


def test_logger_inf_converted_to_null(tmp_path: Path):
    path = _log_path(tmp_path)
    with TrainingLogger(path) as logger:
        logger.log(
            "train",
            step=1,
            tokens_seen=1024,
            train_loss=float("inf"),
            lr=6e-4,
            grad_norm=float("-inf"),
            step_time=60.0,
            tokens_per_sec=8000.0,
        )
    records = _parse_lines(path)
    assert records[0]["train_loss"] is None
    assert records[0]["grad_norm"] is None


def test_logger_json_parseable_with_nan(tmp_path: Path):
    """The raw file content must be strict-JSON parseable (no NaN literals)."""
    path = _log_path(tmp_path)
    with TrainingLogger(path) as logger:
        logger.log(
            "train",
            step=1,
            tokens_seen=1024,
            train_loss=float("nan"),
            lr=6e-4,
            grad_norm=0.5,
            step_time=60.0,
            tokens_per_sec=8000.0,
        )
    raw = path.read_text()
    # json.loads raises on NaN literals — must succeed
    for line in raw.splitlines():
        if line.strip():
            json.loads(line)


# ---------------------------------------------------------------------------
# Append across two "sessions"
# ---------------------------------------------------------------------------


def test_logger_append_across_sessions(tmp_path: Path):
    """Two logger instances appending to the same file accumulate lines."""
    path = _log_path(tmp_path)
    with TrainingLogger(path) as logger1:
        logger1.log(
            "train",
            step=1,
            tokens_seen=1024,
            train_loss=3.5,
            lr=6e-4,
            grad_norm=0.5,
            step_time=60.0,
            tokens_per_sec=8000.0,
        )

    with TrainingLogger(path) as logger2:
        logger2.log(
            "train",
            step=2,
            tokens_seen=2048,
            train_loss=3.2,
            lr=6e-4,
            grad_norm=0.4,
            step_time=58.0,
            tokens_per_sec=9000.0,
        )

    records = _parse_lines(path)
    assert len(records) == 2
    assert records[0]["step"] == 1
    assert records[1]["step"] == 2


# ---------------------------------------------------------------------------
# Rank != 0: no-op
# ---------------------------------------------------------------------------


def test_logger_noop_for_nonzero_rank(tmp_path: Path):
    path = _log_path(tmp_path)
    with TrainingLogger(path, rank=1) as logger:
        logger.log(
            "train",
            step=1,
            tokens_seen=1024,
            train_loss=3.5,
            lr=6e-4,
            grad_norm=0.5,
            step_time=60.0,
            tokens_per_sec=8000.0,
        )
    assert not path.exists()


# ---------------------------------------------------------------------------
# Parent directory created automatically
# ---------------------------------------------------------------------------


def test_logger_creates_parent_dirs(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "c" / "train.jsonl"
    with TrainingLogger(nested) as logger:
        logger.log(
            "train",
            step=1,
            tokens_seen=1024,
            train_loss=3.5,
            lr=6e-4,
            grad_norm=0.5,
            step_time=60.0,
            tokens_per_sec=8000.0,
        )
    assert nested.exists()
