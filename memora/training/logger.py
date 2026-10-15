"""JSONL training logger — rank-0 only, append mode.

Each log event is one JSON object per line, flushed immediately.
NaN and Inf values are converted to null with the source field name recorded
in a ``nonfinite`` list, so every line is strict-JSON valid.

Event types and required fields
--------------------------------
session_start:
    step, tokens_seen, session_id, wall_clock_seconds

session_end:
    step, tokens_seen, session_id, wall_clock_seconds, cumulative_compute_seconds

train:
    step, tokens_seen, train_loss, lr, grad_norm, step_time, tokens_per_sec

val:
    step, tokens_seen, val_loss, perplexity

ckpt:
    step, tokens_seen, checkpoint_seconds, checkpoint_path

sample:
    step, tokens_seen, prompt, text
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

# Fields that must appear in every event of a given type.
# Values map to a "validator" callable or None (presence-only check).
_REQUIRED_FIELDS: dict[str, list[str]] = {
    "session_start": ["step", "tokens_seen", "session_id", "wall_clock_seconds"],
    "session_end": [
        "step",
        "tokens_seen",
        "session_id",
        "wall_clock_seconds",
        "cumulative_compute_seconds",
    ],
    "train": ["step", "tokens_seen", "train_loss", "lr", "grad_norm", "step_time", "tokens_per_sec"],
    "val": ["step", "tokens_seen", "val_loss", "perplexity"],
    "ckpt": ["step", "tokens_seen", "checkpoint_seconds", "checkpoint_path"],
    "sample": ["step", "tokens_seen", "prompt", "text"],
}


def _sanitize(obj: Any, nonfinite: list[str], prefix: str = "") -> Any:
    """Recursively replace NaN/Inf with None and record field names."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            nonfinite.append(prefix)
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v, nonfinite, f"{prefix}.{k}" if prefix else k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, nonfinite, f"{prefix}[{i}]") for i, v in enumerate(obj)]
    return obj


class TrainingLogger:
    """JSONL event logger for training.

    Parameters
    ----------
    log_path:
        Path to the .jsonl log file. Opened in append mode; the parent
        directory is created if it doesn't exist.
    rank:
        Process rank. Logging is a no-op when rank != 0.
    """

    def __init__(self, log_path: Path | str, rank: int = 0) -> None:
        self._rank = rank
        self._path = Path(log_path)
        self._file = None
        if rank == 0:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, event_type: str, **fields: Any) -> None:
        """Append a validated JSON event to the log.

        Parameters
        ----------
        event_type:
            One of the supported event types (see module docstring).
        **fields:
            Event payload. ``wall_clock_seconds`` is added automatically
            if not provided.

        Raises
        ------
        ValueError
            If a required field is missing for the given event type.
        """
        if self._rank != 0:
            return

        required = _REQUIRED_FIELDS.get(event_type)
        if required is None:
            raise ValueError(
                f"Unknown event type {event_type!r}. "
                f"Valid types: {sorted(_REQUIRED_FIELDS)}"
            )

        missing = [f for f in required if f not in fields]
        if missing:
            raise ValueError(
                f"Event type {event_type!r} is missing required fields: {missing}"
            )

        record: dict[str, Any] = {"event": event_type, "ts": time.time()}
        record.update(fields)

        nonfinite: list[str] = []
        clean = _sanitize(record, nonfinite)
        if nonfinite:
            clean["nonfinite"] = nonfinite  # type: ignore[index]

        self._write(clean)

    def close(self) -> None:
        """Flush and close the log file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, record: dict[str, Any]) -> None:
        assert self._file is not None
        line = json.dumps(record, allow_nan=False)
        self._file.write(line + os.linesep)
        self._file.flush()
