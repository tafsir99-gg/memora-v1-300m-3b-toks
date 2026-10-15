"""Data pipeline — preprocessing utilities.

Stages:
  1. split       – hash-based deterministic train/val split
  2. dedup       – exact-deduplication by SHA-256 of raw document text
  3. filter      – character-count proxy length filter (pre-tokenization)
  4. shard       – encode documents, write uint16 binary shards + manifest
  5. loader      – memory-mapped, position-resumable dataloader

CONSTRAINT: This module does NOT import ``datasets`` or ``pyarrow`` at the
top level. Those are only available in the ``data`` optional extra installed
on Kaggle. Any code path that uses them must be import-guarded. Tests use
plain-text fixture files and never touch the optional extra.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import struct
from collections.abc import Iterator

import numpy as np

from memora.tokenizer.wrapper import MemoraTokenizer

# ---------------------------------------------------------------------------
# 1. Hash-based train / val split
# ---------------------------------------------------------------------------

_SPLIT_HASH_BYTES = 4  # 32-bit hash gives uniform distribution with no collisions for <2^16 docs


def document_split(
    text: str,
    val_ratio: float,
    seed: int = 42,
) -> str:
    """Assign a document to 'train' or 'val' deterministically by hashing its text.

    The split is:
      - order-independent: result depends only on text, not processing order
      - reproducible: same inputs always give same output
      - disjoint: a doc is in exactly one split

    Parameters
    ----------
    text:
        The raw document text.
    val_ratio:
        Fraction of documents assigned to val (e.g., 0.005 for 0.5%).
    seed:
        Mixed into the hash to allow different splits.

    Returns
    -------
    'val' or 'train'
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    # Mix seed into the key so different seeds give different splits.
    h = document_hash(text)
    key = f"{seed}:{h}".encode()
    digest = hashlib.sha256(key).digest()[:_SPLIT_HASH_BYTES]
    # Interpret as big-endian unsigned int, normalise to [0, 1).
    value = int.from_bytes(digest, "big") / (2 ** (8 * _SPLIT_HASH_BYTES))
    return "val" if value < val_ratio else "train"


# ---------------------------------------------------------------------------
# 2. Exact deduplication
# ---------------------------------------------------------------------------


def document_hash(text: str) -> str:
    """SHA-256 hex digest of the document text (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ExactDeduplicator:
    """Track seen document hashes and report duplicates.

    Usage::

        dedup = ExactDeduplicator()
        for doc in documents:
            if dedup.is_duplicate(doc):
                continue
            # process doc
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        """Return True if this document has been seen before; register it otherwise."""
        h = document_hash(text)
        if h in self._seen:
            return True
        self._seen.add(h)
        return False

    def __len__(self) -> int:
        return len(self._seen)


# ---------------------------------------------------------------------------
# 3. Character-count length filter (pre-tokenization proxy)
# ---------------------------------------------------------------------------

# Per DATA_PLAN: reject docs with <200 chars (~50 tokens at 4 chars/token)
# or >400,000 chars (~100K tokens at 4 chars/token).
MIN_CHARS = 200
MAX_CHARS = 400_000


def passes_length_filter(text: str, min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS) -> bool:
    """Return True if the document passes the character-count length filter."""
    n = len(text)
    return min_chars <= n <= max_chars


# ---------------------------------------------------------------------------
# 4. Sharding — encode documents to uint16, write binary shards + manifest
# ---------------------------------------------------------------------------

# Shard file format (binary, little-endian):
#   header: 4 bytes = magic "MEMS" (0x4D454D53)
#           4 bytes = number of tokens (uint32)
#   body:   N * 2 bytes = token IDs (uint16, little-endian)
#
# uint16 is sufficient because all vocabs in this project are ≤65535.

SHARD_MAGIC = b"MEMS"
SHARD_TOKENS_PER_FILE = 10_000_000  # ~10M tokens per shard (~20 MB at uint16)


def _write_shard(
    token_ids: list[int],
    path: pathlib.Path,
) -> None:
    """Write a list of token IDs to a binary shard file."""
    n = len(token_ids)
    arr = np.array(token_ids, dtype=np.uint16)
    with open(path, "wb") as f:
        f.write(SHARD_MAGIC)
        f.write(struct.pack("<I", n))  # little-endian uint32
        arr.tofile(f)


def _read_shard(path: pathlib.Path) -> np.ndarray:
    """Read a shard file and return a numpy uint16 array of token IDs."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != SHARD_MAGIC:
            raise ValueError(f"Bad shard magic in {path}: {magic!r}")
        (n,) = struct.unpack("<I", f.read(4))
        arr = np.frombuffer(f.read(n * 2), dtype=np.uint16).copy()
    if len(arr) != n:
        raise ValueError(f"Shard {path}: expected {n} tokens, got {len(arr)}")
    return arr


def build_shards(
    documents: list[str],
    tokenizer: MemoraTokenizer,
    output_dir: pathlib.Path,
    split: str,
    tokens_per_shard: int = SHARD_TOKENS_PER_FILE,
    eos_id: int | None = None,
) -> dict:
    """Encode documents and write binary shards.

    Parameters
    ----------
    documents:
        Pre-filtered, pre-deduplicated document strings.
    tokenizer:
        Trained tokenizer.
    output_dir:
        Directory where shard files are written.
    split:
        'train' or 'val'. Used in shard filenames.
    tokens_per_shard:
        Target token count per shard file.
    eos_id:
        If not None, append this ID after each document.

    Returns
    -------
    Manifest dict with shard file info (see ``build_manifest``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _eos = eos_id if eos_id is not None else tokenizer.eos_id

    shard_index = 0
    buffer: list[int] = []
    shards_info: list[dict] = []
    total_tokens = 0
    total_docs = 0

    def _flush(buf: list[int], idx: int) -> None:
        nonlocal total_tokens
        fname = output_dir / f"{split}_{idx:04d}.bin"
        _write_shard(buf, fname)
        shards_info.append(
            {
                "file": fname.name,
                "tokens": len(buf),
                "sha256": hashlib.sha256(fname.read_bytes()).hexdigest(),
            }
        )
        total_tokens += len(buf)

    for doc in documents:
        ids = tokenizer.encode(doc, add_eos=False)
        ids.append(_eos)
        buffer.extend(ids)
        total_docs += 1

        if len(buffer) >= tokens_per_shard:
            _flush(buffer[:tokens_per_shard], shard_index)
            buffer = buffer[tokens_per_shard:]
            shard_index += 1

    if buffer:
        _flush(buffer, shard_index)

    return {
        "split": split,
        "num_shards": len(shards_info),
        "total_tokens": total_tokens,
        "total_docs": total_docs,
        "shards": shards_info,
    }


def write_manifest(manifest: dict, path: pathlib.Path) -> None:
    """Write the manifest dict as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def read_manifest(path: pathlib.Path) -> dict:
    """Read a manifest JSON file."""
    with open(path, encoding="utf-8") as f:
        result: dict = json.load(f)
    return result


# ---------------------------------------------------------------------------
# 5. Memory-mapped, position-resumable dataloader
# ---------------------------------------------------------------------------


class ShardedTokenLoader:
    """Load token IDs from binary shard files, with position resumption.

    Provides a continuous stream of overlapping ``seq_len``-length windows
    across all shards in order. The current position (shard index + offset)
    can be saved and restored for exact resumption after a crash or session end.

    Parameters
    ----------
    shard_dir:
        Directory containing shard files matching ``{split}_*.bin``.
    split:
        'train' or 'val'.
    seq_len:
        Sequence length (tokens per sample).
    rank:
        Process rank for distributed loading.
    world_size:
        Total number of processes for distributed loading.
    """

    def __init__(
        self,
        shard_dir: pathlib.Path,
        split: str,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.shard_dir = shard_dir
        self.split = split
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

        all_paths = sorted(shard_dir.glob(f"{split}_*.bin"))
        if not all_paths:
            raise FileNotFoundError(f"No shard files found for split '{split}' in {shard_dir}")

        self._shard_paths = all_paths[rank::world_size]
        if not self._shard_paths:
            raise ValueError(f"Not enough shards ({len(all_paths)}) for world_size {world_size}")

        self._shard_idx = 0
        self._offset = 0
        self._current: np.ndarray | None = None
        self._load_shard(self._shard_idx)

        # Pre-compute shard list hash for state_dict identity checks
        self._shard_list_hash = self._compute_shard_list_hash()

    @staticmethod
    def _compute_shard_list_hash_for(paths: list[pathlib.Path]) -> str:
        """SHA-256 over sorted (name, size) pairs of shard files."""
        h = hashlib.sha256()
        for p in sorted(paths):
            stat = p.stat()
            h.update(f"{p.name}:{stat.st_size}".encode())
        return h.hexdigest()[:16]

    def _compute_shard_list_hash(self) -> str:
        return self._compute_shard_list_hash_for(self._shard_paths)

    def _load_shard(self, idx: int) -> None:
        self._current = _read_shard(self._shard_paths[idx])
        self._shard_idx = idx
        # mmap: in production we'd use np.memmap; for correctness we use _read_shard here.

    def state_dict(self) -> dict:
        """Return self-describing state dict to resume.

        All values are plain int or str so the dict is JSON-serialisable.
        load_state_dict raises ValueError if any identity field differs.
        """
        return {
            "split": self.split,
            "seq_len": self.seq_len,
            "rank": self.rank,
            "world_size": self.world_size,
            "num_shards": len(self._shard_paths),
            "shard_list_hash": self._shard_list_hash,
            "shard_idx": self._shard_idx,
            "offset": self._offset,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore position from state dict.

        Raises ValueError if the identity fields don't match this loader.
        """
        mismatches = []
        identity_fields = {
            "split": self.split,
            "seq_len": self.seq_len,
            "rank": self.rank,
            "world_size": self.world_size,
            "num_shards": len(self._shard_paths),
            "shard_list_hash": self._shard_list_hash,
        }
        for field, current_val in identity_fields.items():
            saved_val = state.get(field)
            if saved_val != current_val:
                mismatches.append(f"{field}: saved={saved_val!r}, current={current_val!r}")
        if mismatches:
            raise ValueError(
                "Cannot restore state dict — loader identity mismatch:\n"
                + "\n".join(f"  {m}" for m in mismatches)
            )
        self._shard_idx = state["shard_idx"]
        self._offset = state["offset"]
        self._load_shard(self._shard_idx)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield windows of seq_len+1 tokens with stride seq_len.

        Each window is seq_len+1 tokens: tokens[0:seq_len] are inputs and
        tokens[1:seq_len+1] are targets (standard next-token layout).
        Consecutive windows share exactly one token: window[i][-1] == window[i+1][0].
        Iteration stops when the remaining shards cannot fill a full window.
        """
        assert self._current is not None
        while True:
            # If the current shard has a full window remaining, yield it.
            end = self._offset + self.seq_len + 1  # +1 for the label
            if end <= len(self._current):
                window = self._current[self._offset : end]
                self._offset += self.seq_len
                yield window
            else:
                # Move to next shard.
                next_idx = self._shard_idx + 1
                if next_idx >= len(self._shard_paths):
                    return  # exhausted all shards
                self._load_shard(next_idx)
                self._offset = 0
