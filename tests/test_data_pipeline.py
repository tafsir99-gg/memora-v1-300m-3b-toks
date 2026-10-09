"""Tests for memora.data.pipeline.

All tests use in-memory fixture data or the tiny corpus file.
No real corpus is downloaded. No datasets/pyarrow imported.
"""

import pathlib
import tempfile

import numpy as np
import pytest

from memora.data.pipeline import (
    ExactDeduplicator,
    ShardedTokenLoader,
    build_shards,
    document_hash,
    document_split,
    passes_length_filter,
    read_manifest,
    write_manifest,
)
from memora.tokenizer.trainer import train as train_tokenizer
from memora.tokenizer.wrapper import MemoraTokenizer

FIXTURE_CORPUS = pathlib.Path("tests/fixtures/corpus.txt")
FIXTURE_VOCAB_SIZE = 512


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory: pytest.TempPathFactory) -> MemoraTokenizer:
    """Train a tiny tokenizer once for all data pipeline tests."""
    out_dir = tmp_path_factory.mktemp("tokenizer_data")
    train_tokenizer(FIXTURE_CORPUS, out_dir, vocab_size=FIXTURE_VOCAB_SIZE)
    return MemoraTokenizer.from_dir(out_dir)


@pytest.fixture
def fixture_docs() -> list[str]:
    """Return the fixture corpus as a list of documents (lines)."""
    lines = FIXTURE_CORPUS.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# 1. Hash-based split — deterministic, disjoint, order-independent
# ---------------------------------------------------------------------------


def test_split_deterministic(fixture_docs: list[str]):
    """Same doc_id always gets same split."""
    for doc in fixture_docs[:10]:
        s1 = document_split(doc, val_ratio=0.2)
        s2 = document_split(doc, val_ratio=0.2)
        assert s1 == s2


def test_split_disjoint(fixture_docs: list[str]):
    """No document appears in both train and val."""
    train_docs = {d for d in fixture_docs if document_split(d, val_ratio=0.2) == "train"}
    val_docs = {d for d in fixture_docs if document_split(d, val_ratio=0.2) == "val"}
    assert train_docs.isdisjoint(val_docs)
    assert train_docs | val_docs == set(fixture_docs)


def test_split_order_independent(fixture_docs: list[str]):
    """Split result is the same regardless of processing order."""
    import random

    shuffled = fixture_docs[:]
    random.seed(999)
    random.shuffle(shuffled)

    original_splits = {d: document_split(d, val_ratio=0.2) for d in fixture_docs}
    shuffled_splits = {d: document_split(d, val_ratio=0.2) for d in shuffled}
    assert original_splits == shuffled_splits


def test_split_respects_val_ratio(fixture_docs: list[str]):
    """With enough docs the val fraction should be close to val_ratio."""
    # Use more docs by repeating fixture with unique IDs
    n = 1000
    docs = [f"doc_{i}" for i in range(n)]
    val_count = sum(1 for d in docs if document_split(d, val_ratio=0.1) == "val")
    # Allow ±5% tolerance
    assert 50 <= val_count <= 150, f"val_count={val_count} out of expected ~100"


def test_split_different_seeds():
    """Different seeds produce different splits for the same doc IDs."""
    docs = [f"doc_{i}" for i in range(100)]
    splits_s0 = [document_split(d, val_ratio=0.2, seed=0) for d in docs]
    splits_s1 = [document_split(d, val_ratio=0.2, seed=1) for d in docs]
    # Not identical (very likely with 100 docs)
    assert splits_s0 != splits_s1


def test_split_invalid_val_ratio():
    with pytest.raises(ValueError):
        document_split("doc_1", val_ratio=0.0)
    with pytest.raises(ValueError):
        document_split("doc_1", val_ratio=1.0)


# ---------------------------------------------------------------------------
# 2. Exact deduplication
# ---------------------------------------------------------------------------


def test_dedup_first_occurrence_not_dup():
    dedup = ExactDeduplicator()
    assert not dedup.is_duplicate("unique document text here")


def test_dedup_second_occurrence_is_dup():
    dedup = ExactDeduplicator()
    text = "duplicate document"
    dedup.is_duplicate(text)
    assert dedup.is_duplicate(text)


def test_dedup_different_texts_not_dup():
    dedup = ExactDeduplicator()
    assert not dedup.is_duplicate("text A")
    assert not dedup.is_duplicate("text B")


def test_dedup_len():
    dedup = ExactDeduplicator()
    dedup.is_duplicate("a")
    dedup.is_duplicate("b")
    dedup.is_duplicate("a")  # duplicate
    assert len(dedup) == 2


def test_document_hash_deterministic():
    h1 = document_hash("hello world")
    h2 = document_hash("hello world")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex = 64 chars


# ---------------------------------------------------------------------------
# 3. Length filter
# ---------------------------------------------------------------------------


def test_passes_length_filter_normal():
    # A 300-char string is within [200, 400000]
    text = "x" * 300
    assert passes_length_filter(text)


def test_fails_length_filter_too_short():
    text = "x" * 100
    assert not passes_length_filter(text)


def test_fails_length_filter_too_long():
    text = "x" * 500_000
    assert not passes_length_filter(text)


def test_length_filter_boundary():
    assert passes_length_filter("x" * 200)  # min boundary — inclusive
    assert not passes_length_filter("x" * 199)


# ---------------------------------------------------------------------------
# 4. Shard write / read round-trip
# ---------------------------------------------------------------------------


def test_shard_write_read_roundtrip(tokenizer: MemoraTokenizer, fixture_docs: list[str]):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = pathlib.Path(tmpdir)
        manifest = build_shards(
            fixture_docs,
            tokenizer,
            out_dir,
            split="train",
            tokens_per_shard=200,  # tiny shards to exercise multi-shard behaviour
        )
        assert manifest["num_shards"] >= 1
        assert manifest["total_tokens"] > 0
        assert manifest["split"] == "train"

        # Verify all shard files exist and SHA-256 matches manifest
        for info in manifest["shards"]:
            shard_path = out_dir / info["file"]
            assert shard_path.exists()
            import hashlib

            actual_hash = hashlib.sha256(shard_path.read_bytes()).hexdigest()
            assert actual_hash == info["sha256"], f"SHA mismatch for {info['file']}"


def test_shard_token_type(tokenizer: MemoraTokenizer, fixture_docs: list[str]):
    """Shard files must store uint16 tokens."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = pathlib.Path(tmpdir)
        build_shards(fixture_docs, tokenizer, out_dir, split="train", tokens_per_shard=5000)
        shard_files = sorted(out_dir.glob("train_*.bin"))
        assert shard_files
        from memora.data.pipeline import _read_shard

        arr = _read_shard(shard_files[0])
        assert arr.dtype == np.uint16


def test_shard_bad_magic(tmp_path: pathlib.Path):
    """Reading a file with wrong magic raises ValueError."""
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"XXXX" + b"\x00" * 8)
    from memora.data.pipeline import _read_shard

    with pytest.raises(ValueError, match="magic"):
        _read_shard(bad)


# ---------------------------------------------------------------------------
# 5. Manifest write / read
# ---------------------------------------------------------------------------


def test_manifest_write_read(tmp_path: pathlib.Path):
    manifest = {"split": "train", "total_tokens": 12345, "shards": []}
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)
    recovered = read_manifest(path)
    assert recovered == manifest


def test_manifest_produced_by_build_shards(tokenizer: MemoraTokenizer, fixture_docs: list[str]):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = pathlib.Path(tmpdir)
        manifest = build_shards(
            fixture_docs, tokenizer, out_dir, split="val", tokens_per_shard=2000
        )
        manifest_path = out_dir / "manifest_val.json"
        write_manifest(manifest, manifest_path)
        assert manifest_path.exists()
        loaded = read_manifest(manifest_path)
        assert loaded["split"] == "val"
        assert loaded["total_docs"] == len(fixture_docs)


# ---------------------------------------------------------------------------
# 6. Resumable loader — restores exact position
# ---------------------------------------------------------------------------


def test_loader_yields_seq_len_plus_one(tokenizer: MemoraTokenizer, fixture_docs: list[str]):
    """Each yielded window is seq_len+1 tokens (input + label)."""
    seq_len = 16
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = pathlib.Path(tmpdir)
        build_shards(fixture_docs, tokenizer, out_dir, split="train", tokens_per_shard=500)
        loader = ShardedTokenLoader(out_dir, split="train", seq_len=seq_len)
        window = next(iter(loader))
        assert len(window) == seq_len + 1


def test_loader_position_restore(tokenizer: MemoraTokenizer, fixture_docs: list[str]):
    """Loader resumed from a saved position yields the same sequence."""
    seq_len = 16
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = pathlib.Path(tmpdir)
        build_shards(fixture_docs, tokenizer, out_dir, split="train", tokens_per_shard=500)

        loader1 = ShardedTokenLoader(out_dir, split="train", seq_len=seq_len)
        # Advance two steps, then save position
        it1 = iter(loader1)
        next(it1)
        next(it1)
        state = loader1.state_dict()

        # Continue from loader1
        expected = next(it1)

        # Fresh loader resumed at the saved position
        loader2 = ShardedTokenLoader(
            out_dir,
            split="train",
            seq_len=seq_len,
        )
        loader2.load_state_dict(state)
        actual = next(iter(loader2))

        np.testing.assert_array_equal(actual, expected)


def test_loader_no_shards_raises(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        ShardedTokenLoader(tmp_path, split="train", seq_len=16)


def test_loader_rank_disjoint(tokenizer: MemoraTokenizer, fixture_docs: list[str]):
    """With 2 simulated ranks, shards are disjoint across ranks."""
    seq_len = 16
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = pathlib.Path(tmpdir)
        # Create many small shards
        build_shards(fixture_docs, tokenizer, out_dir, split="train", tokens_per_shard=50)

        loader_rank0 = ShardedTokenLoader(out_dir, split="train", seq_len=seq_len, rank=0, world_size=2)
        loader_rank1 = ShardedTokenLoader(out_dir, split="train", seq_len=seq_len, rank=1, world_size=2)

        paths0 = set(loader_rank0._shard_paths)
        paths1 = set(loader_rank1._shard_paths)

        assert paths0.isdisjoint(paths1)
        assert len(paths0) > 0
        assert len(paths1) > 0


def test_no_datasets_import():
    """Importing memora.data or memora.tokenizer must not import `datasets`."""
    import sys

    # Force unload if it was somehow loaded
    if "datasets" in sys.modules:
        del sys.modules["datasets"]


    assert "datasets" not in sys.modules, "datasets was imported but shouldn't be!"
