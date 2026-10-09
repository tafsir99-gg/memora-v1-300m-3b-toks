"""Tests for memora.tokenizer (trainer + wrapper).

All tests use the tiny fixture corpus (tests/fixtures/corpus.txt).
No real corpus is downloaded. The tokenizer is trained with a small
vocab (512) to keep test time under a second.
"""

import hashlib
import pathlib
import tempfile

import pytest

from memora.tokenizer.trainer import (
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
    tokenizer_sha256,
    train,
)
from memora.tokenizer.wrapper import MemoraTokenizer

FIXTURE_CORPUS = pathlib.Path("tests/fixtures/corpus.txt")
FIXTURE_VOCAB_SIZE = 512  # tiny; keeps training < 1s on Mac


@pytest.fixture(scope="module")
def trained_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Train a tokenizer on the fixture corpus once per test session."""
    out_dir = tmp_path_factory.mktemp("tokenizer")
    train(FIXTURE_CORPUS, out_dir, vocab_size=FIXTURE_VOCAB_SIZE)
    return out_dir


# ---------------------------------------------------------------------------
# Training output
# ---------------------------------------------------------------------------


def test_tokenizer_json_exists(trained_tokenizer_dir: pathlib.Path):
    assert (trained_tokenizer_dir / "tokenizer.json").exists()


def test_tokenizer_config_json_exists(trained_tokenizer_dir: pathlib.Path):
    assert (trained_tokenizer_dir / "tokenizer_config.json").exists()


# ---------------------------------------------------------------------------
# Loading without training code
# ---------------------------------------------------------------------------


def test_loads_from_json_file(trained_tokenizer_dir: pathlib.Path):
    tok = MemoraTokenizer(trained_tokenizer_dir / "tokenizer.json")
    assert tok.vocab_size > 0


def test_loads_from_dir(trained_tokenizer_dir: pathlib.Path):
    tok = MemoraTokenizer.from_dir(trained_tokenizer_dir)
    assert tok.vocab_size > 0


# ---------------------------------------------------------------------------
# Special token IDs are stable
# ---------------------------------------------------------------------------


def test_special_token_ids(trained_tokenizer_dir: pathlib.Path):
    tok = MemoraTokenizer.from_dir(trained_tokenizer_dir)
    assert tok.pad_id == PAD_ID == 0
    assert tok.eos_id == EOS_ID == 1
    assert tok.unk_id == UNK_ID == 2


def test_special_token_strings():
    assert PAD_TOKEN == "<|pad|>"
    assert EOS_TOKEN == "<|eos|>"
    assert UNK_TOKEN == "<|unk|>"


# ---------------------------------------------------------------------------
# Encode / decode round-trip
# ---------------------------------------------------------------------------


def test_encode_returns_list_of_ints(trained_tokenizer_dir: pathlib.Path):
    tok = MemoraTokenizer.from_dir(trained_tokenizer_dir)
    ids = tok.encode("The quick brown fox")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_decode_roundtrip(trained_tokenizer_dir: pathlib.Path):
    tok = MemoraTokenizer.from_dir(trained_tokenizer_dir)
    text = "Education is the foundation of a prosperous society"
    ids = tok.encode(text)
    recovered = tok.decode(ids)
    assert recovered == text


def test_encode_add_eos(trained_tokenizer_dir: pathlib.Path):
    tok = MemoraTokenizer.from_dir(trained_tokenizer_dir)
    ids = tok.encode("Hello world", add_eos=True)
    assert ids[-1] == EOS_ID


def test_decode_skip_special_tokens(trained_tokenizer_dir: pathlib.Path):
    tok = MemoraTokenizer.from_dir(trained_tokenizer_dir)
    ids = tok.encode("Hello", add_eos=True)
    # With skip_special_tokens=True, EOS should not appear
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert EOS_TOKEN not in decoded


def test_decode_keep_special_tokens(trained_tokenizer_dir: pathlib.Path):
    tok = MemoraTokenizer.from_dir(trained_tokenizer_dir)
    ids = [PAD_ID, *tok.encode("Hi"), EOS_ID]
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert PAD_TOKEN in decoded or EOS_TOKEN in decoded


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_encode_determinism(trained_tokenizer_dir: pathlib.Path):
    """Same input always gives same ids."""
    tok = MemoraTokenizer.from_dir(trained_tokenizer_dir)
    text = "Deep learning architectures with many layers"
    assert tok.encode(text) == tok.encode(text)


def test_training_determinism():
    """Training twice on the same corpus with the same params gives the same tokenizer hash."""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        path1 = train(FIXTURE_CORPUS, d1, vocab_size=FIXTURE_VOCAB_SIZE)
        path2 = train(FIXTURE_CORPUS, d2, vocab_size=FIXTURE_VOCAB_SIZE)
        h1 = tokenizer_sha256(path1)
        h2 = tokenizer_sha256(path2)
        assert h1 == h2, f"Tokenizer training is non-deterministic! {h1} != {h2}"


def test_sha256_helper(trained_tokenizer_dir: pathlib.Path):
    """tokenizer_sha256 matches manual hashlib computation."""
    tok_path = trained_tokenizer_dir / "tokenizer.json"
    expected = hashlib.sha256(tok_path.read_bytes()).hexdigest()
    assert tokenizer_sha256(tok_path) == expected


def test_doc_filter_excludes_validation_documents():
    """Tokenizer training uses doc_filter to exclude val set."""
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        # 1. Train without filter
        path1 = train(FIXTURE_CORPUS, d1, vocab_size=FIXTURE_VOCAB_SIZE)

        # 2. Train with filter that excludes everything
        # If it excludes everything, BpeTrainer might raise an error or create an empty vocab.
        # But we just want to ensure doc_filter is actually called and changes the output.
        # Let's filter out exactly half the lines.
        path2 = train(
            FIXTURE_CORPUS,
            d2,
            vocab_size=FIXTURE_VOCAB_SIZE,
            doc_filter=lambda x: hash(x) % 2 == 0
        )
        h1 = tokenizer_sha256(path1)
        h2 = tokenizer_sha256(path2)
        assert h1 != h2, "Tokenizer training did not respect doc_filter"
