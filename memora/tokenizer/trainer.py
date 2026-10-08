"""Tokenizer training on a text corpus.

This module trains a byte-level BPE tokenizer using the HuggingFace tokenizers
library. It is designed to be called on Kaggle (real corpus) or locally
(tiny fixture for tests only).

CONSTRAINT: The real 32k tokenizer is trained on Kaggle CPU, not on the Mac.
On Mac, only the tiny fixture corpus is used (for test purposes).
"""

import hashlib
import json
import pathlib
from collections.abc import Iterator

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# Special token definitions (stable; do not reorder without re-training).
SPECIAL_TOKENS = ["<|pad|>", "<|eos|>", "<|unk|>"]
PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|eos|>"
UNK_TOKEN = "<|unk|>"

PAD_ID = 0
EOS_ID = 1
UNK_ID = 2


def _text_iterator(corpus_path: str | pathlib.Path) -> Iterator[str]:
    """Yield non-empty lines from a plain-text file (one document per line)."""
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def train(
    corpus_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    vocab_size: int = 32768,
) -> pathlib.Path:
    """Train a byte-level BPE tokenizer on the given corpus.

    Parameters
    ----------
    corpus_path:
        Path to a plain-text file with one document per line.
    output_dir:
        Directory where ``tokenizer.json`` and ``tokenizer_config.json`` are saved.
    vocab_size:
        Target vocabulary size. The real model uses 32,768; tests use 512.

    Returns
    -------
    Path to the saved ``tokenizer.json``.
    """
    corpus_path = pathlib.Path(corpus_path)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )

    tokenizer.train_from_iterator(
        _text_iterator(corpus_path),
        trainer=trainer,
    )

    out_path = output_dir / "tokenizer.json"
    tokenizer.save(str(out_path))

    # Write a minimal config file alongside for easy loading.
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "pad_token": PAD_TOKEN,
        "eos_token": EOS_TOKEN,
        "unk_token": UNK_TOKEN,
        "pad_id": PAD_ID,
        "eos_id": EOS_ID,
        "unk_id": UNK_ID,
    }
    with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return out_path


def tokenizer_sha256(tokenizer_path: str | pathlib.Path) -> str:
    """Return the SHA-256 hex digest of a saved ``tokenizer.json`` file."""
    data = pathlib.Path(tokenizer_path).read_bytes()
    return hashlib.sha256(data).hexdigest()
