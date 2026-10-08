"""Tokenizer wrapper: load, encode, decode.

This module provides a thin, stable wrapper around a saved HuggingFace
``tokenizers`` tokenizer. It is intentionally kept minimal so it can be
loaded anywhere without needing the training code.
"""

import pathlib

from tokenizers import Tokenizer

from memora.tokenizer.trainer import EOS_ID, PAD_ID, UNK_ID


class MemoraTokenizer:
    """Load and use a saved byte-level BPE tokenizer.

    Parameters
    ----------
    tokenizer_path:
        Path to ``tokenizer.json`` (as saved by :func:`memora.tokenizer.trainer.train`).
    """

    def __init__(self, tokenizer_path: str | pathlib.Path) -> None:
        self._path = pathlib.Path(tokenizer_path)
        self._tok: Tokenizer = Tokenizer.from_file(str(self._path))

        # Verify special token IDs are stable and match trainer definitions.
        vocab = self._tok.get_vocab()
        assert vocab.get("<|pad|>") == PAD_ID, f"PAD_ID mismatch: got {vocab.get('<|pad|>')}"
        assert vocab.get("<|eos|>") == EOS_ID, f"EOS_ID mismatch: got {vocab.get('<|eos|>')}"
        assert vocab.get("<|unk|>") == UNK_ID, f"UNK_ID mismatch: got {vocab.get('<|unk|>')}"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """Encode a string to a list of integer token IDs.

        Parameters
        ----------
        text:
            Input text.
        add_eos:
            If True, append EOS_ID at the end (used when building training sequences).
        """
        ids: list[int] = self._tok.encode(text).ids
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to a string.

        Parameters
        ----------
        ids:
            Token ID sequence.
        skip_special_tokens:
            If True, special tokens (PAD, EOS, UNK) are omitted from the output.
        """
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # Classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_dir(cls, tokenizer_dir: str | pathlib.Path) -> "MemoraTokenizer":
        """Load from a directory containing ``tokenizer.json``."""
        return cls(pathlib.Path(tokenizer_dir) / "tokenizer.json")
