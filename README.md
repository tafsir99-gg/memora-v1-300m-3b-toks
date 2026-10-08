# Memora v1

> A 300M parameter decoder-only English language model trained on 3B tokens of FineWeb-Edu.

## Overview

Memora v1 is an open-source, reproducible language model project built on commodity GPUs (Kaggle T4) with a focus on transparency, honest documentation, and clean engineering.

- **Architecture:** GPT-style decoder-only transformer, 24 layers, d_model=960, 15 heads (≈297M parameters)
- **Training data:** [FineWeb-Edu sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (ODC-By 1.0)
- **Training compute:** Kaggle free tier (NVIDIA T4, fp16)
- **Token budget:** 1B tokens (committed), 3B tokens (conditional on Phase 6 benchmark)

## Status

🔄 **Phase 4 — Implementation in progress.** Model card and weights will be available after training completes.

See [docs/STATUS.md](docs/STATUS.md) for the current phase and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design decisions.

## Repository Structure

```
memora-v1-300m-3b-toks/
├── configs/          # YAML training configs (300m.yaml, tiny.yaml)
├── memora/           # Python package
│   ├── config.py     # Pydantic config models
│   ├── model/        # Transformer architecture
│   ├── tokenizer/    # BPE tokenizer training and inference
│   ├── data/         # Data pipeline (sharding, loading)
│   ├── training/     # Trainer, optimizer, scheduler
│   ├── evaluation/   # Validation loss, benchmarks
│   ├── inference/    # Text generation
│   └── utils/        # Shared utilities
├── tests/            # Unit and integration tests
├── scripts/          # CLI entry points
└── docs/             # Architecture, training plan, data plan, risks
```

## Reproducibility

All design decisions are documented in `docs/`. Training is deterministic given the same seed and hardware. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for details.

## License

- **Code:** Apache 2.0 (see [LICENSE](LICENSE))
- **Training data:** FineWeb-Edu is released under ODC-By 1.0. Attribution: HuggingFace FineWeb team.
- **Model weights:** To be determined after training.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
