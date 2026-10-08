# Contributing to Memora v1

Thank you for your interest in contributing to Memora v1!

## How to Contribute

### Reporting Issues

Please use the GitHub issue tracker to report bugs or suggest improvements. Include:
- A clear description of the problem
- Steps to reproduce (if a bug)
- Expected vs. actual behavior
- Environment details (Python version, OS, GPU if relevant)

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes following the coding standards below
4. Run the test suite: `pytest tests/`
5. Run the linter: `ruff check memora/ tests/`
6. Run type checking: `mypy memora/`
7. Commit your changes with a clear message
8. Open a pull request against `main`

### Coding Standards

- **Style:** Enforced by `ruff` (linter + formatter). Run `ruff format memora/` before committing.
- **Types:** All public functions and classes should have type annotations.
- **Tests:** New functionality should be accompanied by unit tests in `tests/`.
- **Docs:** Update relevant documentation in `docs/` if you change behaviour.
- **Secrets:** Never commit secrets, API keys, or personal paths. The `.gitignore` excludes common patterns, but please double-check.

### Development Setup

```bash
# Clone and enter the repository
git clone https://github.com/tafsir99-gg/memora-v1-300m-3b-toks.git
cd memora-v1-300m-3b-toks

# Create environment matching Kaggle (recommended for training parity)
uv venv --python 3.11.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# OR use your system Python (>=3.11.11, <3.15 required)
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Verify
python -m memora.utils.environment
pytest tests/
```

### Data and Training

- **Do not commit** tokenized data shards, model checkpoints, or training logs.
- **Do not run** bulk data downloads or training on local machines; use Kaggle for that.
- Local development uses the `configs/tiny.yaml` config for CPU-only smoke tests.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

Contributions are licensed under Apache 2.0, the same as the project.
