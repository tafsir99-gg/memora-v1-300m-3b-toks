# Memora v1 — Reproducibility Guide

## 1. What Is Reproducible

| Component | Guarantee | Notes |
|-----------|-----------|-------|
| Data split (train/val) | **Deterministic** | Hash-based; same data → same split |
| Data order | **Deterministic** given seed | Same seed + rank → same batch sequence |
| Tokenizer | **Deterministic** | Same training data + config → same tokenizer (verified by hash) |
| Model initialization | **Deterministic** given seed | Same config + seed → same initial weights |
| Single-GPU fp32 | **Deterministic** (with caveats) | Requires `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG` |
| Single-GPU fp16 | **Statistically equivalent** | GradScaler dynamics may differ across runs due to fp16 precision |
| Multi-GPU DDP | **Statistically equivalent** | AllReduce non-determinism on fp16; NCCL reduction order not guaranteed |
| Resume | **Statistically equivalent** | All RNG states checkpointed; data position exact; loss within tolerance |

### 1.1 What We Cannot Guarantee

- **Bitwise identical training** across multi-GPU runs: NCCL AllReduce is non-deterministic for fp16 due to floating-point reduction order
- **Bitwise identical training** across different PyTorch/CUDA/cuDNN versions
- **Bitwise identical training** with `torch.compile` (graph-level optimizations may differ)
- **Identical hardware across sessions**: Kaggle may assign different physical T4 GPUs, though compute capability is the same

We will NEVER claim bitwise reproducibility for multi-GPU fp16 training. We claim and verify:
- Deterministic data order
- Deterministic initialization
- Deterministic tokenizer
- Statistically equivalent resume (losses within documented tolerance)

## 2. Versioned Artifacts

Every training run records:

### 2.1 Code Version
```json
{
    "git_commit": "abc123...",
    "git_dirty": false,
    "git_branch": "main"
}
```

### 2.2 Dependencies
```json
{
    "python_version": "3.12.x",
    "pytorch_version": "2.x.x",
    "cuda_version": "12.x",
    "cudnn_version": "8.x.x",
    "packages": {
        "tokenizers": "0.19.x",
        "pydantic": "2.x.x",
        "numpy": "1.x.x",
        "...": "..."
    }
}
```

Pinned via `requirements.txt` (generated from `pyproject.toml` + `pip freeze`).

### 2.3 Configuration
Full YAML config snapshot in every checkpoint.

### 2.4 Tokenizer
SHA-256 hash of `tokenizer/tokenizer.json` recorded in checkpoint and manifest.

### 2.5 Dataset
- Manifest JSON with: source, preprocessing config, split method, document/token counts
- SHA-256 hashes of every shard file
- Manifest hash recorded in checkpoint

### 2.6 Hardware
```json
{
    "hostname": "...",
    "gpu_name": "Tesla T4",
    "gpu_count": 2,
    "gpu_memory_gb": 16,
    "cuda_compute_capability": "7.5"
}
```

### 2.7 Seeds
```json
{
    "training_seed": 42,
    "data_seed": 42,
    "per_rank_seed_formula": "seed + rank"
}
```

### 2.8 RNG States (in checkpoint)
- `random.getstate()`
- `numpy.random.get_state()`
- `torch.random.get_rng_state()`
- `torch.cuda.get_rng_state()` (per device/rank)

## 3. Reproducibility Verification Tests

| Test | Method | Expected Result |
|------|--------|-----------------|
| Tokenizer determinism | Train twice with same config/data/seed | Identical tokenizer.json (same hash) |
| Data split determinism | Process twice with same seed | Identical shard contents (same hashes) |
| Model init determinism | Initialize twice with same config/seed on CPU | Identical weights (bitwise) |
| Forward pass determinism | Two forward passes with same input on CPU | Identical outputs (bitwise) |
| Training determinism (CPU, fp32) | Train 100 steps twice | Identical losses (bitwise, with deterministic mode) |
| Resume equivalence (CPU, fp32) | Train 100 steps vs. train 50 + resume 50 | Identical final weights and losses |
| Resume equivalence (GPU, fp16) | Train 100 steps vs. train 50 + resume 50 | Losses within tolerance ε (document ε after measurement) |

## 4. Reproduction Instructions (Draft)

> To be finalized in Phase 8. This is the planned structure.

```bash
# 1. Clone repository
git clone https://github.com/tafsir99-gg/memora-v1-300m-3b-toks.git
cd memora-v1-300m-3b-toks
git checkout <exact-commit-from-model-card>

# 2. Create environment (Option A: system Python, if >=3.11)
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Create environment (Option B: uv with Kaggle-matching Python 3.11.11)
uv venv --python 3.11.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 3. Verify environment
python -m memora.utils.environment  # prints Python, PyTorch, CUDA versions

# 4. Run tests
pytest tests/ -v

# 5. Count parameters
python scripts/count_params.py --config configs/300m.yaml

# 6. Train tokenizer (or download pre-trained)
python scripts/train_tokenizer.py --config configs/300m.yaml

# 7. Prepare data (or download pre-tokenized shards)
python scripts/prepare_data.py --config configs/300m.yaml

# 8. Train (local CPU test)
python -m memora.training.launcher --config configs/tiny.yaml --max-steps 100

# 9. Train (Kaggle, see notebooks/kaggle_train.py)
# torchrun --nproc_per_node=2 -m memora.training.launcher \
#     --config configs/300m.yaml --resume latest

# 10. Evaluate
python -m memora.evaluation.perplexity --config configs/300m.yaml --checkpoint best
python -m memora.evaluation.sampling --config configs/300m.yaml --checkpoint best

# 11. Generate text
python -m memora.inference.generate \
    --checkpoint checkpoints/best/ \
    --prompt "The meaning of life is" \
    --temperature 0.8 --top-k 50 --max-tokens 128
```
