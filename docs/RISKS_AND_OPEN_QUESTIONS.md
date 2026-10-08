# Memora v1 — Risks and Open Questions

## 1. Risk Register

| # | Risk | Likelihood | Impact | Mitigation | Status |
|---|------|-----------|--------|------------|--------|
| R1 | T4 MFU too low → 3B tokens infeasible in available time | Medium | High | Phase 6 benchmark; fallback to 1-2B tokens; cosine decay is adjustable | OPEN |
| R2 | fp16 training instability (NaN/Inf, loss spikes) | Medium | Medium | GradScaler, grad clipping, fp32 loss computation, loss-scale monitoring; can reduce LR if needed | OPEN |
| R3 | Kaggle GPU unavailability (scarcity reported Sep 2026) | Medium | High | Schedule sessions at off-peak times; have single-GPU fallback config; track availability patterns | OPEN |
| R4 | Checkpoint corruption during session termination | Low | High | Atomic writes (temp+rename), integrity validation, fall-back to previous valid checkpoint, retention policy | MITIGATED BY DESIGN |
| R5 | DDP scaling efficiency worse than assumed (<<85%) | Medium | Medium | Phase 6 benchmark; 1×T4 fallback if scaling < 70%; since 1×T4 is default, DDP scaling is less critical | OPEN |
| R6 | Kaggle session terminates unexpectedly (before graceful shutdown) | Medium | Medium | Frequent lightweight checkpoints (every 100 steps); worst case lose ~5 min of training | MITIGATED BY DESIGN |
| R7 | HF Hub upload fails (network issue) | Low | Medium | Retry logic (3×); fallback to Kaggle output persistence; checkpoint remains valid locally | MITIGATED BY DESIGN |
| R8 | Memory OOM on T4 with planned micro-batch size | Medium | Low | Phase 6 memory sweep; reduce micro_batch_size and increase accumulation | MITIGATED BY DESIGN |
| R9 | Benchmark contamination in FineWeb-Edu | High | Low | Document contamination risk honestly; cannot fully decontaminate web data; run contamination check for exact matches | ACCEPTED |
| R10 | Python version mismatch between local (3.14) and Kaggle (3.11.11) | Low | Medium | Project pinned to `>=3.11.11, <3.15`; developed/tested on both; use `uv venv --python 3.11.11` for parity | MITIGATED BY DESIGN |
| R11 | torch.compile crashes or is slower on T4 | Medium | Low | Default: compile=false; test in Phase 6; easy to toggle | MITIGATED BY DESIGN |
| R12 | Tokenizer training produces suboptimal vocab | Low | Low | Byte-level BPE is robust; 32k vocab is standard; measure tokens-per-byte | ACCEPTED |
| R13 | Data processing takes too long on Kaggle CPU | Low | Medium | Stream-process; only need ~3.5B tokens; estimate <2h; can use local Mac as fallback | NEEDS-ESTIMATION |
| R14 | Disk space on Mac too tight for local dev | Low | Low | Only tiny test data locally; real data on Kaggle; 51 GB currently free | MONITORED |
| R15 | HF Hub repo history grows unboundedly from checkpoint overwrites | Low | Low | Use `upload_file()` (no git clone needed); `hf_hub_download()` downloads single files. Periodically create fresh HF repo if needed. | MITIGATED BY DESIGN |

## 2. Verified Claims Registry

Every factual claim about external systems, with verification status:

| # | Claim | Status | Source | Date Checked |
|---|-------|--------|--------|-------------|
| V1 | Kaggle GPU quota: 30h/week | VERIFIED | [kaggle.com forums](https://kaggle.com), [luminoai.in](https://luminoai.in) | 2026-10-01 |
| V2 | Kaggle session limit: 12h max | VERIFIED | [kaggle.com forums](https://kaggle.com), [clusy.io](https://clusy.io) | 2026-10-01 |
| V3 | A Kaggle 2×T4 session consumes GPU quota at exactly 2× the wall-clock rate (1 wall-clock hour = 2 quota hours) based on community reports. | ASSUMED | [luminoai.in](https://luminoai.in), [kaggle.com forums](https://kaggle.com) (Will verify in Phase 6) | 2026-10-01 |
| V4 | T4 lacks native bf16 | VERIFIED | [NVIDIA T4 datasheet](https://www.nvidia.com/en-us/data-center/tesla-t4/), sm_75 Turing arch | 2026-10-01 |
| V5 | T4 lacks FlashAttention-2 | VERIFIED | [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention), requires Ampere+ (sm_80+) | 2026-10-01 |
| V6 | T4 fp16 peak: 65 TFLOPS | VERIFIED | [NVIDIA T4 datasheet](https://www.nvidia.com/en-us/data-center/tesla-t4/) | 2026-10-01 |
| V7 | T4 compute capability: sm_75 | VERIFIED | [NVIDIA GPU compute capability table](https://developer.nvidia.com/cuda-gpus) | 2026-10-01 |
| V8 | Kaggle output limit: 20 GB | VERIFIED | [kaggle.com/docs](https://kaggle.com/docs) | 2026-10-01 |
| V9 | Kaggle dataset limit: 200 GB (public) | VERIFIED | [kaggle.com/docs](https://kaggle.com/docs) | 2026-10-01 |
| V10 | Kaggle Python version: 3.11.11 (current default) | VERIFIED | [github.com/Kaggle/docker-python](https://github.com/Kaggle/docker-python), kaggle.com release notes | 2026-10-01 |
| V11 | PyTorch supports Python 3.14 | VERIFIED | [pytorch.org](https://pytorch.org/get-started/locally/) (PyTorch 2.10+) | 2026-10-01 |
| V12 | FineWeb-Edu license: ODC-By 1.0 | VERIFIED | [huggingface.co/datasets/HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset card | 2026-10-01 |
| V13 | FineWeb-Edu sample-10BT: ~10B tokens | VERIFIED | [huggingface.co/datasets/HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset card | 2026-10-01 |
| V14 | FineWeb-Edu also subject to CommonCrawl ToU | VERIFIED | [huggingface.co/datasets/HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset card | 2026-10-01 |
| V15 | HF Hub free tier: unlimited public repos, no per-file LFS limit | ASSUMED | General knowledge; not recently re-verified against HF pricing page | 2026-10-01 |
| V16 | Kaggle idle timeout: 20 minutes | VERIFIED | [kaggle.com forums](https://kaggle.com) | 2026-10-01 |
| V17 | T4 VRAM: 16 GB per GPU | VERIFIED | [NVIDIA T4 datasheet](https://www.nvidia.com/en-us/data-center/tesla-t4/) | 2026-10-01 |
| V18 | T4 memory bandwidth: 320 GB/s | VERIFIED | [NVIDIA T4 datasheet](https://www.nvidia.com/en-us/data-center/tesla-t4/) | 2026-10-01 |
| V19 | DDP scaling ~85% on 2×T4 over PCIe | ASSUMED | Typical for small models over PCIe; must be measured in Phase 6 | 2026-10-01 |
| V20 | MFU range 10-30% for 300M on T4 | ASSUMED | Literature range for similar-scale models on T4; must be measured in Phase 6 | 2026-10-01 |

## 3. Resolved Questions (Owner Decisions)

| # | Question | Owner's Decision | Date |
|---|----------|-----------------|------|
| Q1 | Dataset: FineWeb-Edu sample-10BT? | **Yes** — confirmed | 2026-10-01 |
| Q2 | FFN: Standard GELU? | **Yes** — confirmed | 2026-10-01 |
| Q3 | HF Hub repo name: `tafsir99-gg/memora-v1-300m`? | **Yes, if available** — confirmed (search shows name not taken) | 2026-10-01 |

## 4. Remaining Open Questions (Awaiting Owner)

### Q4: Kaggle Account / Secrets Setup
**Question:** Is the Kaggle account set up? Can you confirm you can create notebooks, add secrets (HF_TOKEN), and access GPU?
**Recommended default:** Verify early in Phase 4 to avoid surprises.
**Impact if deferred:** Blocks Kaggle-dependent work in Phases 5-7.

## 5. Decision Log

| # | Decision | Alternatives Considered | Rationale | Evidence/Status |
|---|----------|------------------------|-----------|-----------------|
| D1 | Decoder-only GPT-style architecture | Encoder-decoder, mixture-of-experts | Standard for autoregressive LM; simpler; well-understood scaling | VERIFIED (industry standard) |
| D2 | d_model=960, 24 layers, 15 heads | d=1024/22L, d=768/32L, d=1024/24L (335M, too large) | Closest to 300M target (297M); 64-dim heads are Tensor Core efficient | VERIFIED (arithmetic) |
| D3 | Standard GELU FFN | SwiGLU, GLU variants | **Owner confirmed.** Simpler; proven; 297M fits budget exactly; SwiGLU requires arch rebalancing | DECIDED |
| D4 | Pre-LayerNorm | Post-LayerNorm, RMSNorm | More stable training; standard in modern transformers | VERIFIED (literature consensus) |
| D5 | RoPE position encoding | ALiBi, Absolute, Relative | Standard in modern LLMs; no learnable params; length generalization | VERIFIED (widely adopted) |
| D6 | No bias in attn/FFN | Bias everywhere | Modern convention (LLaMA, GPT-NeoX); fewer params; slightly faster | VERIFIED (modern convention) |
| D7 | Tied input/output embeddings | Untied | Saves 31.5M params; standard at this scale | VERIFIED (GPT-2, many others) |
| D8 | Byte-level BPE (tokenizers lib) | SentencePiece BPE, Unigram | No UNK tokens; fast (Rust); good Python API; standard | VERIFIED (GPT-2 convention) |
| D9 | 32,768 vocab | 16k, 50k, 64k | Power-of-2 for efficiency; adequate for English; good embedding cost/coverage tradeoff | ASSUMED (standard range) |
| D10 | FineWeb-Edu (sample-10BT) | Pile, C4, SlimPajama, RedPajama | **Owner confirmed.** Best quality for size; permissive license; pre-cleaned; educational content focus | DECIDED |
| D11 | fp16 + GradScaler (not bf16) | bf16, fp32 | T4 has no native bf16; fp16 is the only Tensor Core option | VERIFIED (T4 sm_75 specs) |
| D12 | PyTorch SDPA (not FlashAttention-2) | FlashAttention-2, manual attention | FA2 doesn't support T4 (sm_75); SDPA auto-selects best available backend | VERIFIED (FA2 requires Ampere+) |
| D13 | DDP (not FSDP) | FSDP, DeepSpeed, model parallel | Model fits in single GPU memory; DDP is simpler and sufficient | VERIFIED (300M = ~1.2 GB fp32) |
| D14 | Cosine decay LR schedule | WSD (warmup-stable-decay), linear decay | Standard; smooth; proven for LLM pretraining; token-based for session independence | ASSUMED (standard practice) |
| D15 | YAML config + Pydantic validation | TOML, JSON, Hydra, argparse-only | YAML is readable; Pydantic catches errors early; no heavy dependency (vs. Hydra) | ASSUMED (preference + simplicity) |
| D16 | HF Hub for checkpoint persistence | Kaggle Datasets for checkpoints, GCS, S3 | **Owner confirmed repo name.** HF Hub is free, has good Python API, versioned, no infra to manage | DECIDED |
| D17 | JSONL logging (not W&B/TensorBoard) | Weights & Biases, TensorBoard, MLflow | Zero extra dependency; works across sessions; easy to aggregate; plotting script included | ASSUMED (simplicity) |
| D18 | `uv` for local package management | pip, conda, poetry | Already installed; fast; compatible with pip; good lockfile support | VERIFIED (uv available on system) |
| D19 | Apache-2.0 for code | MIT, GPL | Already committed in repo; permissive; compatible with most uses | VERIFIED (LICENSE file exists) |
| D20 | ODC-By 1.0 acknowledgment for data | — | FineWeb-Edu requires attribution; will include in README and model card | VERIFIED (dataset license) |
| D21 | **Default 1×T4** (not 2×T4) | 2×T4 | 2×T4 costs 2× quota but gives <2× throughput → net quota loss. 1×T4 is always more quota-efficient. | ASSUMED (quota rate pending Phase 6 measurement) |
| D22 | **Python >=3.11.11, <3.15** project pin | >=3.11.11, <3.15 | Kaggle runs 3.11.11; local Mac has 3.14. Excludes <3.11.11, while PyTorch 2.10+ wheels allow up to 3.14 | VERIFIED |

## 6. Contradictions and Ambiguities in the Spec (Identified in Phase 0)

| # | Issue | Resolution |
|---|-------|------------|
| C1 | Spec says 2×T4 "may scale poorly" and suggests verifying quota rate → **Community reports: 2×T4 costs 2× quota.** Combined with <100% DDP scaling, this makes 1×T4 always more quota-efficient. | Default to 1×T4. ASSUMED pending Phase 6 measurement (quota meter check). |
| C2 | Spec says "~30 GPU-hours/week" but notes "may vary" | VERIFIED: 30h/week standard quota. GPU availability may be scarce (Sep 2026 reports). |
| C3 | Spec says "fp16 stability issues" → must use GradScaler | Accounted for: GradScaler enabled, fp32 loss computation, grad clipping, loss-scale monitoring. |
| C4 | Spec mentions "earlier byte-level project" to "reuse ideas from" | No earlier project found in the repo or adjacent directories. Nothing to reuse. |
| C5 | Optimizer state (~2.4 GB) + model (~1.2 GB) = ~3.6 GB per full checkpoint × 2 retained + 1 best = ~10.8 GB + 3 lightweight × 1.2 GB = ~14.4 GB | Fits within Kaggle 20 GB output limit. |
| C6 | "Large data never goes into GitHub" vs. tokenizer JSON (~2 MB) | Tokenizer JSON (2 MB) is metadata, not "large data." It goes into GitHub. Binary shards do not. |
| C7 | Preprocessing cost: FineWeb-Edu is already cleaned, so heavy preprocessing (dedup, lang filter) is NOT needed | This resolves the concern about preprocessing being infeasible on Mac/Kaggle CPU. Only light filtering + tokenization needed. |

## 7. Python Version Strategy

| Environment | Python Version | Notes |
|-------------|---------------|-------|
| **Local development (Mac Air M2)** | 3.14.2 (system) | Use `python3.11.11` via pyenv/uv for venv if needed for Kaggle parity testing |
| **Kaggle (training)** | 3.11.11 (Kaggle default) | VERIFIED: Kaggle docker-python uses 3.11.11 |
| **Project pin** | `>=3.11.11, <3.15` | Set in `pyproject.toml`; bounds PyTorch compatibility up to 3.14 |

### Local venv setup instructions

```bash
# Option A: Use system Python 3.14 (works, since PyTorch 2.10+ supports 3.14)
cd memora-v1-300m-3b-toks
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Option B: Use uv with a specific Python version (recommended for Kaggle parity)
cd memora-v1-300m-3b-toks
uv venv --python 3.11.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify
python --version  # Should show 3.11.11 or 3.14.x
python -c "import torch; print(torch.__version__)"
```

**Note:** `uv venv --python 3.11.11` will auto-download Python 3.11.11 if not already installed. This is the recommended approach for Kaggle parity testing.

## 8. Acceptance Criteria for the Project

The project is considered complete when all of the following are verified:

1. ✅ Model architecture implemented and parameter count matches plan (296,972,160)
2. ✅ Tokenizer trained, tested, and versioned
3. ✅ Data pipeline produces deterministic, immutable train/val split
4. ✅ Training runs stably for the full token budget (or documented stopping point)
5. ✅ Checkpoints are resumable across sessions (demonstrated by actual multi-session training)
6. ✅ Evaluation metrics recorded at all milestones
7. ✅ Benchmarks run and results reported honestly
8. ✅ Inference pipeline works independently
9. ✅ All validation ladder stages pass
10. ✅ README, model card, and reproduction guide are complete and accurate
11. ✅ Another person can clone the repo and reproduce the tests
12. ✅ `docs/AUDIT.md` lists verification status for every component
