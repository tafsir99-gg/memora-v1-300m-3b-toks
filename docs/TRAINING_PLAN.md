# Memora v1 — Training Plan

## 1. Training Recipe

### 1.1 Optimizer

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Optimizer | AdamW (decoupled weight decay) | Standard for transformer pretraining; well-understood dynamics |
| β1 | 0.9 | Standard; controls momentum of first moment |
| β2 | 0.95 | Lower than default 0.999; better stability for LLM training (GPT-3, Chinchilla convention) |
| ε | 1e-8 | Standard; prevents division by zero |
| Weight decay | 0.1 | Standard for transformers; applied to all params EXCEPT: LayerNorm weights, LayerNorm biases, and embedding weights |

**Weight decay exclusion rationale:**
- LayerNorm parameters should not be decayed (they are scale/shift parameters, not weights in the traditional sense)
- Embedding weights: debatable; many implementations exclude them. We exclude to follow GPT-2/GPT-3 convention. This is a minor decision at 0.1 decay.

### 1.2 Learning Rate Schedule

#### Scheduler Options (config-driven)

**Default:** `cosine` — until Phase 6 benchmark determines the committed training horizon.

**Option 2:** `wsd` (warmup-stable-decay) — a separate short decay phase that can start at any token count, making it suitable when the final horizon is uncertain. Under WSD, LR stays at peak for the stable phase, then decays over a configurable window (e.g., ~10% of training) once the stop token count is known.

**How the Phase 6 decision is made:** After measuring actual tok/s on 1×T4, if 3B tokens is confirmed feasible within the quota, keep `cosine` (max_tokens=3B). If 1B is the chosen stopping point, switch to `wsd` with a decay window of ~100M tokens starting at 900M tokens, so the model reaches min_lr at exactly 1B. Alternatively, re-initialize `cosine` with max_tokens=1B before the final session.

```
LR (cosine, max_tokens=3B)
 ↑
 |    /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
 |   /                             \
 |  /                               \
 | /                                 \
 |/                                   \___________
 +────────────────────────────────────────────────→ tokens
 0    warmup (~131M)   cosine decay        min_lr
      ~250 steps       → 3B tokens         (hold)
```

> **Note on the 1B milestone under cosine with max_tokens=3B:** At 1B tokens the schedule is ~33% through its decay; LR ≈ peak − 17% ≈ ~83% of peak. The model is not annealed. This is acceptable for an intermediate checkpoint and loss measurement, but means the 1B checkpoint cannot be directly compared to a fully-annealed model. If the Owner wants a fully-annealed 1B model, switch to `wsd` or re-run cosine with max_tokens=1B.

| Parameter | Config key | Default value | Justification |
|-----------|-----------|---------------|---------------|
| Peak LR | `lr.peak` | `6e-4` | GPT-3 heuristic for ~300M; within [3e-4, 1e-3] standard range |
| Min LR | `lr.min` | `6e-5` | 10% of peak; standard end-of-decay floor |
| Warmup tokens | `lr.warmup_tokens` | `131_072_000` (~131M) | ≈250 steps at 524K tokens/step ≈ 4.4% of 3B total tokens. Prevents early instability. Token-based so it is step-count agnostic. |
| Warmup steps (derived) | — | ≈250 | = warmup_tokens / global_batch_tokens |
| Schedule shape | `lr.schedule` | `"cosine"` | `"cosine"` or `"wsd"`. Default stays cosine until Phase 6. |
| WSD decay window | `lr.wsd_decay_tokens` | `100_000_000` | Only active when schedule=wsd. |
| Schedule domain | **Token-based** | — | Defined over total tokens, not steps. Session-independent; correct under gradient accumulation changes. |

**How the cosine schedule works across sessions:**

The decay is parameterized by `tokens_seen` (restored from checkpoint):

```python
if tokens_seen < warmup_tokens:
    lr = peak_lr * (tokens_seen / warmup_tokens)
elif schedule == "cosine":
    progress = (tokens_seen - warmup_tokens) / (max_tokens - warmup_tokens)
    progress = min(progress, 1.0)
    lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
elif schedule == "wsd":
    if tokens_seen < stable_end_tokens:
        lr = peak_lr  # stable phase
    else:
        decay_progress = (tokens_seen - stable_end_tokens) / wsd_decay_tokens
        decay_progress = min(decay_progress, 1.0)
        lr = min_lr + (peak_lr - min_lr) * (1 - decay_progress)  # linear decay
```

If continuing past `max_tokens` (cosine), LR stays at `min_lr`.

### 1.3 Batch Size

All batch parameters are **config-driven**. The table below shows defaults; Phase 6 may change them.

| Parameter | Config key | Default | Justification |
|-----------|-----------|---------|---------------|
| Global batch size (tokens) | `batch.global_tokens` | `524_288` (~512K) | Chinchilla range for ~300M models. ~512 seqs × 1024 tokens. |
| Micro-batch size | `batch.micro_size` | `16` | Per-GPU; 16 × 1024 = 16,384 tokens per micro-step. ESTIMATE — must verify against T4 memory. |
| Gradient accumulation steps | `batch.grad_accum` | Computed | = global_tokens / (micro_size × seq_len × world_size) |
| Sequence length | `model.seq_len` | `1024` | Matches model.max_seq_len |

> **Phase 6 candidate:** `global_tokens = 262_144` (~256K). A smaller batch may raise MFU by reducing gradient accumulation overhead. The Phase 6 benchmark will compare MFU and stable training at both 262K and 524K before committing.

**Example configurations (default 524K):**

| GPU Config | Micro BS | Accum Steps | Effective Batch (tokens) |
|-----------|----------|-------------|-------------------------|
| 1×T4 | 16 | 32 | 16 × 1024 × 32 = 524,288 |
| 2×T4 | 16 | 16 | 16 × 1024 × 16 × 2 = 524,288 |

### 1.4 Precision and Loss Scaling

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Training precision | fp16 | T4 has no native bf16; fp16 is the only Tensor Core option |
| Autocast scope | Forward pass + loss computation in fp16 autocast | Standard practice |
| Master weights | fp32 | Maintained by optimizer; required for fp16 stability |
| GradScaler | Enabled, required | Prevents gradient underflow in fp16 |
| Loss computation | **fp32** | Logits upcasted to fp32 before cross-entropy; critical for numerical stability |
| GradScaler init scale | 2^16 (65536) | PyTorch default; will be auto-adjusted |
| GradScaler growth interval | 2000 | Default; may tune if overflow is frequent |

**fp16 stability concerns on T4:**
- Loss scale overflow possible in early training → GradScaler handles automatically by skipping steps
- Very large logit values can cause fp16 overflow → logits computed in fp32 for loss
- Gradient norm may spike → gradient clipping at 1.0
- Monitor: loss_scale value in logs; if it drops to very low values, training may stall

### 1.5 Initialization

| Component | Init | Justification |
|-----------|------|---------------|
| Token embeddings | N(0, 0.02) | GPT-2 convention |
| Q, K, V projections | N(0, 0.02) | Standard |
| O projection | N(0, 0.02 / √(2 × n_layers)) | Residual-scaled; prevents signal growth through deep residual stream |
| FFN up | N(0, 0.02) | Standard |
| FFN down | N(0, 0.02 / √(2 × n_layers)) | Residual-scaled; same rationale as O projection |
| LayerNorm weight | 1.0 (ones) | Standard; identity transform at init |
| LayerNorm bias | 0.0 (zeros) | Standard |

**Residual scaling rationale:** In a model with L residual connections, each contributes to the residual stream. Without scaling, the variance grows as O(L). Scaling output projections by 1/√(2L) keeps the variance stable. Factor of 2 because each layer has 2 residual connections (attention + FFN).

### 1.6 Other Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Dropout | 0.0 | Standard for pretraining; the 3B token dataset provides sufficient regularization. Dropout hurts throughput and is not used in modern LLM pretraining. |
| Gradient clipping | 1.0 (max_norm) | Prevents gradient explosion spikes; standard value |
| Seed | 42 | Arbitrary but fixed; determines data order, init, sampling |
| Per-rank seed | seed + rank | Ensures different data order per DDP rank while remaining reproducible |

## 2. Tokens-per-Parameter Analysis

At 3B tokens on a ~300M model:
- **Tokens per parameter:** 3B / 297M ≈ **10.1 tokens/param**
- **Chinchilla optimal:** ~20 tokens/param (for compute-optimal training)
- **Status:** **Undertrained** by Chinchilla standards (~50% of optimal)

**Implications:**
- The model will be undertrained; this is acknowledged and accepted given compute constraints
- The model will likely improve with more data (not saturated)
- Loss will not have fully converged by 3B tokens
- The model card must honestly state this limitation
- This makes the 1B milestone (3.4 tokens/param) useful as a comparison point: we expect significant improvement from 1B→3B

## 3. Compute Feasibility Analysis (CORRECTED)

> **STATUS: ESTIMATE.** Based on first-principles calculation, NOT measurement. Will be updated with measured throughput after Phase 6 benchmark.

### 3.1 Key Correction: 2×T4 Quota Rate

A Kaggle 2×T4 session consumes GPU quota at exactly 2× the wall-clock rate (1 wall-clock hour = 2 quota hours) based on community reports ([luminoai.in](https://luminoai.in), 2026-10-01). — ASSUMED (will be verified in Phase 6 by comparing Kaggle's quota meter before and after short 1×T4 and 2×T4 test sessions).

This means: 30h weekly quota → only 15 clock-hours of 2×T4 usage.

**Critical insight:** 1×T4 is ALWAYS more quota-efficient than 2×T4 (since DDP scaling < 100% but quota cost is exactly 200%). **Default recommendation: use 1×T4.** 2×T4 is only useful if wall-clock speed matters more than total quota consumed (e.g., to finish a session's work within the 12h limit).

### 3.2 Assumptions

- T4 peak fp16 Tensor Core: 65 TFLOPS — VERIFIED, NVIDIA specs
- FLOPs per token ≈ 6N ≈ 1.78 × 10⁹ (standard approximation, ignores attention quadratic term)
- Model FLOP Utilization (MFU): range from 10% (plausible low) to 30% (optimistic high) — UNKNOWN-NEEDS-MEASUREMENT
- DDP scaling efficiency on 2×T4 over PCIe: ~85% — ASSUMED, NEEDS-MEASUREMENT
- Effective training per week: ~25 quota-hours (after per-session overhead of ~1-1.5h for setup, checkpoint, upload)

### 3.3 Throughput Estimates — 1×T4 (Recommended Default)

> **Goal definition:** 1B tokens is the committed milestone. 3B tokens is conditional on measured tok/s from Phase 6.

| Scenario | MFU | tok/s | 1B tokens | 3B tokens | Weeks (at ~25h eff/week) |
|----------|-----|-------|-----------|-----------|--------------------------|
| **Plausible Low** | **10%** | **3,611** | **76.9h** | **230.8h** | **1B: 3.1w · 3B: 9.2w** |
| **Plausible Base** | **15%** | **5,417** | **51.3h** | **153.8h** | **1B: 2.1w · 3B: 6.2w** |
| Conservative | 20% | 7,222 | 38.5h | 115.4h | 1B: 1.5w · 3B: 4.6w |
| Target | 25% | 9,028 | 30.8h | 92.3h | 1B: 1.2w · 3B: 3.7w |
| Optimistic High | 30% | 10,833 | 25.6h | 76.9h | 1B: 1.0w · 3B: 3.1w |

### 3.4 Throughput Estimates — 2×T4 (Only if Wall-Clock Speed Needed)

| Scenario | MFU | tok/s | Clock hours | Quota-hours consumed | Weeks |
|----------|-----|-------|-------------|---------------------|-------|
| Plausible Base | 15% | 9,208 | 1B: 30.2h · 3B: 90.5h | 1B: 60.3h · 3B: 181h | 1B: 2.4w · 3B: 7.2w |
| Target | 25% | 15,347 | 1B: 18.1h · 3B: 54.3h | 1B: 36.2h · 3B: 108.6h | 1B: 1.4w · 3B: 4.3w |
| Optimistic High | 30% | 18,417 | 1B: 15.1h · 3B: 45.2h | 1B: 30.2h · 3B: 90.5h | 1B: 1.2w · 3B: 3.6w |

Note: 2×T4 is faster in clock time but consumes **more** weekly quota.

### 3.5 Stop/Extend Decision Rules

After Phase 6 benchmark (measured tok/s on 1×T4):

| Measured tok/s (1×T4) | MFU (approx) | 3B feasibility | Decision |
|------------------------|-------------|----------------|----------|
| ≥ 9,000 | ≥ 25% | 3B in ~3.7 weeks | ✅ **Go to 3B.** Cosine decay over 3B tokens. |
| 5,000 — 9,000 | 15-25% | 3B in 4-6 weeks | ⚠️ **Go to 3B if Owner approves the timeline.** Otherwise target 1-2B. |
| 3,500 — 5,000 | 10-15% | 3B in 6-9 weeks | ⚠️ **Target 1B-2B.** 3B may take too long. Investigate bottleneck. |
| < 3,500 | < 10% | 3B infeasible | ❌ **Stop.** Investigate: is this a software bug? Wrong batch size? Measure and fix before burning quota. |

**Schedule adjustment if target changes:**
- LR schedule is defined by `max_tokens` (and `schedule` type) in config
- **Cosine + stop at 1B:** re-set `max_tokens=1_000_000_000` before the final session's run, OR switch to `wsd` with `stable_end_tokens=900_000_000` and `wsd_decay_tokens=100_000_000`. Both give a fully-annealed 1B model.
- **Cosine + extend past 3B:** continue at `min_lr` (floor hold)
- **WSD:** start decay at any chosen token count; length of decay window controls how sharp the anneal is

## 4. Training Milestones

All milestones on ONE continuously trained model (not separate models).

| Milestone | Tokens | Steps (est.) | GPU-hours (est. @ 25% MFU, 1xT4) | Actions |
|-----------|--------|-------------|------------------|---------|
| Debug | 10M | ~20 | <0.1h | Verify end-to-end pipeline |
| 100M | 100M | ~190 | ~3.1h | First val loss check; verify convergence |
| 300M | 300M | ~572 | ~9.2h | Log val loss, check for instabilities |
| **1B** | 1B | ~1907 | ~30.8h | **Primary intermediate**: full eval suite, benchmarks, samples |
| 2B | 2B | ~3814 | ~61.5h | Eval suite, benchmarks, samples |
| **3B** | 3B | ~5722 | ~92.3h | **Primary target**: full eval, benchmarks, model card |

**Total steps:** 3B tokens / 524,288 tokens/step ≈ 5,722 steps

## 5. Monitoring Checklist

Every training step logs (JSONL format):

```json
{
    "step": 100,
    "tokens_seen": 52428800,
    "train_loss": 6.234,
    "lr": 0.000312,
    "grad_norm": 0.847,
    "loss_scale": 65536.0,
    "micro_step_time_ms": 234.5,
    "tokens_per_sec": 17892,
    "gpu_memory_mb": 12456,
    "gpu_utilization_pct": 87.3,
    "timestamp": "2026-10-15T08:23:45Z",
    "session_id": "session_001",
    "wall_clock_s": 3456.7,
    "cumulative_gpu_hours": 2.34
}
```

Every validation step additionally logs:
```json
{
    "step": 500,
    "val_loss": 5.891,
    "val_perplexity": 362.4,
    "val_bpb": 1.234,
    "val_tokens_evaluated": 10000000,
    "is_best": true
}
```

### 5.1 Anomaly Detection (Manual, but Logged)

Flag for human review:
- Loss spike > 2× running average
- Gradient norm > 10× running average
- Loss scale drops below 1.0
- Training loss increases for 100+ consecutive steps
- Val loss diverges from train loss by > 2× (overfitting or data issue)
- Tokens/sec drops by > 20% (throughput regression)

## 6. Session Management

### 6.1 Graceful Shutdown

The trainer handles these shutdown signals:

1. **SIGTERM**: Kaggle sends this when session is about to end. Trainer catches it, saves checkpoint, uploads, exits cleanly.
2. **Wall-clock deadline**: Configurable (default: 11 hours into a 12-hour session). Trainer checks elapsed time after each step.
3. **Token budget reached**: `tokens_seen >= max_tokens`. Normal completion.

### 6.2 Session Metadata

Each session records:
```json
{
    "session_id": "session_003",
    "start_step": 1500,
    "end_step": 2847,
    "start_tokens": 786432000,
    "end_tokens": 1492992000,
    "gpu_hours": 10.5,
    "avg_tokens_per_sec": 18234,
    "anomalies": [],
    "checkpoint_saved": "checkpoint_step2847.pt",
    "upload_status": "success"
}
```

## 7. Checkpoint Persistence (Detailed)

### 7.1 Checkpoint Sizes

| Checkpoint type | Contents | Estimated size | Calculation |
|----------------|----------|---------------|-------------|
| **Full resumable** | Model (fp32) + AdamW (m+v, fp32) + scheduler + scaler + RNG + metadata | **~3.6 GB** | Model: 297M × 4B = 1.19 GB. Optimizer m: 1.19 GB. Optimizer v: 1.19 GB. Other: ~30 MB. |
| **Lightweight** | Model weights (fp32) + metadata only | **~1.2 GB** | Model: 297M × 4B = 1.19 GB. Metadata: ~1 MB. |
| **Best validation** | Same as full resumable | **~3.6 GB** | Same contents |

### 7.2 Retention Policy

| Type | Kept | Max disk | Justification |
|------|------|----------|---------------|
| Full resumable | Last 2 | ~7.2 GB | Safety margin for corruption/fallback |
| Lightweight | Last 3 | ~3.6 GB | Quick weight snapshots; low cost |
| Best validation | Last 1 | ~3.6 GB | Best model for final release |
| **Total on disk** | | **~14.4 GB** | Safely within Kaggle 20 GB output limit |

**Note:** The 14.4 GB estimate comfortably fits within the 20 GB limit. We can also compress checkpoints (gzip) to reduce on-disk footprint by ~30-50% if needed.

### 7.3 HF Hub Persistence

- **What is uploaded:** Only the latest full resumable checkpoint + metadata sidecar. NOT all retained checkpoints.
- **Upload time:** ~3.6 GB at ~50 MB/s (Kaggle network) ≈ **~72 seconds**. At ~10 MB/s (pessimistic) ≈ **~6 minutes**. Both acceptable.
- **Download time at session start:** Same as upload time. Budget ~2-6 minutes.
- **Repo history growth:** Overwriting the same file in HF Hub still keeps the old versions in Git LFS history. Since each upload is ~3.6 GB, cumulative storage will grow by ~3.6 GB per session (e.g., after 20 sessions, the repo will consume ~72 GB of LFS storage). **Mitigation:** To prevent unbounded growth, we will either use the `HfApi().super_squash_history()` method to compress the commit history, or periodically create a fresh repo. `super_squash_history` is the preferred automated approach.
- **Client-side history:** Using `huggingface_hub.upload_file()` and `hf_hub_download()` avoids cloning the git history locally, entirely sidestepping local disk bloat on Kaggle.
- **Storage limit:** HF Hub free tier allows unlimited public repos with no per-file size limit for LFS. — ASSUMED (We assume 70+ GB of cumulative checkpoints won't trigger rate limits or blocks before squashing).

### 7.4 Fallback if Kaggle Internet is Unavailable

If HF Hub upload fails (network error, HF outage):

1. **Primary fallback:** Save checkpoint to `/kaggle/working/` and commit the notebook output. Kaggle preserves notebook output (up to 20 GB) across sessions as a "Version." The checkpoint can be loaded as an input dataset in the next session.
2. **Secondary fallback:** Save to Kaggle's `/kaggle/tmp/` and manually download via the Kaggle UI if the session is still interactive.
3. **Retry logic:** The upload function retries 3 times with exponential backoff (5s, 15s, 45s) before falling back.
4. **Internet-off mode:** If Kaggle disables internet during the session (some competition notebooks do this), the notebook must pre-download checkpoints and data during the internet-on phase. Training notebooks are NOT competition notebooks, so internet should remain available, but we guard against it.

### 7.5 Checkpoint Round-Trip Protocol

```
Session N (produce checkpoint):
  1. Train → save checkpoint_stepK.pt to /kaggle/working/
  2. Verify checkpoint integrity (size + hash check)
  3. Upload to HF Hub via huggingface_hub.upload_file()
  4. On failure: retry 3x, then fall back to Kaggle output persistence
  5. Log upload status + size + duration

Session N+1 (consume checkpoint):
  1. Download checkpoint via hf_hub_download() [single file, no git clone]
  2. Verify integrity (hash check against sidecar metadata)
  3. If corrupt: re-download; if still corrupt: check Kaggle output for fallback
  4. Load into training state
  5. Verify: model outputs match expected (forward pass on known input)
```

## 8. Multi-Session Training Protocol

> **ESTIMATE.** Session counts and token coverage are derived from throughput tables. Real values depend on measured tok/s from Phase 6.

**Throughput basis (Target MFU=25%, 1×T4):** 9,028 tok/s × 10.5h × 3600 s/h ≈ **341M tokens per effective session**.

**Session map (3B token target):**

| Sessions | Tokens per session (est.) | Cumulative tokens | Sessions needed for 1B | Sessions needed for 3B |
|----------|--------------------------|-------------------|------------------------|------------------------|
| Plausible Low (10%, 3,611 tok/s) | ~136M | — | ~7-8 sessions | ~22 sessions |
| Plausible Base (15%, 5,417 tok/s) | ~205M | — | ~5 sessions | ~15 sessions |
| Conservative (20%, 7,222 tok/s) | ~273M | — | ~4 sessions | ~11 sessions |
| Target (25%, 9,028 tok/s) | ~341M | — | ~3 sessions | ~9 sessions |
| Optimistic High (30%, 10,833 tok/s) | ~409M | — | ~3 sessions | ~7-8 sessions |

**Committed milestone:** 1B tokens (~3 sessions at 25% MFU, more at lower MFU). **3B tokens is conditional** on Phase 6 benchmark confirming sufficient throughput.

**Step reference (at 524K tokens/step):**
- 1B tokens ≈ 1,907 steps
- 3B tokens ≈ 5,722 steps

Each session:
1. Verify GPU, CUDA, PyTorch versions match expectations
2. Download checkpoint from HF Hub (or Kaggle output fallback)
3. Verify checkpoint integrity (hash check)
4. Resume training with `--resume latest`
5. Train until wall-clock limit or token budget
6. Save checkpoint + upload to HF Hub
7. Log session summary

**Between sessions:** Review logs, check for anomalies, update STATUS.md.
