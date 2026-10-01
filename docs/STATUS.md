# Memora v1 — Project Status

> Single source of truth for project state. Updated at every phase boundary and after every major subsystem.

## Current Phase

**Phase 3 — STOP** (Plan written, awaiting Owner review — second iteration)

## Approval State

- **Plan approved by Owner:** NO
- **Date:** —
- **Owner's exact approving words:** —

## Change Log

| Date | Change |
|------|--------|
| 2026-10-01 | Phase 0 inspection + Phase 1/2 initial plan written |
| 2026-10-01 | **Rev 2:** Corrected 2×T4 quota rate (2×, not 1×); added 10-15% MFU pessimistic scenarios; added stop/extend decision rules; expanded checkpoint persistence details (sizes, upload/download times, HF history, fallback); added verified claims registry with source URLs and dates; resolved open questions Q1-Q3; added Python version strategy with local venv instructions; confirmed default 1×T4 recommendation |

## Summary

| Category | Status |
|----------|--------|
| Phase 0 — Inspect | ✅ COMPLETE |
| Phase 1 — Architecture Design | ✅ COMPLETE (documented in ARCHITECTURE.md) |
| Phase 2 — Write the Plan | ✅ COMPLETE (Rev 2 — corrections applied per Owner feedback) |
| Phase 3 — Stop for approval | 🔄 CURRENT |
| Phase 4 — Implementation | ⬜ NOT STARTED (blocked on approval) |
| Phase 5 — Staged Validation | ⬜ NOT STARTED |
| Phase 6 — Benchmark & Pre-commit | ⬜ NOT STARTED |
| Phase 7 — Full Training | ⬜ NOT STARTED |
| Phase 8 — Integration & Docs | ⬜ NOT STARTED |
| Phase 9 — Final Audit | ⬜ NOT STARTED |

## Completed Items

- [x] Repository inspection (git history, file structure, tooling)
- [x] Environment audit (Python 3.14 local, Kaggle 3.11, uv, MacBook Air M2 16GB, ~51GB free disk)
- [x] Kaggle constraint verification (30h/week, 12h sessions, **2×T4 = 2× quota rate**)
- [x] T4 capability verification (no bf16, no FlashAttention-2, sm_75, 65 TFLOPS fp16)
- [x] Compute feasibility estimate (ESTIMATE, pending Phase 6 measurement)
- [x] Parameter count exploration for candidate architectures
- [x] Architecture design document (docs/ARCHITECTURE.md)
- [x] Implementation plan (docs/IMPLEMENTATION_PLAN.md)
- [x] Training plan with pessimistic scenarios and decision rules (docs/TRAINING_PLAN.md)
- [x] Data plan (docs/DATA_PLAN.md)
- [x] Reproducibility guide with venv instructions (docs/REPRODUCIBILITY.md)
- [x] Risk register with verified claims and source URLs (docs/RISKS_AND_OPEN_QUESTIONS.md)
- [x] Resolved Owner questions: FineWeb-Edu confirmed, GELU confirmed, HF repo name confirmed

## In-Progress Items

- [ ] Owner review of updated plan documents (Rev 2)

## Not Started Items

- [ ] All implementation (blocked on approval)
- [ ] All training (blocked on implementation)
- [ ] All evaluation (blocked on training)

## Verification Status

| Item | Status | Source | Date |
|------|--------|--------|------|
| Kaggle 30h/week GPU quota | VERIFIED | kaggle.com forums, luminoai.in | 2026-10-01 |
| Kaggle 12h session limit | VERIFIED | kaggle.com forums, clusy.io | 2026-10-01 |
| **2×T4 = 2× quota rate** | **VERIFIED** | **luminoai.in, kaggle.com forums** | **2026-10-01** |
| T4 lacks native bf16 | VERIFIED | NVIDIA T4 datasheet, sm_75 | 2026-10-01 |
| T4 lacks FlashAttention-2 | VERIFIED | FlashAttention GitHub, requires Ampere+ | 2026-10-01 |
| T4 fp16 peak: 65 TFLOPS | VERIFIED | NVIDIA T4 datasheet | 2026-10-01 |
| Kaggle output limit: 20 GB | VERIFIED | kaggle.com/docs | 2026-10-01 |
| Kaggle dataset limit: 200 GB | VERIFIED | kaggle.com/docs | 2026-10-01 |
| Kaggle Python version: 3.11 | VERIFIED | github.com/Kaggle/docker-python | 2026-10-01 |
| PyTorch supports Python 3.14 | VERIFIED | pytorch.org | 2026-10-01 |
| FineWeb-Edu license: ODC-By 1.0 | VERIFIED | HuggingFace dataset card | 2026-10-01 |
| FineWeb-Edu sample-10BT: ~10B tokens | VERIFIED | HuggingFace dataset card | 2026-10-01 |
| HF Hub free tier: unlimited public repos | ASSUMED | General knowledge; not re-verified | 2026-10-01 |
| Feasibility: 3B tokens in ~77-231h (1×T4) | ESTIMATE | First-principles calc, 10-30% MFU range | 2026-10-01 |
| T4 MFU for 300M model | UNKNOWN-NEEDS-MEASUREMENT | Depends on model, batch size, SDPA backend | — |
| DDP scaling efficiency on 2×T4 PCIe | UNKNOWN-NEEDS-MEASUREMENT | No NVLink, PCIe bandwidth limited | — |
| torch.compile benefit on T4 | UNKNOWN-NEEDS-MEASUREMENT | Must benchmark | — |

## Resolved Owner Decisions

| Question | Decision | Date |
|----------|----------|------|
| Dataset: FineWeb-Edu sample-10BT | **Yes** | 2026-10-01 |
| FFN type: Standard GELU | **Yes** | 2026-10-01 |
| HF repo: `tafsir99-gg/memora-v1-300m` | **Yes, if available** | 2026-10-01 |

## Open Questions Awaiting Owner

1. **Kaggle account readiness:** Is the Kaggle account set up with GPU access and Secrets capability (for HF_TOKEN)?

## GPU Hours Used

| Session | Hours | Tokens | Purpose |
|---------|-------|--------|---------|
| — | 0 | 0 | No GPU time consumed yet |

## Next Step

**Owner must review updated docs/ (Rev 2) and reply "Approved, proceed" or request further changes.**
