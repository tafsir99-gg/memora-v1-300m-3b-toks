# Memora v1 — Project Status

> Single source of truth for project state. Updated at every phase boundary and after every major subsystem.

## Current Phase

**Phase 4 — IMPLEMENTATION** (Stage 2 Complete. WAITING FOR REVIEW before Stage 3)

## Approval State

- **Plan approved by Owner:** YES
- **Date:** 2026-10-08
- **Owner's exact approving words:** "Approved as amended. This is my explicit approval to implement the plan in docs/, with the amendments below."
- **Amendments applied:** Rev 4 (see Change Log below)

## Change Log

| Date | Change |
|------|--------|
| 2026-10-01 | Phase 0 inspection + Phase 1/2 initial plan written |
| 2026-10-01 | **Rev 2:** Corrected 2×T4 quota rate (ASSUMED 2×); added 10-15% MFU pessimistic scenarios; added stop/extend decision rules; expanded checkpoint persistence details; added verified claims registry; resolved open questions Q1-Q3; added Python version strategy |
| 2026-10-01 | **Rev 3:** Pinned Python to `>=3.11.11, <3.15`; clarified 2×T4 quota rate as ASSUMED; updated dataset redistribution to ASSUMED; documented HF Hub `super_squash_history`; relabeled MFU as "plausible"; 1B as committed milestone (3B conditional) |
| 2026-10-08 | **Rev 4 (Owner amendments):** Warmup changed from 2,000 steps to ~131M tokens (~250 steps); removed wrong "~200K total steps" claim; added WSD scheduler (`cosine`/`wsd` config-driven); added 1B-under-cosine-not-annealed note; global batch size config-driven with 262K as Phase 6 candidate; section 8 session table recomputed from throughput tables, labeled ESTIMATE; DATA_PLAN Option B marked NOT ALLOWED; tokenizer sample corrected to ~0.5-2 GB (not 5-10M documents); hash split applied before tokenizer sampling; Stage 2 length filter uses character-count proxy; CPU quota marked ASSUMED; D21/C1 changed to ASSUMED; R10 updated to `>=3.11.11,<3.15`; REPRODUCIBILITY 2.2 python version fixed; Kaggle torch constraint documented |

## Summary

| Category | Status |
|----------|--------|
| Phase 0 — Inspect | ✅ COMPLETE |
| Phase 1 — Architecture Design | ✅ COMPLETE (documented in ARCHITECTURE.md) |
| Phase 2 — Write the Plan | ✅ COMPLETE (Rev 4 — Owner approved) |
| Phase 3 — Stop for approval | ✅ COMPLETE |
| Phase 4 — Implementation Stage 1 (repo foundation) | ✅ COMPLETE |
| Phase 4 — Implementation Stage 2 (config system) | ✅ COMPLETE |
| Phase 4 — Implementation Stage 3+ | ⬜ NOT STARTED (STOPPED for Owner review) |
| Phase 5 — Staged Validation | ⬜ NOT STARTED |
| Phase 6 — Benchmark & Pre-commit | ⬜ NOT STARTED |
| Phase 7 — Full Training | ⬜ NOT STARTED |
| Phase 8 — Integration & Docs | ⬜ NOT STARTED |
| Phase 9 — Final Audit | ⬜ NOT STARTED |

## Completed Items

- [x] Repository inspection (git history, file structure, tooling)
- [x] Environment audit (Python 3.14 local, Kaggle 3.11.11, uv, MacBook Air M2 16GB, ~51GB free disk)
- [x] Kaggle constraint verification (30h/week, 12h sessions, 2×T4 quota rate = ASSUMED 2×)
- [x] T4 capability verification (no bf16, no FlashAttention-2, sm_75, 65 TFLOPS fp16)
- [x] Compute feasibility estimate (ESTIMATE, pending Phase 6 measurement)
- [x] Parameter count exploration for candidate architectures
- [x] Architecture design document (docs/ARCHITECTURE.md)
- [x] Implementation plan (docs/IMPLEMENTATION_PLAN.md)
- [x] Training plan with plausible MFU scenarios and decision rules (docs/TRAINING_PLAN.md)
- [x] Data plan (docs/DATA_PLAN.md)
- [x] Reproducibility guide with venv instructions (docs/REPRODUCIBILITY.md)
- [x] Risk register with verified claims and source URLs (docs/RISKS_AND_OPEN_QUESTIONS.md)
- [x] Resolved Owner questions: FineWeb-Edu confirmed, GELU confirmed, HF repo name confirmed
- [x] Rev 4 amendments applied and committed
- [x] Stage 1: Repository Foundation (pyproject.toml, ruff, mypy, CONTRIBUTING.md, CODE_OF_CONDUCT.md)
- [x] Stage 2: Configuration System (Pydantic models, YAML configs, tests)

## In-Progress Items

- None. Stopped for Owner review.

## Not Started Items

- [ ] Stage 3+: All further implementation (WAITING for Owner review)
- [ ] All training (blocked on implementation)
- [ ] All evaluation (blocked on training)

## Binding Rules (Active)

1. **Never** `git push`. Never commit secrets or local absolute paths.
2. **No** dataset downloads, no training runs. Mac handles only tiny test data.
3. Install only into project `.venv` (`uv venv --python 3.11.11 .venv`). Ask before deleting old venv.
4. Only packages listed in the plan. Ask before each new install.
5. If implementation reveals a plan problem: stop and propose an amendment.
6. Never fabricate numbers: mark VERIFIED / ASSUMED / UNKNOWN with source and date.
7. **KAGGLE TORCH:** Do not reinstall or upgrade torch on Kaggle. Use the image's preinstalled version.

## Verification Status

| Item | Status | Source | Date |
|------|--------|--------|------|
| Kaggle 30h/week GPU quota | VERIFIED | kaggle.com forums, luminoai.in | 2026-10-01 |
| Kaggle 12h session limit | VERIFIED | kaggle.com forums, clusy.io | 2026-10-01 |
| A Kaggle 2×T4 session consumes GPU quota at exactly 2× the wall-clock rate | ASSUMED (Pending Phase 6 check) | luminoai.in, kaggle.com forums | 2026-10-01 |
| T4 lacks native bf16 | VERIFIED | NVIDIA T4 datasheet, sm_75 | 2026-10-01 |
| T4 lacks FlashAttention-2 | VERIFIED | FlashAttention GitHub, requires Ampere+ | 2026-10-01 |
| T4 fp16 peak: 65 TFLOPS | VERIFIED | NVIDIA T4 datasheet | 2026-10-01 |
| Kaggle output limit: 20 GB | VERIFIED | kaggle.com/docs | 2026-10-01 |
| Kaggle dataset limit: 200 GB | VERIFIED | kaggle.com/docs | 2026-10-01 |
| Kaggle Python version: 3.11.11 | VERIFIED | github.com/Kaggle/docker-python | 2026-10-01 |
| PyTorch supports Python 3.14 (PyTorch 2.10+) | VERIFIED | pytorch.org | 2026-10-01 |
| FineWeb-Edu license: ODC-By 1.0 | VERIFIED | HuggingFace dataset card | 2026-10-01 |
| FineWeb-Edu sample-10BT: ~10B tokens, ~9.7M documents | VERIFIED | HuggingFace dataset card | 2026-10-01 |
| HF Hub free tier: unlimited public repos | ASSUMED | General knowledge; not re-verified | 2026-10-01 |
| Kaggle CPU sessions do not consume GPU quota | ASSUMED | kaggle.com forums | 2026-10-01 |
| Feasibility: 1B tokens in ~25-77h (1×T4) depending on MFU | ESTIMATE | First-principles calc, 10-30% MFU range | 2026-10-01 |
| T4 MFU for 300M model | UNKNOWN-NEEDS-MEASUREMENT | Depends on model, batch size, SDPA backend | — |

## Resolved Owner Decisions

| Question | Decision | Date |
|----------|----------|------|
| Dataset: FineWeb-Edu sample-10BT | **Yes** | 2026-10-01 |
| FFN type: Standard GELU | **Yes** | 2026-10-01 |
| HF repo: `tafsir99-gg/memora-v1-300m` | **Yes, if available** | 2026-10-01 |
| Plan Rev 4 approved? | **Yes, approved as amended** | 2026-10-08 |

## Open Questions Awaiting Owner

1. **Kaggle account readiness:** Is the Kaggle account set up with GPU access and Secrets capability (for HF_TOKEN)?

## GPU Hours Used

| Session | Hours | Tokens | Purpose |
|---------|-------|--------|---------|
| — | 0 | 0 | No GPU time consumed yet |

## Next Steps

1. ✅ Commit Rev 4 doc amendments: `"Plan rev 4: Owner amendments"`
2. 🔄 Implement Stage 1 (repository foundation) — then run tests and commit
3. ⬜ Implement Stage 2 (configuration system) — then run tests and commit
4. ⬜ **STOP** — wait for Owner review before Stage 3 (tokenizer)
