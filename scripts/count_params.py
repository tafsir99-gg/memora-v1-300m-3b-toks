"""Parameter counter script.

Builds the model on the `meta` device to avoid memory allocation, computes
the parameter count breakdown, and asserts it matches the analytic formula.
"""

import argparse
import sys

import torch

from memora.config import MemoraConfig
from memora.model import MemoraModel


def count_params(config: MemoraConfig) -> None:
    # 1. Analytic Count
    m_cfg = config.model
    embedding = m_cfg.vocab_size * m_cfg.d_model

    # per layer
    qkv = 3 * m_cfg.d_model * m_cfg.d_model
    proj = m_cfg.d_model * m_cfg.d_model
    if m_cfg.use_bias:
        qkv += 3 * m_cfg.d_model
        proj += m_cfg.d_model
    attention = qkv + proj

    fc1 = m_cfg.d_model * m_cfg.d_ff
    fc2 = m_cfg.d_ff * m_cfg.d_model
    if m_cfg.use_bias:
        fc1 += m_cfg.d_ff
        fc2 += m_cfg.d_model
    ffn = fc1 + fc2

    norms = 4 * m_cfg.d_model  # ln_1 (weight+bias) + ln_2 (weight+bias)
    per_layer = attention + ffn + norms

    all_layers = m_cfg.n_layers * per_layer
    final_norm = 2 * m_cfg.d_model

    lm_head = m_cfg.vocab_size * m_cfg.d_model

    total_untied = embedding + all_layers + final_norm + lm_head
    total_tied = embedding + all_layers + final_norm

    analytic_total = total_tied if m_cfg.tie_embeddings else total_untied

    # 2. PyTorch Count (meta device)
    with torch.device("meta"):
        model = MemoraModel(m_cfg)

    actual_total = sum(p.numel() for p in model.parameters())

    # Avoid double-counting tied weights when computing breakdowns from model params
    seen = set()
    actual_emb = 0
    actual_attn = 0
    actual_ffn = 0
    actual_norms = 0
    actual_final_norm = 0
    actual_lm_head = 0

    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))

        if name.startswith("embedding"):
            actual_emb += p.numel()
        elif "attn" in name:
            # We just sum all layers' attention
            actual_attn += p.numel()
        elif "mlp" in name:
            actual_ffn += p.numel()
        elif name.endswith("ln_1.weight") or name.endswith("ln_1.bias") or \
             name.endswith("ln_2.weight") or name.endswith("ln_2.bias"):
            actual_norms += p.numel()
        elif name.startswith("ln_f"):
            actual_final_norm += p.numel()
        elif name.startswith("lm_head"):
            actual_lm_head += p.numel()

    # Per layer average
    actual_attn_per_layer = actual_attn // m_cfg.n_layers
    actual_ffn_per_layer = actual_ffn // m_cfg.n_layers
    actual_norms_per_layer = actual_norms // m_cfg.n_layers

    # 3. Report
    print(f"--- Parameter Count Breakdown ({m_cfg.d_model} d_model, {m_cfg.n_layers} layers) ---")
    print(f"Embedding:         {actual_emb:,}")
    print(f"Per-layer attn:    {actual_attn_per_layer:,}")
    print(f"Per-layer FFN:     {actual_ffn_per_layer:,}")
    print(f"Per-layer norms:   {actual_norms_per_layer:,}")
    print(f"All {m_cfg.n_layers} layers:       {actual_attn + actual_ffn + actual_norms:,}")
    print(f"Final norm:        {actual_final_norm:,}")
    print(f"LM head (untied):  {actual_lm_head:,}" if actual_lm_head > 0 else "LM head:           (tied to embedding)")
    print("-" * 45)
    print(f"TOTAL:             {actual_total:,}")
    print(f"Analytic total:    {analytic_total:,}")

    untied_total = sum(p.numel() for name, p in model.named_parameters())
    print(f"(Tied vs Untied):  {actual_total:,} tied, {untied_total:,} untied")

    if actual_total != analytic_total:
        print(f"ERROR: Model parameter count {actual_total} != analytic count {analytic_total}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count model parameters")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = MemoraConfig.from_yaml(args.config)
    count_params(config)


if __name__ == "__main__":
    main()
