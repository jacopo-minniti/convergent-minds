#!/usr/bin/env python
"""
Utility script to visualize how the locality bias changes GPT-2 attentions.

The script runs a prompt through several model variants (different decay rates,
untrained vs. pretrained) and saves heatmaps plus distance profiles to `dumps/`.
"""

import argparse
import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from brainscore_language.models.locality_gpt.model import LocalityGPT2Attention  # noqa: E402


def parse_decay_rate(decay_str: str) -> Optional[float]:
    if decay_str is None:
        return None
    cleaned = decay_str.strip().lower()
    if cleaned in {"none", "baseline", "default"}:
        return None
    return float(cleaned)


def build_model(model_name: str, decay_rate: Optional[float], untrained: bool, device: torch.device):
    if untrained:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    if decay_rate is not None:
        for idx, layer in enumerate(model.transformer.h):
            layer.attn = LocalityGPT2Attention(model.config, layer_idx=idx, decay_rate=decay_rate)

    model.to(device)
    model.eval()
    return model


def collect_average_attention(model, tokenizer, prompt: str, max_length: int, device: torch.device):
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)

    attn_stack = torch.stack(outputs.attentions, dim=0)
    attn_avg = attn_stack.mean(dim=2).squeeze(1).cpu()
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    return attn_avg, tokens


def plot_heatmap(matrix: torch.Tensor, tokens: List[str], title: str, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.6), 5))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def compute_distance_profile(matrix: torch.Tensor) -> List[float]:
    seq_len = matrix.shape[-1]
    distances = torch.arange(seq_len)
    dist_matrix = torch.abs(distances.unsqueeze(0) - distances.unsqueeze(1))
    profile = []
    for dist in range(seq_len):
        mask = dist_matrix == dist
        if mask.any():
            profile.append(matrix[mask].mean().item())
        else:
            profile.append(float("nan"))
    return profile


def plot_distance_profile(profile: List[float], title: str, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(profile)), profile, marker="o")
    ax.set_xlabel("|i - j| (token distance)")
    ax.set_ylabel("Average attention weight")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def visualize(decay_labels: List[str], args):
    if not args.prompt:
        raise ValueError("Please provide --prompt to visualize attention.")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for label in decay_labels:
        decay_rate = parse_decay_rate(label)
        model = build_model(args.model_name, decay_rate, args.untrained, device)
        attn_layers, tokens = collect_average_attention(model, tokenizer, args.prompt, args.max_length, device)
        avg_matrix = attn_layers.mean(dim=0)
        profile = compute_distance_profile(avg_matrix)
        label_str = "baseline" if decay_rate is None else f"decay_{decay_rate}"
        results.append(
            {
                "label": label_str,
                "matrix": avg_matrix,
                "profile": profile,
                "tokens": tokens,
            }
        )
        basepath = os.path.join(args.save_dir, f"{args.output_prefix}_{label_str}")
        plot_heatmap(avg_matrix, tokens, f"{args.model_name} ({label_str})", f"{basepath}_heatmap.png")
        plot_distance_profile(profile, f"Distance profile ({label_str})", f"{basepath}_distance.png")

    if args.compare_to_first and len(results) > 1:
        reference = results[0]["matrix"]
        tokens_for_plot = results[0]["tokens"]
        for current in results[1:]:
            diff = current["matrix"] - reference
            basepath = os.path.join(
                args.save_dir,
                f"{args.output_prefix}_{current['label']}_vs_{results[0]['label']}",
            )
            plot_heatmap(
                diff,
                tokens_for_plot,
                f"Difference: {current['label']} - {results[0]['label']}",
                f"{basepath}_difference.png",
            )


def main():
    parser = argparse.ArgumentParser(description="Visualize GPT-2 attention patterns under locality modifications.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to feed into the model.")
    parser.add_argument(
        "--decay-rates",
        nargs="+",
        default=["none", "1.0", "-0.3"],
        help="List of decay rates; use 'none' for the vanilla model.",
    )
    parser.add_argument("--model-name", type=str, default="gpt2", help="HuggingFace model identifier.")
    parser.add_argument("--untrained", action="store_true", help="Use untrained random weights.")
    parser.add_argument("--max-length", type=int, default=64, help="Maximum tokenized prompt length.")
    parser.add_argument("--save-dir", type=str, default="dumps", help="Directory to store plots.")
    parser.add_argument("--output-prefix", type=str, default="attention", help="Prefix for the saved files.")
    parser.add_argument(
        "--compare-to-first",
        action="store_true",
        help="If set, saves difference heatmaps relative to the first decay rate.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device string passed to torch.device().")
    args = parser.parse_args()

    visualize(args.decay_rates, args)


if __name__ == "__main__":
    main()
