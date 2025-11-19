#!/usr/bin/env python
"""
Utility script to visualize how the locality bias changes GPT-2 attentions.

The script runs a prompt through several model variants (different decay rates,
untrained vs. pretrained) and saves heatmaps plus distance profiles to `dumps/`.
This helps inspect whether supposedly different settings actually yield the
same attention structure, complementing the scalar benchmark scores.
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
from brainscore_language import load_benchmark  # noqa: E402

from brainscore_language.models.locality_gpt.model import LocalityGPT2Attention  # noqa: E402


def parse_decay_rate(decay_str: str) -> Optional[float]:
    """Returns None if string is 'none', otherwise a float."""
    if decay_str is None:
        return None
    cleaned = decay_str.strip().lower()
    if cleaned in {"none", "baseline", "default"}:
        return None
    return float(cleaned)


def build_model(model_name: str, decay_rate: Optional[float], untrained: bool, device: torch.device):
    """Loads GPT-2 and optionally injects LocalityGPT2Attention layers."""
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
    """Runs the prompt and returns attention tensors averaged over heads."""
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)

    # Each element of outputs.attentions is (batch, heads, seq, seq)
    attn_stack = torch.stack(outputs.attentions, dim=0)  # (layers, batch, heads, seq, seq)
    attn_avg = attn_stack.mean(dim=2).squeeze(1).cpu()  # (layers, seq, seq)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    return attn_avg, tokens


def fetch_prompt_from_benchmark(benchmark_name: str, index: int, preferred_column: Optional[str]) -> str:
    """Loads a benchmark and returns one of its stimulus sentences."""
    benchmark = load_benchmark(benchmark_name)

    # Prefer using the NeuroidAssembly directly if present (e.g., Pereira benchmarks expose .data).
    assembly = getattr(benchmark, "data", None)
    if assembly is not None and "stimulus" in assembly.coords:
        texts = assembly["stimulus"].values
        if len(texts) == 0:
            raise ValueError(f"Benchmark {benchmark_name} has an empty stimulus coordinate.")
        return str(texts[index % len(texts)])

    # Fall back to a stored StimulusSet/DataFrame if available.
    stimulus_set = None
    target = getattr(benchmark, "_target", None)
    if target is not None and hasattr(target, "stimulus_set"):
        stimulus_set = target.stimulus_set
    if stimulus_set is None:
        raise ValueError(
            f"Could not access stimuli for benchmark {benchmark_name}. "
            "Benchmark object lacks `.data['stimulus']` and `_target.stimulus_set`."
        )

    if not hasattr(stimulus_set, "iloc"):
        raise ValueError("Stimulus set is not DataFrame-like; cannot extract prompt automatically.")

    if not len(stimulus_set):
        raise ValueError(f"Benchmark {benchmark_name} stimulus set is empty.")

    column_candidates = []
    if preferred_column:
        column_candidates.append(preferred_column)
    column_candidates.extend(["sentence", "text", "passage", "content", "stimulus"])

    chosen_column = None
    for column in column_candidates:
        if column in stimulus_set.columns:
            chosen_column = column
            break

    if chosen_column is None:
        raise ValueError(
            f"Could not find a text column in stimulus set. Available columns: {list(stimulus_set.columns)}",
        )

    row = stimulus_set.iloc[index % len(stimulus_set)]
    text = row[chosen_column]
    if not isinstance(text, str):
        raise ValueError(f"Selected column '{chosen_column}' does not contain strings (value: {text}).")
    return text


def plot_heatmap(matrix: torch.Tensor, tokens: List[str], title: str, filepath: str):
    """Saves an attention heatmap averaged across layers."""
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
    """Averages attention weights by absolute token distance."""
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
    """Saves a line plot of average attention weight vs. token distance."""
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
    prompt = args.prompt
    if prompt is None:
        if not args.benchmark_name:
            raise ValueError("Provide either --prompt or --benchmark-name.")
        prompt = fetch_prompt_from_benchmark(
            benchmark_name=args.benchmark_name,
            index=args.benchmark_index,
            preferred_column=args.benchmark_column,
        )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, truncation_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for label in decay_labels:
        decay_rate = parse_decay_rate(label)
        model = build_model(args.model_name, decay_rate, args.untrained, device)
        attn_layers, tokens = collect_average_attention(model, tokenizer, prompt, args.max_length, device)
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
    parser.add_argument("--prompt", type=str, help="Text prompt to feed into the model.")
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
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default=None,
        help="If set, pulls a prompt from this brainscore benchmark instead of --prompt.",
    )
    parser.add_argument(
        "--benchmark-index",
        type=int,
        default=0,
        help="Row index from the benchmark stimulus set to use as prompt.",
    )
    parser.add_argument(
        "--benchmark-column",
        type=str,
        default=None,
        help="Optional column name in the stimulus set that contains the text (defaults to common names).",
    )
    args = parser.parse_args()

    visualize(args.decay_rates, args)


if __name__ == "__main__":
    main()
