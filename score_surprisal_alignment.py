import os
import sys
# Force local import of brainscore by adding the current directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from brainscore import load_benchmark, ArtificialSubject
from brainscore.model_helpers.huggingface import HuggingfaceSubject, get_layer_names
from brainscore.utils.ceiling import ceiling_normalize
from tqdm import tqdm
import json
import datetime
from sklearn.model_selection import GroupKFold, KFold



def compute_contextual_surprisal(model, tokenizer, sentence, context_sentences, device='cuda'):
    """
    Computes TOTAL sentence surprisal: s(x) = sum(-log p(y_t | y_<t, context))
    """
    model.eval()
    
    # Construct input: [Context] [Sentence]
    # We want to loss ONLY on [Sentence].
    
    # 1. Tokenize context and sentence separately to find lengths
    if context_sentences:
        context_str = " ".join(context_sentences)
        # Add separator if needed, but usually space is enough or EOS? 
        # GPT2 just continues. "Context. Sentence"
        full_text = f"{context_str} {sentence}"
    else:
        context_str = ""
        full_text = sentence
        
    inputs = tokenizer(full_text, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    
    # Create labels: -100 for context, regular IDs for sentence
    labels = input_ids.clone()
    
    if context_sentences:
        # We need to know where the sentence starts in token space.
        # This is tricky with BPE. 
        # Strategy: Tokenize context alone, see length.
        context_tokens = tokenizer(context_str, return_tensors='pt')['input_ids']
        # The sentence tokens start after context tokens. 
        # Note: " " might be merged. 
        # Safer way: 
        # Run tokenizer on context, len is N. 
        # Labels[:N] = -100
        # But " " + "sentence" might merge the space with first word of sentence?
        # Let's hope the space provided in f"{context_str} {sentence}" cleanly separates or aligns reasonably.
        # Given it's a correlation study, small boundary errors are acceptable compared to no context.
        
        c_len = context_tokens.shape[1]
        
        # Determine effective context length in full_text
        # If full_text tokens are shorter than context+sentence tokens sum, there was a merge.
        # Usually context ends with punctuation.
        # Let's just mask the first c_len tokens.
        if c_len < labels.shape[1]:
            labels[:, :c_len] = -100
        else:
            # Fallback if something weird happened (e.g. empty sentence?)
            labels[:, :] = -100 # Should not happen
            
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        # outputs.loss is the MEAN NLL strictly over tokens where label != -100
        # So Total Surprisal = loss * num_target_tokens
        
        # Count target tokens
        num_target_tokens = (labels != -100).sum().item()
        
        if num_target_tokens > 0:
            total_surprisal = outputs.loss.item() * num_target_tokens
        else:
            total_surprisal = 0.0
            
    return total_surprisal

def compute_static_baseline(surprisals, topics):
    """
    b_static(c) = mean(s_i for s_i in topic c)
    relative_static = s_i - b_static(c_i)
    """
    unique_topics = np.unique(topics)
    topic_means = {}
    for topic in unique_topics:
        indices = [i for i, t in enumerate(topics) if t == topic]
        topic_means[topic] = np.mean([surprisals[i] for i in indices])
        
    relative_surprisals = []
    for i, s in enumerate(surprisals):
        topic = topics[i]
        relative_surprisals.append(s - topic_means[topic])
        
    return np.array(relative_surprisals)

def compute_moving_baseline(surprisals, topics):
    """
    b_move(i) = mean(s_j for j < i and topic(j) == topic(i))
    relative_move = s_i - b_move(i)
    """
    relative_surprisals = []
    
    topic_history = {t: [] for t in np.unique(topics)}
    
    # Precompute static means for priors
    static_means = {}
    for topic in np.unique(topics):
        indices = [i for i, t in enumerate(topics) if t == topic]
        static_means[topic] = np.mean([surprisals[i] for i in indices])
    
    for i, s in enumerate(surprisals):
        topic = topics[i]
        history = topic_history[topic]
        
        if len(history) > 0:
            baseline = np.mean(history)
        else:
            baseline = static_means[topic] # Prior
        
        relative_surprisals.append(s - baseline)
        topic_history[topic].append(s)
        
    return np.array(relative_surprisals)



def main():
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--localize", action="store_true")
    parser.add_argument("--num-units", type=int, default=256)
    parser.add_argument("--benchmark", default="Pereira2018.243sentences-partialr2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--topic_wise_cv", action="store_true", default=True)
    parser.add_argument("--no_topic_wise_cv", dest="topic_wise_cv", action="store_false")
    args = parser.parse_args()

    # Device setup
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    if not args.save_path:
        args.save_path = "data/scores/surprisal_alignment_debug"
    os.makedirs(args.save_path, exist_ok=True)

    # 1. Load Model
    model_path = args.model
    # Resolve possible paths
    if os.path.isabs(args.model) and os.path.isdir(args.model):
        model_path = args.model
    elif os.path.isdir(args.model):
        model_path = args.model
    else:
        possible_path = os.path.join(os.getcwd(), "models", args.model)
        if os.path.isdir(possible_path):
            model_path = possible_path
        else:
            model_path = args.model
            
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if args.untrained:
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
    model.to(device)
    model.eval()

    # 2. Load Benchmark & Data
    print(f"Loading benchmark: {args.benchmark}")
    benchmark = load_benchmark(args.benchmark)
    
    if not hasattr(benchmark, 'data'):
        raise ValueError("Benchmark must have 'data' attribute loaded.")
    
    assembly = benchmark.data
    print(f"Assembly coords: {list(assembly.coords.keys())}")
    print(f"Assembly attrs: {list(assembly.attrs.keys())}")
    
    stimuli = assembly['stimulus'] 
    
    sentences = stimuli['sentence'].values
    passage_labels = assembly['passage_label'].values
    # In BrainScore assemblies, there might be 'stimulus_id' coordinate
    stimulus_ids = assembly['stimulus_id'].values
    
    print("Computing Surprisals in Presentation Order...")
    
    # We iterate over the distinct presentations in the assembly
    # Note: assembly is (presentation x neuroid). 
    # The 'presentation' dimension is aligned with sentences/passage_labels.
    
    ordered_surprisals = []
    ordered_topics = []
    
    # Context cache: passage_label -> list of previous sentences
    # This ensures that even if presentation is interleaved (unlikely for Pereira but possible),
    # we maintain the narrative context for that passage.
    passage_contexts = {p: [] for p in np.unique(passage_labels)}
    
    # Cache surprisals to avoid re-computing for identical (sentence, context) pairs?
    # Context grows, so cache hit rate might be low unless we hash the list.
    # Given dataset size (~384 * N_subjects?), it's small enough to just compute.
    # Actually, Pereira is usually ~243 or 384 sentences. 
    # If assembly has repetitions (multiple subjects), we don't want to recompute 10x.
    # We should compute distinct (passage, sentence_idx) items?
    # But "Moving Baseline" depends on presentation order.
    # If the assembly repeats the *same* experiment sequence for 5 subjects, 
    # the "moving baseline" should be the same for all.
    # We can just compute a flat list aligned with assembly.
    
    # Optimization: If the assembly is Subject x Presentation stacked, 
    # it might be huge. 
    # But usually BrainScore Pereira benchmark provides a single averaged assembly 
    # OR a raw assembly with a 'presentation' multi-index.
    # Let's verify size.
    print(f"Assembly shape: {assembly.shape}")
    
    for i, (sid, sentence, topic) in tqdm(enumerate(zip(stimulus_ids, sentences, passage_labels)), total=len(sentences)):
        
        # Current Context
        ctx = passage_contexts[topic]
        
        # Compute Surprisal
        # Note: We compute for EVERY sample. 
        # If this is too slow, we can cache by (sid, len(ctx)).
        s_val = compute_contextual_surprisal(model, tokenizer, sentence, ctx, device)
        
        ordered_surprisals.append(s_val)
        ordered_topics.append(topic)
        
        # Update context
        # We assume the sentences come in reading order for that passage.
        # If the assembly contains REPEATS of the same sentence (e.g. subject 1 read it, subject 2 read it),
        # we shouldn't append to context multiple times for the SAME reading instance.
        # But if it's the concatenated data of multiple subjects...
        # Standard Pereira assembly in BrainScore is usually averaged or provides unique stimuli?
        # If it's unique stimuli, len(assembly) == 243 or 384. 
        # If it's raw, it might be thousands.
        # Ideally we compute unique measures.
        
        # CRITICAL: We need to know if we are "advancing" the text.
        # If the next sample is the SAME sentence (different subject), context shouldn't grow.
        # If the next sample is NEW sentence, context grows.
        
        # Let's check if this is a repeat of the last step for this topic?
        # Or simpler: The context is the list of *unique* preceding sentences in this passage.
        # But we don't know the order of unique sentences a priori unless we sort.
        # User said "respect actual order defined in assembly".
        # If assembly has Subject 1 (Sent A, Sent B) ... Subject 2 (Sent A, sent B)...
        # Then Moving Baseline for Subject 2 Sent A should NOT include Subject 1's sentences.
        
        # Heuristic: 
        # If we see a sentence we've already seen in this context, do we append it? NO.
        # Context is strictly the distinct previous sentences.
        if len(ctx) == 0 or ctx[-1] != sentence:
            passage_contexts[topic].append(sentence)

    
    ordered_surprisals = np.array(ordered_surprisals)
    ordered_topics = np.array(ordered_topics)
    
    print("Computing Baselines...")
    # 1. Raw (Total)
    raw_surprisals = ordered_surprisals
    
    # 2. Static Relative
    static_relative = compute_static_baseline(ordered_surprisals, ordered_topics)
    
    # 3. Moving Relative
    # Computed on the flat array in presentation order
    moving_relative = compute_moving_baseline(ordered_surprisals, ordered_topics)
    
    # 4. Correlation Analysis
    Y = assembly.values    # Y data
    
    # FILTER FOR LANGUAGE NETWORKS
    # NOTE: The benchmark object (Pereira2018) already filters for the language network 
    # and high-reliability voxels in its __init__. 
    # See brainscore/benchmarks/pereira2018/benchmark.py.
    # So Y is *already* the filtered data. We don't need to check for 'atlas' or filter again.
    # This matches the behavior of score_alignment.py which implicitly trusts the benchmark's data.
    
    Y_filtered = Y
    roi_info = "Benchmark Default (Likely Language ROI)"
    print(f"Using {Y.shape[1]} neuroids from benchmark data.")

    # Helper for vectorized correlation
    def compute_correlation(x, Y):
        x_c = x - np.mean(x)
        x_norm = np.linalg.norm(x_c)
        if x_norm == 0:
            return np.zeros(Y.shape[1])
        Y_c = Y - np.mean(Y, axis=0) # (N, V)
        Y_norm = np.linalg.norm(Y_c, axis=0) # (V,)
        Y_norm[Y_norm == 0] = 1e-9
        dot_prod = np.dot(x_c, Y_c) # (V,)
        return dot_prod / (x_norm * Y_norm)

    results = {}
    print("\n=== Calculating Correlations (Contextual + Total + ROI Filter) ===")
    
    # Pack aligned surprisals (they are already aligned by loop order)
    aligned_surprisals = {
        'raw': raw_surprisals,
        'static': static_relative,
        'moving': moving_relative
    }
    
    configs = ['raw', 'static', 'moving']
    
    for config in configs:
        x = aligned_surprisals[config]
        corrs = compute_correlation(x, Y_filtered)
        
        corrs = corrs[~np.isnan(corrs)]
        
        if len(corrs) == 0:
            mean_c = median_c = std_c = max_c = p90 = p10 = 0.0
        else:
            mean_c = float(np.mean(corrs))
            median_c = float(np.median(corrs))
            std_c = float(np.std(corrs))
            max_c = float(np.max(corrs))
            p90 = float(np.percentile(corrs, 90))
            p10 = float(np.percentile(corrs, 10))
        
        print(f"{config.capitalize()}: Mean r={mean_c:.4f}, Median={median_c:.4f}, Max={max_c:.4f}, 90th%={p90:.4f}")
        
        results[config] = {
            "mean_correlation": mean_c,
            "median_correlation": median_c,
            "std_correlation": std_c,
            "max_correlation": max_c,
            "p90_correlation": p90,
            "p10_correlation": p10
        }

    # 5. Save Info
    info = {
        "model": args.model,
        "benchmark": args.benchmark,
        "timestamp": datetime.datetime.now().isoformat(),
        "args": vars(args),
        "params": {
            "surprisal_type": "total_contextual",
            "roi": roi_info,
            "n_neuroids": Y_filtered.shape[1]
        },
        "results": results
    }
    
    save_file = os.path.join(args.save_path, "info.json")
    with open(save_file, 'w') as f:
        json.dump(info, f, indent=4)
        
    print(f"Done. Results saved to {save_file}")

if __name__ == "__main__":
    main()

