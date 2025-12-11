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

    # 1. Load Model (Only for Surprisal calculation)
    # Load Model and Tokenizer
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
    stimuli = assembly['stimulus'] 
    
    sentences = stimuli['sentence'].values
    passage_labels = assembly['passage_label'].values
    
    print("Extracting unique sentences and computing surprisal...")
    
    unique_passages = sorted(set(passage_labels))
    
    ordered_surprisals = []
    ordered_topics = []
    ordered_ids = []
    
    for passage in unique_passages:
        # Get unique stimuli IDs for this passage
        passage_indices = [i for i, p in enumerate(passage_labels) if p == passage]
        passage_stim_ids = assembly['stimulus_id'].values[passage_indices]
        unique_p_stim_ids = sorted(list(set(passage_stim_ids))) 
        
        # Maintain context for this passage
        passage_context = []
        
        for sid in unique_p_stim_ids:
            # Find index in assembly
            idx = np.where(assembly['stimulus_id'].values == sid)[0][0]
            sentence = sentences[idx]
            topic = passage_labels[idx]
            
            ordered_ids.append(sid)
            ordered_topics.append(topic)
            
            # Compute Total Surprisal with Context
            s_val = compute_contextual_surprisal(model, tokenizer, sentence, passage_context, device)
            ordered_surprisals.append(s_val)
            
            # Add current sentence to context for next one
            passage_context.append(sentence)
    
    
    ordered_surprisals = np.array(ordered_surprisals)
    ordered_topics = np.array(ordered_topics)
    
    print("Computing Baselines...")
    # 1. Raw (Total)
    raw_surprisals = ordered_surprisals
    
    # 2. Static Relative
    static_relative = compute_static_baseline(ordered_surprisals, ordered_topics)
    
    # 3. Moving Relative
    moving_relative = compute_moving_baseline(ordered_surprisals, ordered_topics)
    
    # Store in a handy dict
    surprisal_map = {}
    for i, sid in enumerate(ordered_ids):
        surprisal_map[sid] = {
            'raw': raw_surprisals[i],
            'static': static_relative[i],
            'moving': moving_relative[i]
        }

    # 4. Correlation Analysis
    # We map surprisal to the brain data order
    assembly_stim_ids = assembly['stimulus_id'].values
    n_samples = len(assembly_stim_ids)
    
    aligned_surprisal = {
        'raw': np.zeros(n_samples),
        'static': np.zeros(n_samples),
        'moving': np.zeros(n_samples)
    }
    
    for i, sid in enumerate(assembly_stim_ids):
        for key in aligned_surprisal:
            aligned_surprisal[key][i] = surprisal_map[sid][key]
            
    # Y data
    Y = assembly.values # (N, V)
    
    # FILTER FOR LANGUAGE NETWORKS
    # Check if we have ROI info
    Y_filtered = Y
    roi_info = None
    
    if 'atlas' in assembly.coords:
        print("Found 'atlas' coordinate. Filtering for Language Network...")
        # Pereira atlas usually has 'language', 'auditory', etc.
        # Let's inspect unique values if possible, for now assume 'language' substring
        atlas_vals = assembly['atlas'].values
        # Simple boolean mask
        mask = np.array(['lang' in str(v).lower() for v in atlas_vals])
        if mask.sum() > 0:
            Y_filtered = Y[:, mask]
            print(f"Filtered from {Y.shape[1]} to {Y_filtered.shape[1]} neuroids.")
            roi_info = "Language ROI"
        else:
            print("No 'language' ROIs found in atlas. Using all neuroids.")
    else:
        print("No atlas/ROI coordinate found. Using all neuroids.")
        roi_info = "Whole Brain"

    
    # Helper for vectorized correlation
    def compute_correlation(x, Y):
        # Center x
        x_c = x - np.mean(x)
        x_norm = np.linalg.norm(x_c)
        if x_norm == 0:
            return np.zeros(Y.shape[1])
            
        # Center Y
        Y_c = Y - np.mean(Y, axis=0) # (N, V)
        Y_norm = np.linalg.norm(Y_c, axis=0) # (V,)
        
        Y_norm[Y_norm == 0] = 1e-9
        
        dot_prod = np.dot(x_c, Y_c) # (V,)
        corr = dot_prod / (x_norm * Y_norm)
        return corr

    # Results container
    results = {}
    
    print("\n=== Calculating Correlations (Contextual + Total + ROI Filter) ===")
    
    configs = ['raw', 'static', 'moving']
    
    for config in configs:
        x = aligned_surprisal[config]
        corrs = compute_correlation(x, Y_filtered)
        
        # Drop NaNs if any
        corrs = corrs[~np.isnan(corrs)]
        
        if len(corrs) == 0:
            mean_corr, median_corr, max_corr, p90 = 0,0,0,0
        else:
            mean_corr = float(np.mean(corrs))
            median_corr = float(np.median(corrs))
            std_corr = float(np.std(corrs))
            max_corr = float(np.max(corrs))
            p90 = float(np.percentile(corrs, 90))
            p10 = float(np.percentile(corrs, 10))
        
        print(f"{config.capitalize()}: Mean r={mean_corr:.4f}, Median={median_corr:.4f}, Max={max_corr:.4f}, 90th%={p90:.4f}")
        
        results[config] = {
            "mean_correlation": mean_corr,
            "median_correlation": median_corr,
            "std_correlation": std_corr,
            "max_correlation": max_corr,
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

