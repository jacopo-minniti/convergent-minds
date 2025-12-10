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


def compute_raw_surprisal(model, tokenizer, sentences, device='cuda'):
    """
    Computes sentence surprisal: s(x) = (1/|x|) * sum(-log p(y_t | y_<t))
    """
    model.eval()
    surprisals = []
    
    # We might need to batch this if there are many sentences, but for Pereira (~243 or 384) it's fine.
    for sentence in tqdm(sentences, desc="Computing Raw Surprisal"):
        inputs = tokenizer(sentence, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            # HuggingFace loss is average NLL per token. 
            # We want sum NLL / length. Which is basically what HF loss returns (cross entropy is mean).
            # But let's be precise.
            # Loss = - (1/N) sum log P(token).
            # So loss.item() is exactly the surprisal according to the formula s(x) = 1/|x| * ...
            # Wait, HF loss usually ignores padding. Here batch size is 1, so no padding.
            loss = outputs.loss.item()
            surprisals.append(loss)
            
    return np.array(surprisals)

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
    # Assuming surprisals and topics are ordered by presentation
    
    # We need to track history per topic
    topic_history = {t: [] for t in np.unique(topics)}
    
    # For the first occurrence, we need a prior. The user said:
    # "either use the global mean over all sentences, or use the static topic mean"
    # Let's use static topic mean as prior to be robust.
    # So we need to calculate static means first or compute them on the fly?
    # "static topic mean (b_static(c_i)) as a prior."
    
    # Let's precompute static means for priors
    # This might leak future info if strict causal, but 'static topic mean' implies accessible long run avg?
    # Re-reading: "use the static topic mean (b_static(c_i)) as a prior."
    # Yes, usually static baseline is computed over the whole set.
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
    # localize args removed as not needed for pure surprisal, but kept for script compat if needed?
    # User script passed --localize. I'll keep the parser args to avoid breaking the calling script but ignore them.
    parser.add_argument("--localize", action="store_true")
    parser.add_argument("--num-units", type=int, default=256)
    
    parser.add_argument("--benchmark", default="Pereira2018.243sentences-partialr2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    # topic_wise_cv might not be needed for simple correlation, but keeping arg valid
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
    # Resolve possible paths: absolute, relative, or under 'models/' directory
    if os.path.isabs(args.model) and os.path.isdir(args.model):
        model_path = args.model
    elif os.path.isdir(args.model):
        model_path = args.model
    else:
        # Check if the model exists under the 'models' subdirectory
        possible_path = os.path.join(os.getcwd(), "models", args.model)
        if os.path.isdir(possible_path):
            model_path = possible_path
        else:
            # Assume it's a HuggingFace model identifier
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
        # config = model.config # variable not used but safe to have
        
    model.to(device)
    model.eval()

    # 2. Load Benchmark & Data
    print(f"Loading benchmark: {args.benchmark}")
    benchmark = load_benchmark(args.benchmark)
    
    if not hasattr(benchmark, 'data'):
        raise ValueError("Benchmark must have 'data' attribute loaded.")
    
    assembly = benchmark.data
    stimuli = assembly['stimulus'] # Xarray or similar
    
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
        
        for sid in unique_p_stim_ids:
            # Find index in assembly (Just grabbing the first occurrence)
            idx = np.where(assembly['stimulus_id'].values == sid)[0][0]
            sentence = sentences[idx]
            topic = passage_labels[idx]
            
            ordered_ids.append(sid)
            ordered_topics.append(topic)
            
            # Compute/Get Surprisal
            inputs = tokenizer(sentence, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                s_val = outputs.loss.item() 
            ordered_surprisals.append(s_val)
    
    # Clean up model to free memory? Not strictly necessary if 8G mem.
    
    ordered_surprisals = np.array(ordered_surprisals)
    ordered_topics = np.array(ordered_topics)
    
    print("Computing Baselines...")
    # 1. Raw
    raw_surprisals = ordered_surprisals
    
    # 2. Static Relative
    static_relative = compute_static_baseline(ordered_surprisals, ordered_topics)
    
    # 3. Moving Relative
    moving_relative = compute_moving_baseline(ordered_surprisals, ordered_topics)
    
    # Store in a handy dict keyed by stimulus_id
    surprisal_map = {}
    for i, sid in enumerate(ordered_ids):
        surprisal_map[sid] = {
            'raw': raw_surprisals[i],
            'static': static_relative[i],
            'moving': moving_relative[i]
        }

    # 4. Correlation Analysis
    # We want to correlate Surprisal Vector vs Brain Data (Neuroids)
    
    # Prepare Brain Data Matrix (Y) aligned to ordered_ids?
    # No, assembly has its own order. We should map Surprisal to Assembly.
    # assembly (Y) shape: (Presentations, Neuroids)
    
    # Create aligned surprisal vectors
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
    
    # Helper for vectorized correlation
    def compute_correlation(x, Y):
        """
        x: (N,)
        Y: (N, V)
        Returns: (V,) correlation coefficients
        """
        # Center x
        x_c = x - np.mean(x)
        x_norm = np.linalg.norm(x_c)
        if x_norm == 0:
            return np.zeros(Y.shape[1])
            
        # Center Y
        Y_c = Y - np.mean(Y, axis=0) # (N, V)
        Y_norm = np.linalg.norm(Y_c, axis=0) # (V,)
        
        # Avoid division by zero
        Y_norm[Y_norm == 0] = 1e-9
        
        # Correlation
        # cov = dot(x_c, Y_c)
        dot_prod = np.dot(x_c, Y_c) # (V,)
        corr = dot_prod / (x_norm * Y_norm)
        return corr

    # Results container
    results = {}
    
    print("\n=== Calculating Correlations ===")
    
    # Iterate configs
    configs = ['raw', 'static', 'moving']
    
    for config in configs:
        x = aligned_surprisal[config]
        corrs = compute_correlation(x, Y)
        
        mean_corr = float(np.mean(corrs))
        median_corr = float(np.median(corrs))
        std_corr = float(np.std(corrs))
        
        print(f"{config.capitalize()}: Mean r = {mean_corr:.4f}, Median r = {median_corr:.4f}")
        
        results[config] = {
            "mean_correlation": mean_corr,
            "median_correlation": median_corr,
            "std_correlation": std_corr,
            "all_correlations": corrs.tolist() 
        }

    # 5. Save Info
    info = {
        "model": args.model,
        "benchmark": args.benchmark,
        "timestamp": datetime.datetime.now().isoformat(),
        "args": vars(args),
        "results": results
    }
    
    save_file = os.path.join(args.save_path, "info.json")
    with open(save_file, 'w') as f:
        json.dump(info, f, indent=4)
        
    print(f"Done. Results saved to {save_file}")

if __name__ == "__main__":
    main()

