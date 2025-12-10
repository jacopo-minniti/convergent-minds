import os
import sys
# Force local import of brainscore by adding the current directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import numpy as np
import pandas as pd
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
        config = model.config
        
    model.to(device)
    model.eval()

    # 2. Load Benchmark & Data
    print(f"Loading benchmark: {args.benchmark}")
    benchmark = load_benchmark(args.benchmark)
    # Ensure usage of topic_wise_cv
    benchmark.topic_wise_cv = args.topic_wise_cv

    # We need the stimulus set to get sentences and topics
    # Benchmark.data['stimulus'] usually holds this
    if not hasattr(benchmark, 'data'):
        raise ValueError("Benchmark must have 'data' attribute loaded.")
    
    assembly = benchmark.data
    stimuli = assembly['stimulus'] # Xarray or similar
    
    # Depending on brainscore version, getting the actual text might differ
    # Usually stimuli['sentence'] or stimuli['word'] etc.
    # For Pereira, it's 'sentence'.
    sentences = stimuli['sentence'].values

    passage_labels = assembly['passage_label'].values
    
    # IMPORTANT: Ensure sentences are sorted by presentation order if we do moving baseline
    # But calculate_raw_surprisal should map per sentence.
    # We should iterate in the order they appear in assembly's stimulus dimension?
    # Pereira assembly usually has neuroid x presentation.
    # 'presentation' maps to 'stimulus_id'.
    # We need unique sentences.
    
    print("Extracting unique sentences and computing surprisal...")
    # Get unique stimuli in presentation order (if defined) or just unique set
    # The benchmark methods usually sort by passage.
    
    unique_passages = sorted(set(passage_labels))
    
    # We need to construct a mapping: stimulus_id -> info
    # Because actual scoring aligns by stimulus_id
    stimulus_id_to_info = {}
    
    # We need to process sentences in "presentation order" for moving baseline.
    # Pereira experiments were presented in blocks. 
    # If we iterate by passage (sorted) and then by sentence within passage, is that the order?
    # The prompt says: "You need the presentation order of sentences."
    # Does the dataset have 'presentation_order'?
    # It might NOT be strictly defined in the static assembly if it aggregates multiple subjects.
    # However, for the purpose of this "causal experiment", we can assume the order in the dataset 
    # (or sorted by passage/sentence ID) is the canonical order or "simulated" order.
    # Let's iterate passages sorted (as in benchmark.py) and sentences within them.
    
    ordered_surprisals = []
    ordered_topics = []
    ordered_ids = []
    
    # To avoid re-computing for duplicates if any
    
    for passage in unique_passages:
        # Get stimuli for this passage
        # In benchmark.py: passage_indexer = [s == passage for s in passage_labels]
        # But we need them ordered. 
        # Assuming the benchmark logic:
        passage_indices = [i for i, p in enumerate(passage_labels) if p == passage]
        # This gets ALL presentations. Pereira has repeated presentations?
        # Usually stimulus set is unique sentences.
        # Let's check stimuli dataframe directly.
        
        # Actually benchmark.data['stimulus'] is likely the stimulus set details repeated for presentation?
        # No, self.data has dim 'presentation'. 
        
        # Let's just get the unique stimulus IDs for this passage
        passage_stim_ids = assembly['stimulus_id'].values[passage_indices]
        # Unique them
        unique_p_stim_ids = sorted(list(set(passage_stim_ids))) # Sort to be deterministic
        
        for sid in unique_p_stim_ids:
            # Find the sentence string
            # Find index in assembly
            idx = np.where(assembly['stimulus_id'].values == sid)[0][0]
            sentence = sentences[idx]
            topic = passage_labels[idx]
            
            ordered_ids.append(sid)
            ordered_topics.append(topic)
            
            # Compute/Get Surprisal
            # We compute it right here (or cache it)
            # Batching would be faster but let's do one by one for clarity and safety
            inputs = tokenizer(sentence, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                s_val = outputs.loss.item() 
            ordered_surprisals.append(s_val)

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
    
    # 3. Run Scoring Loop for Ablations
    # We need to replicate the scoring call but with modified X_llm
    
    # First, get standard embeddings (activations)
    # We can use the Subject to get them, or reuse if possible.
    # The benchmark calls candidate.digest_text(). 
    # We should instantiate the subject properly.
    
    hidden_dim = getattr(config, "n_embd", getattr(config, "hidden_size", 768))
    localizer_kwargs = {
        'top_k': args.num_units,
        'batch_size': args.batch_size,
        'hidden_dim': hidden_dim
    }
    
    layer_names = get_layer_names(args.model)
    subject = HuggingfaceSubject(
        model_id=args.model + ("-untrained" if args.untrained else ""),
        model=model,
        tokenizer=tokenizer,
        region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: layer_names},
        use_localizer=args.localize,
        localizer_kwargs=localizer_kwargs
    )
    
    # We need to run the subject on the benchmark to get 'neural' (activations)
    # But we want to inject surprisal into the features before the final regression.
    # Benchmark.__call__ does everything opaque.
    # We need to breakdown Benchmark.__call__ logic or rely on a modified flow.
    # Since we can't modify benchmark.py, we have to copy the logic here using the `linear_partial_r2` metric directly.
    
    from alignment.metrics.linear_partial_r2 import linear_partial_r2
    
    # Generate Activations (X_llm_base)
    print("Generating Model Activations...")
    # We can use the subject.digest_text for each passage like benchmark does
    
    predictions_list = []
    
    # Re-iterate passages (sorted) to match benchmark consistency
    for passage in unique_passages:
        passage_indices = [p == passage for p in passage_labels]
        # We need the stimuli xarray for this passage
        # Currently 'stimuli' is the whole column. We need the subset.
        # The benchmark aligns by 'presentation' dim in assembly.
        # But 'digest_text' expects a list of strings or xarray with specific structure?
        # digest_text input: Union[List[str], xr.DataArray]
        
        # Let's get the standard stimulus set subset
        # In benchmark: passage_stimuli = stimuli[passage_indexer]
        # passage_indexer matches 'presentation' dimension of assembly
        # stimuli is assembly['stimulus']
        
        passage_indexer = [p == passage for p in passage_labels]
        # passage_stimuli = stimuli[passage_indexer] # slicing xarray
        # Use boolean indexing on valid dimensions is tricky in some xarray versions if dims don't match
        # But here 'stimuli' is coordinate of assembly?
        # assembly.sel(presentation=...)
        pass_data = assembly.isel(presentation=passage_indexer)
        pass_stimuli = pass_data['stimulus']
        
        output = subject.digest_text(pass_stimuli.values)
        passage_predictions = output['neural']
        passage_predictions['stimulus_id'] = 'presentation', pass_stimuli['stimulus_id'].values
        predictions_list.append(passage_predictions)

    predictions = xr.concat(predictions_list, dim='presentation')
    
    # Prepare X_obj
    # Load objective features same as benchmark
    filename = f"pereira2018_{benchmark.experiment.replace('sentences', '')}_obj.npz"
    filepath = os.path.join("data", filename)
    data_obj = np.load(filepath, allow_pickle=True)
    X_obj = data_obj['X_obj']
    obj_stimulus_ids = data_obj['stimulus_ids']
    
    pred_stimulus_ids = predictions['stimulus_id'].values
    obj_id_to_idx = {sid: i for i, sid in enumerate(obj_stimulus_ids)}
    indices = [obj_id_to_idx[sid] for sid in pred_stimulus_ids]
    X_obj_aligned = X_obj[indices]
    
    # Prepare Base X_llm
    X_llm_base = predictions.values
    
    # Prepare Y
    data_id_to_idx = {sid: i for i, sid in enumerate(assembly['stimulus_id'].values)}
    y_indices = [data_id_to_idx[sid] for sid in pred_stimulus_ids]
    y_aligned = assembly.values[y_indices]
    passage_labels_aligned = assembly['passage_label'].values[y_indices]
    
    # Define Splits (Topic-Wise)
    if args.topic_wise_cv:
        gkf = GroupKFold(n_splits=10)
        splits = list(gkf.split(X_llm_base, y_aligned, groups=passage_labels_aligned))
    else:
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        splits = list(kf.split(X_llm_base, y_aligned))

    # 4. Ablation Loop
    ablations = [
        ("No Surprisal", None),
        ("Raw Surprisal", "raw"),
        ("Static Relative", "static"),
        ("Moving Relative", "moving")
    ]
    
    ablation_results = {}
    
    print("\n=== Running Ablations ===")
    
    for name, mode in ablations:
        print(f"Scoring: {name}")
        
        if mode is None:
            X_llm_curr = X_llm_base
        else:
            # Construct surprisal column
            # Must match pred_stimulus_ids order
            surp_col = []
            for sid in pred_stimulus_ids:
                surp_col.append(surprisal_map[sid][mode])
            surp_col = np.array(surp_col)[:, np.newaxis] # (N, 1)
            
            X_llm_curr = np.hstack([X_llm_base, surp_col])
            
        # Run Metric
        score_val, diagnostics = linear_partial_r2(
            X_obj=X_obj_aligned,
            X_llm=X_llm_curr,
            y=y_aligned,
            splits=splits
        )
        
        # Collect relevant stats
        # We focus on the Partial R2
        # Normalize it if ceiling is available
        
        # Ceiling
        # We need the ceiling from the benchmark object
        # benchmark.ceiling
        
        normalized_partial_r2 = None
        if benchmark.ceiling is not None:
            ceiling_values = benchmark.ceiling.values
            median_ceiling_r2 = np.median(ceiling_values ** 2)
            if median_ceiling_r2 > 0:
                normalized_partial_r2 = float(score_val / median_ceiling_r2)
            else:
                normalized_partial_r2 = 0.0
                
        ablation_results[name] = {
            "partial_r2": float(score_val),
            "normalized_partial_r2": normalized_partial_r2,
            "llm_explained_variance": diagnostics.get('llm_explained_variance'), # Raw LLM exp var
            "joint_explained_variance": diagnostics.get('obj_llm_explained_variance')
        }
        
        print(f"  Result: Norm Partial R2 = {normalized_partial_r2:.4f}")

    # 5. Save Info
    info = {
        "model": args.model,
        "benchmark": args.benchmark,
        "timestamp": datetime.datetime.now().isoformat(),
        "args": vars(args),
        "results": ablation_results
    }
    
    save_file = os.path.join(args.save_path, "info.json")
    with open(save_file, 'w') as f:
        json.dump(info, f, indent=4)
        
    print(f"Done. Results saved to {save_file}")

if __name__ == "__main__":
    main()

