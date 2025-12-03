import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from brainscore import score, load_benchmark, ArtificialSubject
from brainscore.model_helpers.huggingface import HuggingfaceSubject, get_layer_names
from models.locality_gpt.model import LocalityGPT2


def main():
    parser = argparse.ArgumentParser(description="Simple Pipeline for BrainScore")
    parser.add_argument("--model", default="distilgpt2", help="Model identifier (e.g., distilgpt2, gpt2)")
    parser.add_argument("--untrained", action="store_true", help="Use an untrained version of the model")
    parser.add_argument("--localize", action="store_true", help="Perform localization before scoring")
    parser.add_argument("--num-units", type=int, default=256, help="Number of units to select during localization")
    parser.add_argument("--benchmark", default="Pereira2018.384sentences-linear", help="Benchmark identifier")
    parser.add_argument("--device", default="cuda", help="Device to use (cpu, cuda)")
    parser.add_argument("--save_path", default=None, help="Directory to save results (overrides --output_dir)")
    parser.add_argument("--decay-rate", type=float, default=1.0, help="Decay rate for LocalityGPT2")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for localization")
    parser.add_argument("--topic_wise_cv", action="store_true", help="Use topic-wise cross-validation (GroupKFold)")
    parser.add_argument("--no_topic_wise_cv", dest="topic_wise_cv", action="store_false", help="Use random cross-validation (KFold)")
    parser.set_defaults(topic_wise_cv=True)
    args = parser.parse_args()

    # Handle save_path logic
    if args.save_path:
        args.output_dir = args.save_path

    device = args.device
    # Allow numeric device IDs (e.g., 0) to refer to CUDA if available
    if device.isdigit():
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    seeds = [0, 1, 2, 3, 4]
    all_results = []
    
    for seed in seeds:
        print(f"\n=== Running Seed {seed} ===")
        # Set seed
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        print(f"Loading model: {args.model} (Untrained: {args.untrained})")
        
        if "locality_gpt" in args.model:
            # For LocalityGPT, we need to handle config manually if needed, but it seems it handles itself.
            # However, we need hidden_dim for localizer_kwargs if we want to be consistent, 
            # but LocalityGPT might not use the same localizer logic or might handle it internally.
            # Let's check LocalityGPT init. It takes localizer_kwargs.
            
            base_model_id = "gpt2"
            config = AutoConfig.from_pretrained(base_model_id)
            hidden_dim = getattr(config, "n_embd", getattr(config, "hidden_size", 768))
            
            localizer_kwargs = {
                'top_k': args.num_units,
                'batch_size': args.batch_size,
                'hidden_dim': hidden_dim
            }
            
            layer_names = get_layer_names(base_model_id)
            subject = LocalityGPT2(
                model_id=base_model_id,
                region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: layer_names},
                untrained=args.untrained,
                use_localizer=args.localize,
                localizer_kwargs=localizer_kwargs, 
                decay_rate=args.decay_rate
            )
            subject.model.to(device)
            subject.model.eval()
        else:
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
    
            layer_names = get_layer_names(args.model)
            
            hidden_dim = getattr(config, "n_embd", getattr(config, "hidden_size", 768))
            localizer_kwargs = {
                'top_k': args.num_units,
                'batch_size': args.batch_size,
                'hidden_dim': hidden_dim
            }
            
            subject = HuggingfaceSubject(
                model_id=args.model + ("-untrained" if args.untrained else ""),
                model=model,
                tokenizer=tokenizer,
                region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: layer_names},
                use_localizer=args.localize,
                localizer_kwargs=localizer_kwargs
            )
    
        # Score
        print(f"Loading benchmark: {args.benchmark}")
        benchmark = load_benchmark(args.benchmark)
        if hasattr(benchmark, 'topic_wise_cv'):
            print(f"Setting topic_wise_cv to {args.topic_wise_cv}")
            benchmark.topic_wise_cv = args.topic_wise_cv
    
        print("Scoring model...")
        results = score(subject, benchmark)
        print(f"Score (Seed {seed}): {results}")
        all_results.append(results)

    # Average results
    print("\n=== Averaging Results ===")
    avg_score = np.mean([float(r.values) if r.values.size == 1 else np.mean(r.values) for r in all_results])
    
    # Extract diagnostics from the first result (assuming structure is consistent)
    # But we need to average the scalar metrics inside diagnostics too
    
    def get_attr(res, key, default=None):
        return res.attrs.get(key, default)
        
    def get_diag(res, key, default=None):
        diag = res.attrs.get('diagnostics', {})
        if isinstance(diag, str):
            try:
                import ast
                diag = ast.literal_eval(diag)
            except:
                diag = {}
        return diag.get(key, default)

    # Collect metrics
    obj_corrs = [get_attr(r, 'objective_alignment_score') for r in all_results]
    llm_corrs = [get_attr(r, 'original_alignment_score') for r in all_results]
    
    obj_corrs_norm = [get_attr(r, 'objective_normalized_alignment_score') for r in all_results]
    llm_corrs_norm = [get_attr(r, 'original_normalized_alignment_score') for r in all_results]
    
    obj_vars = [get_diag(r, 'objective_explained_variance') for r in all_results]
    llm_vars = [get_diag(r, 'obj_llm_explained_variance') for r in all_results] # Combined
    
    # Helper to average ignoring Nones
    def safe_mean(lst):
        valid = [x for x in lst if x is not None]
        return float(np.mean(valid)) if valid else None

    avg_obj_corr = safe_mean(obj_corrs)
    avg_llm_corr = safe_mean(llm_corrs)
    avg_obj_corr_norm = safe_mean(obj_corrs_norm)
    avg_llm_corr_norm = safe_mean(llm_corrs_norm)
    avg_obj_var = safe_mean(obj_vars)
    avg_llm_var = safe_mean(llm_vars)
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. Create structured info.json
    import datetime
    import sys
            
    info = {
        "model": args.model,
        "benchmark": args.benchmark,
        "untrained": args.untrained,
        "localize": args.localize,
        "num_units": args.num_units,
        "device": args.device,
        "decay_rate": args.decay_rate if "locality_gpt" in args.model else None,
        "timestamp": datetime.datetime.now().isoformat(),
        "args": vars(args),
        "command": " ".join(sys.argv),
        "scores": {
            "correlation": {
                "objective": avg_obj_corr,
                "llm": avg_llm_corr
            },
            "correlation_normalized": {
                "objective": avg_obj_corr_norm,
                "llm": avg_llm_corr_norm
            },
            "explained_variance": {
                "objective": avg_obj_var,
                "llm": avg_llm_var,
                "partial": avg_score # Partial R2 is the main score
            }
        }
    }

    # Use the last result object for saving NetCDF (it contains the full structure)
    # Ideally we would average the NetCDF too but that's complex. 
    # Saving the last seed's result is a reasonable proxy for the structure, 
    # but the info.json contains the averaged scalars.
    results = all_results[-1]

    # 2. Fix Serialization Error: Clean attributes
    # The error "Invalid value for attr 'raw': <xarray.Score ...>" happens because 
    # complex objects are stored in attrs. We convert them to string or remove them.
    print("Cleaning results attributes for serialization...")
    for key, value in list(results.attrs.items()):
        # NetCDF only supports str, Number, ndarray, list, tuple. Dict is NOT supported.
        # We also want to be careful with lists/tuples containing non-primitives.
        if isinstance(value, dict):
            print(f"Converting dictionary attribute '{key}' to string for NetCDF compatibility.")
            results.attrs[key] = str(value)
        elif not isinstance(value, (str, int, float, list, tuple, type(None))):
            # Try to convert numpy arrays to list if possible, otherwise stringify
            try:
                import numpy as np
                if isinstance(value, np.ndarray):
                    continue # ndarray is supported
            except ImportError:
                pass
            
            print(f"Converting attribute '{key}' of type {type(value)} to string.")
            try:
                results.attrs[key] = str(value)
            except Exception as e:
                print(f"Failed to convert attribute '{key}': {e}. Removing it.")
                del results.attrs[key]

    # 3. Save results (NetCDF)
    save_path_nc = os.path.join(args.output_dir, "score.nc")
    results.to_netcdf(save_path_nc)
    print(f"Results (Seed 4) saved to: {save_path_nc}")

    # 4. Save info.json
    save_path_json = os.path.join(args.output_dir, "info.json")
    import json
    with open(save_path_json, 'w') as f:
        # Handle potential non-serializable types in info (like numpy types)
        def default(o):
            import numpy as np
            if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8,
                              np.uint16, np.uint32, np.uint64)):
                return int(o)
            elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
                return float(o)
            elif isinstance(o, (np.ndarray,)):
                return o.tolist()
            return str(o)
            
        json.dump(info, f, indent=4, default=default)
    print(f"Run info saved to: {save_path_json}")

    # 4. Save Benchmark Examples (Stimuli)
    print("Attempting to save benchmark examples...")
    try:
        # Try to access stimulus_set from the benchmark
        # This depends on the specific benchmark implementation in BrainScore
        stimulus_set = None
        if hasattr(benchmark, 'stimulus_set'):
            stimulus_set = benchmark.stimulus_set
        elif hasattr(benchmark, '_stimulus_set'):
            stimulus_set = benchmark._stimulus_set
        
        if stimulus_set is not None:
            examples_path = os.path.join(args.output_dir, "benchmark_examples.csv")
            # stimulus_set is usually a pandas DataFrame-like object
            if hasattr(stimulus_set, 'to_csv'):
                stimulus_set.to_csv(examples_path)
                print(f"Benchmark examples saved to: {examples_path}")
            else:
                print(f"Stimulus set found but does not support to_csv: {type(stimulus_set)}")
        else:
            print("No stimulus_set found in benchmark object.")
    except Exception as e:
        print(f"Failed to save benchmark examples: {e}")

    # 5. Plot Score Distribution (if applicable)
    if results.size > 1:
        print("Generating score distribution plot...")
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 6))
            # Flatten results for histogram
            values = results.values.flatten()
            sns.histplot(values, kde=True)
            plt.title(f"Score Distribution - {args.benchmark}")
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.axvline(x=np.mean(values), color='r', linestyle='--', label=f'Mean: {np.mean(values):.4f}')
            plt.legend()
            
            plot_path = os.path.join(args.output_dir, "score_distribution.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Score distribution plot saved to: {plot_path}")
        except Exception as e:
            print(f"Failed to generate score distribution plot: {e}")

    # 6. Attention Plotting Removed
    # Attention plots are no longer generated automatically.
    # Use the scripts in the 'alignment' folder to generate plots and statistics.
    print("Skipping attention plots (use 'alignment' scripts).")

if __name__ == "__main__":
    main()
