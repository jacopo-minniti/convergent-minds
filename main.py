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
    parser.add_argument("--num-units", type=int, default=1000, help="Number of units to select during localization")
    parser.add_argument("--benchmark", default="Pereira2018.384sentences-linear", help="Benchmark identifier")
    parser.add_argument("--device", default="cuda", help="Device to use (cpu, cuda)")
    parser.add_argument("--output_dir", default=".", help="Directory to save results")
    args = parser.parse_args()

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

    localizer_kwargs = {'num_units': args.num_units}

    print(f"Loading model: {args.model} (Untrained: {args.untrained})")
    
    if "locality_gpt" in args.model:
        base_model_id = "gpt2"
        layer_names = get_layer_names(base_model_id)
        subject = LocalityGPT2(
            model_id=base_model_id,
            region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: layer_names},
            untrained=args.untrained,
            use_localizer=args.localize,
            localizer_kwargs=localizer_kwargs, 
            decay_rate=-0.7
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
        
        model.to(device)
        model.eval()

        layer_names = get_layer_names(args.model)
        
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

    print("Scoring model...")
    results = score(subject, benchmark)
    print(f"Score: {results}")
    
    # Save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    save_path = os.path.join(args.output_dir, f"score_{args.model}_{args.benchmark}.nc")
    results.to_netcdf(save_path)
    print(f"Results saved to: {save_path}")

if __name__ == "__main__":
    main()
