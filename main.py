import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from brainscore import score, load_benchmark, ArtificialSubject
from brainscore.model_helpers.huggingface import HuggingfaceSubject, get_layer_names

def main():
    parser = argparse.ArgumentParser(description="Simple Pipeline for BrainScore")
    parser.add_argument("--model", default="distilgpt2", help="Model identifier (e.g., distilgpt2, gpt2)")
    parser.add_argument("--untrained", action="store_true", help="Use an untrained version of the model")
    parser.add_argument("--localize", action="store_true", help="Perform localization before scoring")
    parser.add_argument("--num-units", type=int, default=1000, help="Number of units to select during localization")
    parser.add_argument("--benchmark", default="Pereira2018.384sentences-cka", help="Benchmark identifier")
    parser.add_argument("--device", default="cuda", help="Device to use (cpu, cuda)")
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"Loading model: {args.model} (Untrained: {args.untrained})")
    
    # Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.untrained:
        config = AutoConfig.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
    
    model.to(device)
    model.eval()

    layer_names = get_layer_names(args.model)
    
    localizer_kwargs = None
    if args.localize:
        localizer_kwargs = {
            "top_k": args.num_units,
            "batch_size": 8,
            "hidden_dim": model.config.hidden_size
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

    print("Scoring model...")
    results = score(subject, benchmark)
    print(f"Score: {results}")

if __name__ == "__main__":
    main()
