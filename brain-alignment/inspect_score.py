import pickle
import warnings
warnings.filterwarnings("ignore")

def inspect_score(name, score_path):
    """
    Inspect a score file and print raw and normalized scores side by side.
    
    Args:
        name (str): Name to display for this score
        score_path (str): Path to the pickle file containing the score
    """
    # Load the Score object
    with open(score_path, 'rb') as f:
        layer_scores = pickle.load(f)
    
    # Extract raw and normalized scores
    normalized_score = layer_scores.item()
    raw_score = layer_scores.attrs['raw'].item() if 'raw' in layer_scores.attrs else None
    
    # Print scores side by side
    print(f"{name}:")
    if raw_score is not None:
        print(f"  Raw Score: {raw_score:.6f} | Normalized Score: {normalized_score:.6f}")
    else:
        print(f"  Normalized Score: {normalized_score:.6f} (Raw score not available)")
    
    # Print per-layer scores if available
    if 'layer' in layer_scores.coords:
        print(f"  Layer-wise scores:")
        for layer in layer_scores['layer'].values:
            layer_score = layer_scores.sel(layer=layer)
            print(f"    Layer {layer}: {layer_score.item():.6f}")

def main():
    # Dictionary of score names and their corresponding file paths
    score_files = {
        "Untrained-GPT2": "dumps/scores_untrained_gpt2_Pereira2018.243_linear.pkl",
        "Untrained-LocalityGPT2 (1.0)": "dumps/scores_untrained_locality_1.0_gpt2_Pereira2018.243_linear.pkl",
        "Untrained-LocalityGPT2 (-0.3)": "dumps/scores_untrained_locality_-0.3_gpt2_Pereira2018.243_linear.pkl"
    }
    
    print("=== Score Inspection Results ===")
    for name, path in score_files.items():
        try:
            inspect_score(name, path)
            print()  # Add blank line between scores
        except FileNotFoundError:
            print(f"{name}: File not found at {path}")
            print()
        except Exception as e:
            print(f"{name}: Error loading score - {e}")
            print()

if __name__ == "__main__":
    main()