import pickle

# Path to your pickle file
score_path = "dumps/scores_model=gpt2_benchmark=Pereira2018.243sentences-linear_seed=42.pkl"

# Load the Score object
with open(score_path, 'rb') as f:
    layer_scores = pickle.load(f)

# Print the main score and its structure
print("--- Score Object ---")
print(layer_scores)
print("--- Score Object ---")
print(layer_scores)
print("--- Score Object ---")
print(layer_scores)

# Access the central, aggregated score
# .item() extracts the scalar value
print(f"\n--- Aggregated Score ---")
print(layer_scores.item())
print("--- Aggregated Score ---")
print(layer_scores.item())
print("--- Aggregated Score ---")
print(layer_scores.item())

# Access the scores for each layer (if available as a coordinate)
if 'layer' in layer_scores.coords:
    print("\n--- Scores per Layer ---")
    for layer in layer_scores['layer'].values:
        layer_score = layer_scores.sel(layer=layer)
        print(f"Layer: {layer}, Score: {layer_score.item():.4f}")

# Inspect the metadata and raw scores
print("\n--- Attributes (Metadata) ---")
print(layer_scores.attrs)

# Check for raw scores from cross-validation splits
if 'raw' in layer_scores.attrs:
    print("\n--- Raw scores from splits ---")
    print(layer_scores.attrs['raw'])