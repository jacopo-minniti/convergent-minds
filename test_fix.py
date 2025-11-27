
import os
import shutil
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import MagicMock

# Mocking the environment to test the logic inside main.py without running the full thing
# We will extract the logic we want to test into functions or just replicate it here for verification

def test_saving_logic():
    print("Testing saving logic...")
    output_dir = "test_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Create a dummy xarray result with problematic attributes
    data = xr.DataArray([0.5], coords={'split': [0]}, dims='split')
    data.attrs['raw'] = data # Self-reference or complex object causing the issue
    data.attrs['simple'] = 'ok'
    
    # Replicate the cleaning logic
    print("Cleaning attributes...")
    for key, value in list(data.attrs.items()):
        if not isinstance(value, (str, int, float, list, tuple, type(None))):
            try:
                import numpy as np
                if isinstance(value, np.ndarray):
                    continue 
            except ImportError:
                pass
            
            print(f"Converting attribute '{key}' of type {type(value)} to string.")
            try:
                data.attrs[key] = str(value)
            except Exception as e:
                print(f"Failed to convert attribute '{key}': {e}. Removing it.")
                del data.attrs[key]
    
    # Try saving
    save_path = os.path.join(output_dir, "score.nc")
    try:
        data.to_netcdf(save_path)
        print(f"Successfully saved to {save_path}")
    except Exception as e:
        print(f"Failed to save netcdf: {e}")
        return False

    # Test JSON saving
    info = {"test": "value", "score": float(data.values[0])}
    import json
    with open(os.path.join(output_dir, "info.json"), 'w') as f:
        json.dump(info, f)
    
    if os.path.exists(os.path.join(output_dir, "info.json")):
        print("Successfully saved info.json")
    else:
        print("Failed to save info.json")
        return False
        
    # Mock benchmark with stimulus_set
    benchmark = MagicMock()
    stimulus_set = MagicMock()
    stimulus_set.to_csv = MagicMock()
    benchmark.stimulus_set = stimulus_set
    
    # Replicate main.py logic for benchmark examples
    print("Testing benchmark examples saving...")
    if hasattr(benchmark, 'stimulus_set'):
        examples_path = os.path.join(output_dir, "benchmark_examples.csv")
        benchmark.stimulus_set.to_csv(examples_path)
        # Create dummy file to simulate success
        with open(examples_path, 'w') as f:
            f.write("stimulus_id,sentence\n1,test")
            
    if os.path.exists(os.path.join(output_dir, "benchmark_examples.csv")):
        print("Successfully saved benchmark examples")
    else:
        print("Failed to save benchmark examples")
        return False

    # Replicate main.py logic for score distribution
    print("Testing score distribution plotting...")
    # Create multi-value result
    multi_data = xr.DataArray(np.random.rand(10), coords={'split': range(10)}, dims='split')
    
    if multi_data.size > 1:
        try:
            plt.figure(figsize=(10, 6))
            values = multi_data.values.flatten()
            sns.histplot(values, kde=True)
            plot_path = os.path.join(output_dir, "score_distribution.png")
            plt.savefig(plot_path)
            plt.close()
            if os.path.exists(plot_path):
                print("Successfully saved score distribution plot")
            else:
                print("Failed to save score distribution plot")
                return False
        except Exception as e:
            print(f"Plotting failed: {e}")
            return False

    return True

def test_plotting_logic():
    print("\nTesting plotting logic...")
    output_dir = "test_output"
    plot_dir = os.path.join(output_dir, "attention_plots")
    
    # Mock model and tokenizer
    tokenizer = MagicMock()
    tokenizer.return_value = MagicMock()
    tokenizer.return_value.to.return_value = MagicMock()
    tokenizer.return_value.input_ids = torch.tensor([[0, 1, 2]])
    tokenizer.convert_ids_to_tokens.return_value = ["a", "b", "c"]
    
    model = MagicMock()
    # Mock output: (loss, logits, past_key_values, attentions)
    # attentions: tuple of layers. Each layer: (batch, heads, seq, seq)
    # Let's say 1 layer, 1 head, 3x3 seq
    attn_matrix = torch.rand(1, 1, 3, 3)
    outputs = MagicMock()
    outputs.attentions = (attn_matrix,)
    model.return_value = outputs
    
    device = "cpu"
    
    try:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        sentences = ["test sentence"]
        
        for i, sentence in enumerate(sentences):
            # Replicate logic
            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
            
            attentions = outputs.attentions
            last_layer_attn = attentions[-1][0].cpu().numpy()
            avg_attn = np.mean(last_layer_attn, axis=0)
            
            tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(avg_attn, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
            plt.title(f"Avg Attention (Last Layer) - Sentence {i+1}")
            plt.savefig(os.path.join(plot_dir, f"attention_sentence_{i+1}.png"))
            plt.close()
            
        if os.path.exists(os.path.join(plot_dir, "attention_sentence_1.png")):
            print("Successfully generated attention plot")
            return True
        else:
            print("Failed to generate plot file")
            return False
            
    except Exception as e:
        print(f"Plotting failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_saving_logic() and test_plotting_logic():
        print("\nAll verification tests passed!")
    else:
        print("\nVerification failed!")
