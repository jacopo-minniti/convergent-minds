import os
import torch
import random 
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import setup_hooks, get_layer_names, write_pickle

# Set tokenizer parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "False"

def seed_everything(seed: int):    
    """Set seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Fed10_LocLangDataset(Dataset):
    """Custom dataset for the Fedorenko2010 language localization experiment."""
    def __init__(self, dirpath):
        # Load and concatenate all CSV files from the directory
        paths = glob(f"{dirpath}/*.csv")
        data = pd.read_csv(paths[0])
        for path in paths[1:]:
            run_data = pd.read_csv(path)
            data = pd.concat([data, run_data])

        # Combine stimuli columns into a single sentence
        data["sent"] = data["stim2"].apply(str.lower)
        for stimuli_idx in range(3, 14):
            data["sent"] += " " + data[f"stim{stimuli_idx}"].apply(str.lower)

        # Separate sentences and non-words
        self.sentences = data[data["stim14"]=="S"]["sent"]
        self.non_words = data[data["stim14"]=="N"]["sent"]

    def __getitem__(self, idx):
        """Return a single sample from the dataset."""
        return self.sentences.iloc[idx].strip(), self.non_words.iloc[idx].strip()
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.sentences)

def extract_representations(model, 
    input_ids, 
    attention_mask,
    layer_names,
    embed_agg,
):
    """Extract hidden representations from the model."""
    
    # Set up hooks to capture layer activations
    batch_activations = {layer_name: [] for layer_name in layer_names}
    hooks, layer_representations = setup_hooks(model, layer_names)

    # Run the model in inference mode
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Process activations for each sample in the batch
    for sample_idx in range(len(input_ids)):
        seq_len = attention_mask[sample_idx].sum()
        for layer_name in layer_names:
            # Aggregate activations based on the chosen method
            if embed_agg == "mean":
                activations = layer_representations[layer_name][sample_idx][-seq_len:].mean(dim=0).cpu()
            elif embed_agg == "last-token":
                activations = layer_representations[layer_name][sample_idx][-1].cpu()
            else:
                raise ValueError(f"{embed_agg} not implemented")
            
            batch_activations[layer_name] += [activations]

    # Remove hooks after use
    for hook in hooks:
        hook.remove()

    return batch_activations

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('--model-name',  type=str, default="gpt2", help='HuggingFace model name')
    parser.add_argument('--dataset-name',  type=str, default="fedorenko10", help='Dataset name')
    parser.add_argument('--seed',  type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--batch-size',  type=int, default=32, help='Batch size for processing')
    parser.add_argument('--embed-agg',  type=str, default="last-token", help='Aggregation method for embeddings')
    parser.add_argument('--cuda',  type=int, default=0, help='CUDA device index')
    parser.add_argument('--overwrite',  action='store_true', help='Overwrite existing files') 
    args = parser.parse_args()

    # --- Setup ---
    seed_everything(seed=args.seed)
    num_samples = 240 # Number of samples in the Fedorenko2010 dataset

    # Define save path and check for existing files
    savepath = f"language-localization/reps_model={args.model_name}_dataset={args.dataset_name}_pretrained=True_agg={args.embed_agg}_seed={args.seed}.pkl"
    if os.path.exists(savepath) and not args.overwrite:
        print(f"> Already Exists: {savepath}")
        exit()

    # --- Model and Tokenizer Loading ---
    print(f"> Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="cpu", torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # --- Dataset and Dataloader ---
    if args.dataset_name == "fedorenko10":
        # Path to the Fedorenko2010 stimuli, assuming it's in the parent directory
        dirpath = f"../../language-localization/fedorenko10_stimuli"
        lang_dataset = Fed10_LocLangDataset(dirpath)
    else:
        raise ValueError(f"Dataset {args.dataset_name} not implemented")

    layer_names: list[str] = get_layer_names(args.model_name)
    hidden_dim = model.config.hidden_size
    
    lang_dataloader = DataLoader(lang_dataset, batch_size=args.batch_size, num_workers=16)

    # --- Device Configuration ---
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else "cpu"
    print(f"> Using Device: {device}")

    model.eval()
    model.to(device)

    # --- Representation Extraction ---
    # Initialize dictionary to store final representations
    final_layer_representations = {
        "sentences": {layer_name: np.zeros((num_samples, hidden_dim)) for layer_name in layer_names},
        "non-words": {layer_name: np.zeros((num_samples, hidden_dim)) for layer_name in layer_names}
    }
    
    print("> Extracting representations...")
    for batch_idx, batch_data in tqdm(enumerate(lang_dataloader), total=len(lang_dataloader)):
        sents, non_words = batch_data
        
        # Tokenize sentences and non-words
        sent_tokens = tokenizer(sents, truncation=True, max_length=12, return_tensors='pt', padding=True).to(device)
        non_words_tokens = tokenizer(non_words, truncation=True, max_length=12, return_tensors='pt', padding=True).to(device)
        
        # Extract representations for both conditions
        batch_real_actv = extract_representations(model, sent_tokens["input_ids"], sent_tokens["attention_mask"], layer_names, args.embed_agg)
        batch_rand_actv = extract_representations(model, non_words_tokens["input_ids"], non_words_tokens["attention_mask"], layer_names, args.embed_agg)

        # Store extracted representations
        for layer_name in layer_names:
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + len(sents)
            final_layer_representations["sentences"][layer_name][start_idx:end_idx] = torch.stack(batch_real_actv[layer_name]).numpy()
            final_layer_representations["non-words"][layer_name][start_idx:end_idx] = torch.stack(batch_rand_actv[layer_name]).numpy()

    # --- Saving Results ---
    print(f"> Saving representations to {savepath}")
    write_pickle(savepath, final_layer_representations)