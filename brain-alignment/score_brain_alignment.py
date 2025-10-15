import os
import sys
import json
import torch
import random 
import argparse
import warnings
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from brainscore_language import load_benchmark, ArtificialSubject
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject, get_layer_names
from transformers import AutoModelForCausalLM, AutoTokenizer


warnings.filterwarnings('ignore') 

class ModelSubject(HuggingfaceSubject):
    """A wrapper for HuggingFace models to be used as brain-score subjects."""
    def start_neural_recording(self, recording_target: ArtificialSubject.RecordingTarget, recording_type: ArtificialSubject.RecordingType):
        """Specifies which layers to record from."""
        if recording_target not in self.region_layer_mapping:
            raise NotImplementedError(f"Recording target {recording_target} not supported.")
        self._layers_to_record = self.region_layer_mapping[recording_target]

def seed_everything(seed: int):    
    """Set seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def write_pickle(path, data):
    """Write data to a pickle file."""
    with open(path, 'wb') as f:
        pkl.dump(data, f)

def score_model(
        model_name: str,
        benchmark_name: str,
        cuda: int,
        seed: int = 42,
        debug: bool = False,
        overwrite: bool = False,
):
    """Scores a model on a given brain-score benchmark."""
    seed_everything(seed=seed)

    # --- Path Definitions and File Checks ---
    model_id = f"model={model_name}_benchmark={benchmark_name}_seed={seed}"
    savepath = f"brain-alignment/dumps/scores_{model_id}.pkl"
    
    if os.path.exists(savepath) and not debug and not overwrite:
        print(f"> Run Already Exists: {savepath}")
        data = pd.read_pickle(savepath)
        print(data)
        return 

    # --- Benchmark Loading ---
    benchmark = load_benchmark(benchmark_name)
    print(f"> Running {model_id}")

    # --- Model and Tokenizer Loading ---
    device = f"cuda:{cuda}" if torch.cuda.is_available() and cuda >= 0 else "cpu"
    print(f"> Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    model.eval()

    # --- Layer and Model Subject Setup ---
    layer_names = get_layer_names(model_name, None)
     
    print("> Layer Names")
    print(layer_names)
    print()
    
    # Create a brain-score subject from the model
    layer_model = ModelSubject(
        model_id=model_id, 
        model=model, 
        tokenizer=tokenizer, 
        region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system: layer_names
        }
    )

    # --- Scoring ---
    print("> Running benchmark...")
    layer_scores = benchmark(layer_model)

    # --- Saving Results ---
    if not debug or overwrite:
        print(f"> Saving scores to {savepath}")
        write_pickle(savepath, layer_scores)

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('--model-name',  type=str, default="gpt2", help='HuggingFace model name')
    parser.add_argument('--benchmark-name',  type=str, default="Pereira2018.384sentences-cka", help='Brain-score benchmark name')
    parser.add_argument('--debug',  action='store_true', help='Debug mode')
    parser.add_argument('--overwrite',  action='store_true', help='Overwrite existing files')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')
    args = parser.parse_args()

    score_model(**vars(args))