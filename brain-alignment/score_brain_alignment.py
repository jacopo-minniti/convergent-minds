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
from brainscore_language import load_benchmark, ArtificialSubject, load_model
from brainscore_language.model_helpers.huggingface import HuggingfaceSubject, get_layer_names
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


warnings.filterwarnings('ignore') 

class ModelSubject(HuggingfaceSubject):
    """A wrapper for HuggingFace models to be used as brain-score subjects."""
    def __init__(self, model_id, model, tokenizer, region_layer_mapping, lang_unit_mask=None):
        super().__init__(model_id=model_id, model=model, tokenizer=tokenizer, region_layer_mapping=region_layer_mapping)
        self.lang_unit_mask = lang_unit_mask

    def start_neural_recording(self, recording_target: ArtificialSubject.RecordingTarget, recording_type: ArtificialSubject.RecordingType):
        """Specifies which layers to record from."""
        from collections import defaultdict
        print(">>>>> START NEURAL RECORDING <<<<<")

        if recording_target not in self.region_layer_mapping:
            raise NotImplementedError(f"Recording target {recording_target} not supported.")

        if not self.lang_unit_mask:
            print(">>>>> No language mask. Using parent start_neural_recording. <<<<<")
            super().start_neural_recording(recording_target, recording_type)
            return

        self._layer_representations = defaultdict(list)
        self._recording_target = recording_target
        self._recording_type = recording_type
        self._layers = self.region_layer_mapping[recording_target]
        print(f">>>>> Layers available in model: {self._layers} <<<<<")
        
        if self.lang_unit_mask:
            print(f">>>>> Language mask layers: {list(self.lang_unit_mask.keys())} <<<<<")
            layers_with_units = [l for l, u in self.lang_unit_mask.items() if len(u) > 0]
            print(f">>>>> Language mask layers with units: {layers_with_units} <<<<<")

        if hasattr(self, '_hook_handles'):
            for handle in self._hook_handles:
                handle.remove()
        
        self._hook_handles = []

        for layer_name in self._layers:
            layer = self._get_layer(layer_name)
            handle = layer.register_forward_hook(self._forward_hook_with_mask(layer_name))
            self._hook_handles.append(handle)
            print(f">>>>> Registered hook for layer: {layer_name} <<<<<")

    def _forward_hook_with_mask(self, layer_name):
        def hook(module, input, output):
            print(f">>>>> Hook called for layer: {layer_name} <<<<<")
            if layer_name in self.lang_unit_mask:
                unit_indices = self.lang_unit_mask[layer_name]
                if len(unit_indices) > 0:
                    print(f">>>>> Layer {layer_name} in mask with {len(unit_indices)} units. Recording. <<<<<")
                    activations = output[0][:, :, unit_indices]
                    self._layer_representations[layer_name].append(activations)
                else:
                    print(f">>>>> Layer {layer_name} in mask but 0 units. Skipping. <<<<<")
            else:
                print(f">>>>> Layer {layer_name} not in mask. Skipping. <<<<<")
        return hook

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

def read_pickle(path):
    """Read data from a pickle file."""
    with open(path, 'rb') as f:
        return pkl.load(f)

def score_model(
        model_name: str,
        benchmark_name: str,
        cuda: int,
        seed: int = 42,
        debug: bool = False,
        overwrite: bool = False,
        untrained: bool = False,
        lang_mask_path: str = None,
):
    """Scores a model on a given brain-score benchmark."""
    seed_everything(seed=seed)

    # --- Path Definitions and File Checks ---
    model_id = f"model={model_name}_benchmark={benchmark_name}_seed={seed}"
    savepath = f"dumps/scores_{model_id}.pkl"

    lang_unit_mask = None
    if lang_mask_path:
        print(f"> Loading language unit mask from: {lang_mask_path}")
        lang_unit_mask = read_pickle(lang_mask_path)
        model_id += f"_lang-mask={os.path.basename(lang_mask_path).replace('.pkl','')}"
    
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
    
    if 'modified' in model_name:
        subject = load_model(model_name)
        model = subject.basemodel
        tokenizer = subject.tokenizer
        model.to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if untrained:
            print("> Using an UNTRAINED model")
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_config(config)
            model.to(device)
            model_id += "_untrained"
        else:
            print("> Using a PRETRAINED model")
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
        },
        lang_unit_mask=lang_unit_mask
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
    parser.add_argument('--lang-mask-path', type=str, default=None, help='Path to language unit mask file')
    parser.add_argument('--untrained', action='store_true', help='Use an untrained model')
    args = parser.parse_args()

    score_model(**vars(args))