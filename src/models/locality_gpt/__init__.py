from brainscore import ArtificialSubject
from .model import LocalityGPT2

def get_locality_gpt2():
    return LocalityGPT2(model_id='gpt2', region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})