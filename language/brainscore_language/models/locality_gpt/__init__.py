from brainscore_language import model_registry
from brainscore_language import ArtificialSubject
from .model import LocalityGPT2

model_registry['locality-gpt2'] = lambda: LocalityGPT2(model_id='gpt2', region_layer_mapping={
    ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})