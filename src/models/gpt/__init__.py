from brainscore import ArtificialSubject
from brainscore.model_helpers.huggingface import HuggingfaceSubject

# layer assignment based on choosing the maximally scoring layer on Pereira2018-encoding from
# https://github.com/mschrimpf/neural-nlp/blob/master/precomputed-scores.csv

def get_openai_gpt():
    return HuggingfaceSubject(model_id='openai-gpt', region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

def get_distilgpt2():
    return HuggingfaceSubject(model_id='distilgpt2', region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.5'})

def get_gpt2():
    return HuggingfaceSubject(model_id='gpt2', region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

def get_gpt2_medium():
    return HuggingfaceSubject(model_id='gpt2-medium', region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.22'})

def get_gpt2_large():
    return HuggingfaceSubject(model_id='gpt2-large', region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.33'})

def get_gpt2_xl():
    return HuggingfaceSubject(model_id='gpt2-xl', region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.43'})

def get_gpt_neo_125m():
    return HuggingfaceSubject(model_id='EleutherAI/gpt-neo-125m', region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.11'})

def get_gpt_neo_2_7B():
    return HuggingfaceSubject(model_id='EleutherAI/gpt-neo-2.7B', region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.31'})

def get_gpt_neo_1_3B():
    return HuggingfaceSubject(model_id='EleutherAI/gpt-neo-1.3B', region_layer_mapping={
        ArtificialSubject.RecordingTarget.language_system: 'transformer.h.18'})

models = {
    'openai-gpt': get_openai_gpt,
    'distilgpt2': get_distilgpt2,
    'gpt2': get_gpt2,
    'gpt2-medium': get_gpt2_medium,
    'gpt2-large': get_gpt2_large,
    'gpt2-xl': get_gpt2_xl,
    'gpt-neo-125m': get_gpt_neo_125m,
    'gpt-neo-2.7B': get_gpt_neo_2_7B,
    'gpt-neo-1.3B': get_gpt_neo_1_3B,
}
