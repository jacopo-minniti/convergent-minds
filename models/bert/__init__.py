from brainscore import ArtificialSubject
from brainscore.model_helpers.huggingface import HuggingfaceSubject

def get_bert_base_uncased():
    return HuggingfaceSubject(
        model_id='bert-base-uncased',
        region_layer_mapping={
            ArtificialSubject.RecordingTarget.language_system: 'encoder.layer.9' # Selecting a late layer like GPT2's layer 11 equivalent
        },
        representation_token_index=0 # [CLS] token
    )

models = {
    'bert-base-uncased': get_bert_base_uncased,
}
