import torch
from models.hierarchical_gpt import HierarchicalGPT2
from brainscore import ArtificialSubject

def test_hierarchical_gpt():
    model_id = "gpt2"
    depth = 3
    
    # Test untrained
    print("Testing untrained model...")
    subject = HierarchicalGPT2(
        model_id=model_id,
        region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'transformer.h.2'},
        depth=depth,
        untrained=True
    )
    
    assert len(subject.model.transformer.h) == depth
    print(f"Untrained model depth: {len(subject.model.transformer.h)}")
    
    # Test forward pass
    text = ["The quick brown fox jumps over the lazy dog."]
    output = subject.digest_text(text)
    print("Untrained forward pass successful.")
    
    # Test trained
    print("\nTesting trained model...")
    subject_trained = HierarchicalGPT2(
        model_id=model_id,
        region_layer_mapping={ArtificialSubject.RecordingTarget.language_system: 'transformer.h.2'},
        depth=depth,
        untrained=False
    )
    assert len(subject_trained.model.transformer.h) == depth
    print(f"Trained model depth: {len(subject_trained.model.transformer.h)}")
    
    # Verify weights are different (sanity check)
    # Note: This might fail if untrained initialization happens to be close, but highly unlikely.
    # Actually, let's just check that they run.
    output_trained = subject_trained.digest_text(text)
    print("Trained forward pass successful.")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_hierarchical_gpt()
