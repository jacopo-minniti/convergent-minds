
import numpy as np
from alignment.metrics.linear_partial_r2 import linear_partial_r2

def test_linear_partial_r2_perfect_prediction():
    # Case where X_llm perfectly predicts y, and X_obj predicts nothing.
    n_samples = 20
    n_features_obj = 5
    n_features_llm = 5
    n_neuroids = 2

    np.random.seed(42)
    X_obj = np.random.randn(n_samples, n_features_obj)
    X_llm = np.random.randn(n_samples, n_features_llm)

    # y is exactly X_llm[:, 0:2]
    y = X_llm[:, :n_neuroids]

    # Splits: 2 splits
    indices = np.arange(n_samples)
    splits = [
        (indices[:10], indices[10:]),
        (indices[10:], indices[:10])
    ]

    score, diagnostics = linear_partial_r2(X_obj, X_llm, y, splits)

    print(f"Score: {score}")
    print(f"Diagnostics: {diagnostics}")
    
    # R2_combined should be close to 1.0
    # R2_baseline should be close to 0.0 (random noise)
    # Delta R2 should be close to 1.0

    assert score > 0.9, f"Score {score} is not > 0.9"

if __name__ == "__main__":
    try:
        test_linear_partial_r2_perfect_prediction()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
