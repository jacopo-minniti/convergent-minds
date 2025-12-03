import numpy as np
from alignment.metrics.linear_partial_r2 import linear_partial_r2
from sklearn.model_selection import KFold

def test_linear_partial_r2():
    np.random.seed(42)
    n_samples = 100
    n_features_obj = 5
    n_features_llm = 10
    n_neuroids = 2
    
    # Generate synthetic data
    X_obj = np.random.randn(n_samples, n_features_obj)
    X_llm = np.random.randn(n_samples, n_features_llm)
    
    # y is linear combination of X_obj and X_llm + noise
    w_obj = np.random.randn(n_features_obj, n_neuroids)
    w_llm = np.random.randn(n_features_llm, n_neuroids)
    
    y = X_obj @ w_obj + X_llm @ w_llm + 0.1 * np.random.randn(n_samples, n_neuroids)
    
    splits = list(KFold(n_splits=5).split(X_obj))
    
    score, diagnostics = linear_partial_r2(X_obj, X_llm, y, splits)
    
    print(f"Score (Delta R2): {score}")
    print(f"Original Alignment Score (LLM Pearson): {diagnostics['original_alignment_score']}")
    print(f"Objective Alignment Score (Obj Pearson): {diagnostics['objective_alignment_score']}")
    
    assert score > 0.1, f"Score too low: {score}"
    assert diagnostics['original_alignment_score'] > 0.5, f"LLM Pearson too low: {diagnostics['original_alignment_score']}"
    assert diagnostics['objective_alignment_score'] > 0.5, f"Obj Pearson too low: {diagnostics['objective_alignment_score']}"
    
    print("Verification passed!")

if __name__ == "__main__":
    test_linear_partial_r2()
