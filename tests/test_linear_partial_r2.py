import unittest
import numpy as np
from alignment.metrics.linear_partial_r2 import linear_partial_r2

class TestLinearPartialR2(unittest.TestCase):
    def test_perfect_prediction(self):
        # Synthetic data
        n_samples = 100
        n_features_obj = 5
        n_features_llm = 10
        n_neuroids = 2
        
        np.random.seed(42)
        X_obj = np.random.randn(n_samples, n_features_obj)
        X_llm = np.random.randn(n_samples, n_features_llm)
        
        # y is perfectly predicted by X_obj + X_llm
        # y = X_obj @ w_obj + X_llm @ w_llm
        w_obj = np.random.randn(n_features_obj, n_neuroids)
        w_llm = np.random.randn(n_features_llm, n_neuroids)
        
        y = X_obj @ w_obj + X_llm @ w_llm
        
        # Create splits (simple KFold)
        indices = np.arange(n_samples)
        splits = []
        for i in range(5):
            test_idx = indices[i*20 : (i+1)*20]
            train_idx = np.setdiff1d(indices, test_idx)
            splits.append((train_idx, test_idx))
            
        # Run metric
        # We expect high R2 combined, and positive delta R2
        score, diagnostics = linear_partial_r2(X_obj, X_llm, y, splits)
        
        print(f"Score: {score}")
        print(f"Diagnostics: {diagnostics.keys()}")
        
        # Check if score is reasonable (should be close to 1.0 combined, and delta > 0)
        # Since we use Ridge, it might not be exactly 1.0 due to regularization, but should be high.
        avg_r2_combined = diagnostics['obj_llm_explained_variance']
        self.assertGreater(avg_r2_combined, 0.9)
        self.assertGreater(score, 0.0)

    def test_no_signal_llm(self):
        # y depends only on X_obj
        n_samples = 100
        X_obj = np.random.randn(n_samples, 5)
        X_llm = np.random.randn(n_samples, 10) # Random noise
        y = X_obj @ np.random.randn(5, 1)
        
        indices = np.arange(n_samples)
        splits = [ (indices[:80], indices[80:]) ]
        
        score, diagnostics = linear_partial_r2(X_obj, X_llm, y, splits)
        
        # Delta R2 should be near 0 (or negative due to overfitting/noise)
        print(f"No Signal Score: {score}")
        self.assertLess(abs(score), 0.1)

if __name__ == '__main__':
    unittest.main()
