import unittest
import sys
from unittest.mock import MagicMock

# Mock wordfreq
sys.modules["wordfreq"] = MagicMock()

import numpy as np
from alignment.objective_features import compute_length_features, compute_position_features, compute_word_position_features, simple_tokenize

class TestObjectiveFeatures(unittest.TestCase):
    def test_simple_tokenize(self):
        s = "Hello, world! This is a test."
        tokens = simple_tokenize(s)
        self.assertEqual(tokens, ["hello", "world", "this", "is", "a", "test"])

    def test_length_features(self):
        tokens = ["hello", "world"]
        feats = compute_length_features(tokens, "Hello world")
        self.assertEqual(len(feats), 1)
        self.assertEqual(feats[0], 2.0)

    def test_position_features(self):
        # Test 1st position
        feats = compute_position_features(1, 4)
        self.assertEqual(len(feats), 4)
        self.assertEqual(feats, [1.0, 0.0, 0.0, 0.0])
        
        # Test 3rd position
        feats = compute_position_features(3, 4)
        self.assertEqual(feats, [0.0, 0.0, 1.0, 0.0])
        
        # Test out of bounds (should be all zeros based on my implementation)
        feats = compute_position_features(5, 4)
        self.assertEqual(feats, [0.0, 0.0, 0.0, 0.0])

    def test_word_position_features(self):
        tokens = ["a", "b", "c"]
        feats = compute_word_position_features(tokens)
        self.assertEqual(len(feats), 9)
        # Check if values are reasonable (between 0 and 1)
        self.assertTrue(all(0.0 <= x <= 1.0 for x in feats))
        
        # Test empty
        feats = compute_word_position_features([])
        self.assertEqual(feats, [0.0] * 9)
        
        # Test single word
        feats = compute_word_position_features(["word"])
        self.assertEqual(len(feats), 9)

if __name__ == '__main__':
    unittest.main()
