"""
Unit tests for the Transformer model.
"""

import unittest
import torch
from models.transformer_model import TransformerModel

class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        """Set up a sample Transformer model instance and test data."""
        self.model = TransformerModel()
        self.test_data = torch.randn(1, 3, 128, 128)  # Batch size 1, 3 channels, 128x128 image

    def test_forward_pass(self):
        """Test the forward pass of the Transformer model."""
        output = self.model(self.test_data)
        self.assertEqual(output.ndim, 2, "Output from Transformer should be a 2D tensor.")

if __name__ == "__main__":
    unittest.main()
