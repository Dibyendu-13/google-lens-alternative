"""
Unit tests for the CNN model.
"""

import unittest
import torch
from models.cnn_model import CNNModel

class TestCNN(unittest.TestCase):
    def setUp(self):
        """Set up a sample CNN instance and test data."""
        self.model = CNNModel()
        self.test_data = torch.randn(1, 3, 128, 128)  # Batch size 1, 3 channels, 128x128 image

    def test_forward_pass(self):
        """Test the forward pass of the CNN."""
        output = self.model(self.test_data)
        self.assertEqual(output.ndim, 2, "Output from CNN should be a 2D tensor.")

if __name__ == "__main__":
    unittest.main()
