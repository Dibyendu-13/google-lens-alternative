"""
Unit tests for the Autoencoder model.
"""

import unittest
import torch
from models.autoencoder import AutoEncoder

class TestAutoEncoder(unittest.TestCase):
    def setUp(self):
        """Set up a sample autoencoder instance and test data."""
        self.model = AutoEncoder()
        self.test_data = torch.randn(1, 3, 128, 128)  # Batch size 1, 3 channels, 128x128 image

    def test_forward_pass(self):
        """Test the forward pass of the Autoencoder."""
        output = self.model(self.test_data)
        self.assertEqual(output.shape, self.test_data.shape, "Output shape mismatch in Autoencoder.")

if __name__ == "__main__":
    unittest.main()
