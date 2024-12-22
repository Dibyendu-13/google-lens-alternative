"""
Unit tests for the Siamese Network model.
"""

import unittest
import torch
from models.siamese_network import SiameseNetwork

class TestSiameseNetwork(unittest.TestCase):
    def setUp(self):
        """Set up a sample Siamese network instance and test data."""
        self.model = SiameseNetwork()
        self.test_data1 = torch.randn(1, 3, 128, 128)  # Batch size 1, 3 channels, 128x128 image
        self.test_data2 = torch.randn(1, 3, 128, 128)  # Second input image

    def test_forward_pass(self):
        """Test the forward pass of the Siamese Network."""
        output = self.model(self.test_data1, self.test_data2)
        self.assertEqual(output.shape, torch.Size([1]), "Output should be a single similarity score.")

if __name__ == "__main__":
    unittest.main()
