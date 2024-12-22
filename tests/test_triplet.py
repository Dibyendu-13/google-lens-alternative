"""
Unit tests for the Triplet Network with SimCLR.
"""

import unittest
import torch
from models.triplet_network import TripletNetwork

class TestTripletNetwork(unittest.TestCase):
    def setUp(self):
        """Set up a sample Triplet Network instance and test data."""
        self.model = TripletNetwork()
        self.anchor = torch.randn(1, 3, 128, 128)  # Anchor image
        self.positive = torch.randn(1, 3, 128, 128)  # Positive image
        self.negative = torch.randn(1, 3, 128, 128)  # Negative image

    def test_forward_pass(self):
        """Test the forward pass of the Triplet Network."""
        anchor_embedding, positive_embedding, negative_embedding = self.model(self.anchor, self.positive, self.negative)
        self.assertEqual(anchor_embedding.ndim, 2, "Anchor embedding should be a 2D tensor.")
        self.assertEqual(positive_embedding.ndim, 2, "Positive embedding should be a 2D tensor.")
        self.assertEqual(negative_embedding.ndim, 2, "Negative embedding should be a 2D tensor.")

if __name__ == "__main__":
    unittest.main()
