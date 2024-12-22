"""
Triplet Network with SimCLR:
Learns embeddings by comparing anchor, positive, and negative samples.
"""
import torch
import torch.nn as nn

class TripletNetwork(nn.Module):
    """
    Triplet Network for image similarity with SimCLR augmentation.
    """
    def __init__(self):
        super(TripletNetwork, self).__init__()
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward_once(self, x):
        """
        Forward pass for a single image.
        Args:
            x (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Feature embedding.
        """
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x

    def forward(self, anchor, positive, negative):
        """
        Forward pass for triplet network.
        Args:
            anchor (torch.Tensor): Anchor image tensor.
            positive (torch.Tensor): Positive image tensor.
            negative (torch.Tensor): Negative image tensor.
        Returns:
            torch.Tensor: Embeddings for anchor, positive, and negative.
        """
        anchor_out = self.forward_once(anchor)
        positive_out = self.forward_once(positive)
        negative_out = self.forward_once(negative)
        return anchor_out, positive_out, negative_out
