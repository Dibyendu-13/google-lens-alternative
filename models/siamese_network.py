"""
Siamese Network:
A pairwise learning approach for image similarity using shared weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    Siamese Network for image similarity comparison.
    Takes two input images and computes a similarity score.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Shared CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Fully connected layers for similarity score
        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward_once(self, x):
        """
        Forward pass for a single image through the feature extractor.
        Args:
            x (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Feature embedding.
        """
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x

    def forward(self, x1, x2):
        """
        Forward pass for Siamese network.
        Args:
            x1 (torch.Tensor): First input image tensor.
            x2 (torch.Tensor): Second input image tensor.
        Returns:
            torch.Tensor: Similarity score.
        """
        embedding1 = self.forward_once(x1)
        embedding2 = self.forward_once(x2)
        distance = torch.abs(embedding1 - embedding2)
        output = torch.sigmoid(self.fc2(F.relu(self.fc1(distance))))
        return output
