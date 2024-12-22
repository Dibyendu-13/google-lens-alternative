"""
CNN Model:
A supervised learning approach leveraging convolutional layers for feature extraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    Convolutional Neural Network for image feature extraction and similarity comparison.
    """
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # Output: (64, 128, 128)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: (128, 64, 64)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Output: (256, 32, 32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass for CNN model.
        Args:
            x (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Class logits or feature embeddings.
        """
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
