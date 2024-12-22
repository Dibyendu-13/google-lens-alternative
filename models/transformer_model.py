"""
Transformer-based Model:
A self-attention mechanism for global feature extraction.
"""
import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class TransformerModel(nn.Module):
    """
    Vision Transformer (ViT) for image feature extraction.
    """
    def __init__(self, num_classes=10):
        super(TransformerModel, self).__init__()
        # Pretrained ViT backbone
        self.vit = vit_b_16(pretrained=True)
        self.fc = nn.Linear(self.vit.heads.in_features, num_classes)
        self.vit.heads = nn.Identity()  # Remove default classification head

    def forward(self, x):
        """
        Forward pass for Transformer model.
        Args:
            x (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Class logits or feature embeddings.
        """
        features = self.vit(x)
        output = self.fc(features)
        return output
