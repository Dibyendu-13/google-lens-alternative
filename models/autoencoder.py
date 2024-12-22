"""
Autoencoder Model: 
An unsupervised learning approach for image reconstruction and representation learning.
"""
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    """
    AutoEncoder for image reconstruction.
    Encodes input images into latent representations and decodes them back to the original image dimensions.
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder: Reduces the image dimensions to a latent representation
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, 32, 32)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: (256, 16, 16)
            nn.ReLU()
        )
        # Decoder: Reconstructs the image from the latent representation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def forward(self, x):
        """
        Forward pass for the AutoEncoder.
        Args:
            x (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Reconstructed image tensor.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
