# src/main.py
import torch
from models.autoencoder_model import Autoencoder, train_autoencoder
from models.cnn_model import CNNModel, train_cnn_model
from models.siamese_network import SiameseNetwork, train_siamese_network
from models.transformer_model import VisionTransformerModel, train_vit_model
from models.triplet_network import TripletNetwork, train_triplet_network
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(root='./data/train', transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Choose a model to train
    model = CNNModel()  # Example: use CNN model for training
    train_cnn_model(model, train_loader)

if __name__ == "__main__":
    main()
