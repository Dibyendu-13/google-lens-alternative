# src/models/transformer_model.py
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTFeatureExtractor

class VisionTransformerModel(nn.Module):
    def __init__(self):
        super(VisionTransformerModel, self).__init__()
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    def forward(self, x):
        return self.model(x).logits

def train_vit_model(model, train_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = torch.mean(outputs)  # Customize loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

# Example usage
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(root='./data/train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

model = VisionTransformerModel()
train_vit_model(model, train_loader)
