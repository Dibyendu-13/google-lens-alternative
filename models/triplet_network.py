# src/models/triplet_network.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class TripletNetwork(nn.Module):
    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, 128)  # Embedding size
        )

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, anchor, positive, negative):
        anchor_out = self.forward_one(anchor)
        positive_out = self.forward_one(positive)
        negative_out = self.forward_one(negative)
        return anchor_out, positive_out, negative_out

def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
    neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return loss.mean()

def train_triplet_network(model, train_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for anchor, positive, negative in train_loader:
            optimizer.zero_grad()
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            loss = triplet_loss(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

# Example usage
train_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(root='./data/train', transform=train_transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

model = TripletNetwork()
train_triplet_network(model, train_loader)
