# src/models/siamese_network.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        out1 = self.forward_one(input1)
        out2 = self.forward_one(input2)
        return out1, out2

def train_siamese_network(model, train_loader, epochs=10):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for img1, img2, label in train_loader:
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

# Example usage
train_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load your dataset for siamese network and adjust the data to suit siamese training
train_data = datasets.ImageFolder(root='./data/train', transform=train_transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

model = SiameseNetwork()
train_siamese_network(model, train_loader)
