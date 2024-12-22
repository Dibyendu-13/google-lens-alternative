import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def load_data(data_dir, batch_size=32, img_size=(128, 128), shuffle=True):
    """
    Load the image dataset and return a DataLoader.
    
    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for loading data.
        img_size (tuple): Size to which images will be resized.
        shuffle (bool): Whether to shuffle the dataset.
    
    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    # Define the transformations to apply to the images
    transform = transforms.Compose([
        transforms.Resize(img_size),  # Resize images to the desired size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images with ImageNet stats
    ])
    
    # Load the dataset from the directory, using ImageFolder for directory-based datasets
    dataset = ImageFolder(root=data_dir, transform=transform)
    
    # Create a DataLoader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader

def preprocess_data(data_loader, augment=False):
    """
    Preprocess data before training/evaluation (e.g., augmentation, normalization, etc.).
    
    Args:
        data_loader (DataLoader): The DataLoader containing the dataset.
        augment (bool): Whether to apply data augmentation (default False).
    
    Returns:
        DataLoader: A preprocessed DataLoader with possible augmentation.
    """
    # Define the base transformation
    transformations = [
        transforms.Resize((128, 128)),  # Resize all images to a standard size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images with ImageNet stats
    ]
    
    # If augment flag is True, apply additional transformations
    if augment:
        transformations = [
            transforms.RandomHorizontalFlip(),  # Apply random horizontal flip
            transforms.RandomRotation(20),  # Random rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
        ] + transformations  # Add base transformations after augmentation
    
    # Apply the transformations to the dataset
    transform = transforms.Compose(transformations)
    
    # Load the dataset with the transformations applied
    dataset = ImageFolder(root=data_loader.dataset.root, transform=transform)
    
    # Return a DataLoader for the augmented/preprocessed dataset
    return DataLoader(dataset, batch_size=data_loader.batch_size, shuffle=data_loader.shuffle)
