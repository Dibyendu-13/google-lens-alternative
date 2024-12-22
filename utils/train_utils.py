import torch
import os

def train_model(model, train_loader, optimizer, criterion, device='cpu', num_epochs=10, log_interval=10):
    """
    A utility function for training the model.
    
    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The training dataset loader.
        optimizer (optimizer): The optimizer to use for training.
        criterion (Loss function): The loss function.
        device (str): Device to run the model on ('cpu' or 'cuda').
        num_epochs (int): Number of epochs for training.
        log_interval (int): How often to log the training progress.
    
    Returns:
        model: The trained model.
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Training loop
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy (for logging)
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(labels).sum().item()
            total_predictions += labels.size(0)
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss / (batch_idx+1):.4f}")
        
        # Log per epoch loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_predictions
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    
    return model

def save_model_checkpoint(model, checkpoint_path='checkpoints/model_checkpoint.pth', optimizer=None, epoch=None):
    """
    Save the trained model checkpoint to a file.
    
    Args:
        model (nn.Module): The trained model.
        checkpoint_path (str): Path where the model will be saved.
        optimizer (optimizer, optional): The optimizer state to save. Default is None.
        epoch (int, optional): Current epoch number to save. Default is None.
    """
    # Create the checkpoints directory if it does not exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Save the model state_dict and optimizer state (if available)
    checkpoint = {'model_state_dict': model.state_dict()}
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")
