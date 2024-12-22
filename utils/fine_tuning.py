import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

def fine_tune_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, checkpoint_path='checkpoints/fine_tuned_model.pth'):
    """
    Fine-tune the given model on the training dataset and evaluate its performance on the validation dataset.
    
    This function performs the following steps:
    1. Initializes the loss function (CrossEntropyLoss) for classification tasks.
    2. Uses the Adam optimizer to update model parameters based on gradients.
    3. Applies a StepLR scheduler to adjust the learning rate periodically.
    4. Runs the training loop for the specified number of epochs.
    5. Evaluates the model on the validation set after each epoch.
    6. Saves the fine-tuned model's weights to a specified checkpoint path.
    
    Args:
        model (torch.nn.Module): The deep learning model to be fine-tuned.
        train_loader (DataLoader): The DataLoader object for the training dataset.
        val_loader (DataLoader): The DataLoader object for the validation dataset.
        num_epochs (int, optional): The number of epochs for fine-tuning. Defaults to 10.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        checkpoint_path (str, optional): The path where the fine-tuned model's state_dict will be saved. Defaults to 'checkpoints/fine_tuned_model.pth'.
    
    Returns:
        model (torch.nn.Module): The fine-tuned model after training.
    """
    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Loss function for classification tasks
    optimizer = Adam(model.parameters(), lr=learning_rate)  # Adam optimizer for parameter updates
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Reduces learning rate by a factor of 0.1 every 5 epochs

    # Start fine-tuning
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0  # Initialize loss tracker
        
        # Training loop: Iterate through the training data
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero out gradients from previous iteration
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagate gradients
            optimizer.step()  # Update model parameters
            running_loss += loss.item()  # Track running loss

        # Update the learning rate scheduler
        scheduler.step()

        # Output the training loss for the current epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader):.4f}")
        
        # Validation loop: Evaluate the model's performance on the validation set
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0  # Initialize validation loss tracker
        
        with torch.no_grad():  # Disable gradient computation during evaluation
            for inputs, labels in val_loader:
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate loss
                val_loss += loss.item()  # Accumulate validation loss
        
        # Output the validation loss for the current epoch
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

    # Save the fine-tuned model's parameters to a checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model fine-tuned and saved to {checkpoint_path}")
    
    return model

