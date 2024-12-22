import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

def fine_tune_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, checkpoint_path='checkpoints/fine_tuned_model.pth'):
    """
    Fine-tune the model using the training dataset and evaluate using the validation dataset.
    
    Args:
        model (nn.Module): The model to be fine-tuned.
        train_loader (DataLoader): Training dataset loader.
        val_loader (DataLoader): Validation dataset loader.
        num_epochs (int): Number of epochs to fine-tune.
        learning_rate (float): Learning rate for the optimizer.
        checkpoint_path (str): Path to save the fine-tuned model.
    
    Returns:
        model: The fine-tuned model.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training loop
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Scheduler step
        scheduler.step()

        # Print training loss for each epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader)}")
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        print(f"Validation Loss: {val_loss/len(val_loader)}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), checkpoint_path)
    return model
