import torch
from .metrics import calculate_precision, calculate_recall, calculate_accuracy

def evaluate_model(model, data_loader, device='cpu'):
    """
    Evaluate the model on a given dataset and compute precision, recall, and accuracy.
    
    Args:
        model (nn.Module): The trained model to be evaluated.
        data_loader (DataLoader): The DataLoader for the evaluation dataset.
        device (str): The device to run the model on ('cpu' or 'cuda').
    
    Returns:
        dict: A dictionary containing evaluation metrics: precision, recall, accuracy.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Lists to store true labels and predictions
    y_true = []
    y_pred = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for inputs, labels in data_loader:
            # Move data to the target device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get model predictions
            outputs = model(inputs)
            
            # Get the predicted class with the highest probability
            _, predicted = torch.max(outputs, 1)
            
            # Append the true and predicted labels to lists
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Compute the evaluation metrics
    try:
        precision = calculate_precision(y_true, y_pred)
        recall = calculate_recall(y_true, y_pred)
        accuracy = calculate_accuracy(y_true, y_pred)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {'precision': None, 'recall': None, 'accuracy': None}
    
    # Return the evaluation metrics in a dictionary
    return {'precision': precision, 'recall': recall, 'accuracy': accuracy}

