from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np

def calculate_precision(y_true, y_pred):
    """
    Calculate precision given true labels and predicted labels.
    
    Args:
        y_true (list or array): True labels.
        y_pred (list or array): Predicted labels.
    
    Returns:
        float: Precision score.
    """
    # Ensure inputs are numpy arrays for consistency
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Validate that the lengths of y_true and y_pred match
    if len(y_true) != len(y_pred):
        raise ValueError("Length of true labels and predicted labels must be the same.")
    
    try:
        precision = precision_score(y_true, y_pred, average='weighted')
    except Exception as e:
        print(f"Error calculating precision: {e}")
        return None
    
    return precision

def calculate_recall(y_true, y_pred):
    """
    Calculate recall given true labels and predicted labels.
    
    Args:
        y_true (list or array): True labels.
        y_pred (list or array): Predicted labels.
    
    Returns:
        float: Recall score.
    """
    # Ensure inputs are numpy arrays for consistency
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Validate that the lengths of y_true and y_pred match
    if len(y_true) != len(y_pred):
        raise ValueError("Length of true labels and predicted labels must be the same.")
    
    try:
        recall = recall_score(y_true, y_pred, average='weighted')
    except Exception as e:
        print(f"Error calculating recall: {e}")
        return None
    
    return recall

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy given true labels and predicted labels.
    
    Args:
        y_true (list or array): True labels.
        y_pred (list or array): Predicted labels.
    
    Returns:
        float: Accuracy score.
    """
    # Ensure inputs are numpy arrays for consistency
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Validate that the lengths of y_true and y_pred match
    if len(y_true) != len(y_pred):
        raise ValueError("Length of true labels and predicted labels must be the same.")
    
    try:
        accuracy = accuracy_score(y_true, y_pred)
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        return None
    
    return accuracy
