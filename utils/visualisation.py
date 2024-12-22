import matplotlib.pyplot as plt

def plot_metrics(metrics, title="Metrics Plot", xlabel="Epochs", ylabel="Score", figsize=(10, 6)):
    """
    Plot the performance metrics (e.g., accuracy, precision, recall) over epochs.
    
    Args:
        metrics (dict): A dictionary where keys are the metric names (e.g., 'accuracy', 'precision')
                        and values are lists or arrays of metric values over epochs.
        title (str): Title of the plot. Default is "Metrics Plot".
        xlabel (str): Label for the x-axis. Default is "Epochs".
        ylabel (str): Label for the y-axis. Default is "Score".
        figsize (tuple): Tuple indicating the figure size. Default is (10, 6).
    
    Example:
        metrics = {'accuracy': [0.6, 0.7, 0.8, 0.9], 'precision': [0.65, 0.75, 0.85, 0.95]}
        plot_metrics(metrics)
    """
    plt.figure(figsize=figsize)
    
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name, marker='o')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_losses(train_losses, val_losses, title="Loss Plot", xlabel="Epochs", ylabel="Loss", figsize=(10, 6)):
    """
    Plot the training and validation losses over epochs.
    
    Args:
        train_losses (list): List of training losses over epochs.
        val_losses (list): List of validation losses over epochs.
        title (str): Title of the plot. Default is "Loss Plot".
        xlabel (str): Label for the x-axis. Default is "Epochs".
        ylabel (str): Label for the y-axis. Default is "Loss".
        figsize (tuple): Tuple indicating the figure size. Default is (10, 6).
    
    Example:
        train_losses = [0.6, 0.5, 0.4, 0.3]
        val_losses = [0.65, 0.55, 0.45, 0.35]
        plot_losses(train_losses, val_losses)
    """
    plt.figure(figsize=figsize)
    
    # Plotting training and validation losses
    plt.plot(train_losses, label="Training Loss", marker='o', linestyle='-', color='b')
    plt.plot(val_losses, label="Validation Loss", marker='o', linestyle='--', color='r')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
