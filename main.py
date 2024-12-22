import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import SimpleCNN, ResNetModel  # Replace with actual models from your models folder
from utils import load_data, train_model, save_model_checkpoint, evaluate_model, fine_tune_model
from utils.metrics import calculate_precision, calculate_recall, calculate_accuracy
from utils.visualization import plot_metrics, plot_losses

def main():
    """
    Main function to train, evaluate, fine-tune, and visualize model performance.

    - Initializes model architecture.
    - Loads the dataset.
    - Trains the model and saves checkpoints.
    - Evaluates the trained model and fine-tunes if necessary.
    - Visualizes model performance using evaluation metrics and loss curves.

    This function assumes the availability of required utility functions and model architectures.
    """

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters for the experiment
    data_dir = 'path_to_data'  # Path to the dataset (change accordingly)
    batch_size = 32  # Batch size for training
    img_size = (128, 128)  # Image size for resizing
    num_epochs = 10  # Number of epochs to train the model
    learning_rate = 0.001  # Learning rate for the optimizer
    model_checkpoint_path = 'checkpoints/model_checkpoint.pth'  # Path to save model checkpoint
    fine_tuned_model_path = 'checkpoints/fine_tuned_model.pth'  # Path to save fine-tuned model

    # Load dataset using the utility function (implement load_data in utils)
    train_loader = load_data(data_dir=data_dir, batch_size=batch_size, img_size=img_size, shuffle=True)
    val_loader = load_data(data_dir=data_dir, batch_size=batch_size, img_size=img_size, shuffle=False)
    print("Datasets loaded successfully.")

    # Initialize the model architecture
    # You can switch between models like SimpleCNN or ResNetModel here
    model = SimpleCNN().to(device)  # Example: Using SimpleCNN model (you can change to ResNetModel)
    print(f"Model architecture initialized: {model}")

    # Define the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
    criterion = nn.CrossEntropyLoss()  # Cross entropy loss for classification tasks

    # Training the model
    print("Starting model training...")
    trained_model = train_model(model, train_loader, optimizer, criterion, num_epochs=num_epochs)
    print("Training completed.")

    # Save the trained model checkpoint
    save_model_checkpoint(trained_model, model_checkpoint_path)
    print(f"Model saved at {model_checkpoint_path}")

    # Evaluate the trained model on the validation data
    print("Evaluating the model on the validation data...")
    eval_metrics = evaluate_model(trained_model, val_loader, device)
    print(f"Evaluation metrics: {eval_metrics}")

    # Fine-tune the model if needed (use a lower learning rate for fine-tuning)
    print("Fine-tuning the model...")
    fine_tuned_model = fine_tune_model(trained_model, train_loader, val_loader, num_epochs=5, learning_rate=0.0001, checkpoint_path=fine_tuned_model_path)
    print("Fine-tuning completed.")

    # Evaluate the fine-tuned model
    print("Evaluating the fine-tuned model on validation data...")
    fine_tuned_metrics = evaluate_model(fine_tuned_model, val_loader, device)
    print(f"Fine-tuned model evaluation metrics: {fine_tuned_metrics}")

    # Visualization of evaluation metrics
    # Comparing the evaluation metrics (e.g., precision, recall, accuracy) of the base and fine-tuned models
    plot_metrics({'Precision': [eval_metrics['precision'], fine_tuned_metrics['precision']]}, title="Precision Comparison")
    plot_metrics({'Recall': [eval_metrics['recall'], fine_tuned_metrics['recall']]}, title="Recall Comparison")
    plot_metrics({'Accuracy': [eval_metrics['accuracy'], fine_tuned_metrics['accuracy']]}, title="Accuracy Comparison")

    # Visualization of training and validation losses
    # If losses were tracked during training, plot the losses for each epoch
    train_losses = [0.1, 0.08, 0.06, 0.04]  # Example training losses (replace with actual tracked values)
    val_losses = [0.12, 0.1, 0.09, 0.07]    # Example validation losses (replace with actual tracked values)
    plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()
