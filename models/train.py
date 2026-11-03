"""
Model Training and Evaluation Pipeline
======================================

Provides modular, standard routines to train feedforward neural networks using
PyTorch. Designed to produce well-trained models for subsequent verification and
robustness evaluation.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train a model for a single epoch.

    Args:
        model: PyTorch neural network to train.
        dataloader: Training DataLoader.
        criterion: Loss function (e.g., nn.CrossEntropyLoss).
        optimizer: Optimization algorithm (e.g., Adam or SGD).
        device: Computing device ('cpu' or 'cuda').

    Returns:
        Dictionary containing average 'loss' and 'accuracy' for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Flatten inputs if necessary (e.g. 28x28 images to 784-dim vectors)
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), -1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = correct / max(total_samples, 1)

    return {"loss": avg_loss, "accuracy": accuracy}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model performance on a test/validation dataset.

    Args:
        model: PyTorch neural network.
        dataloader: Validation/Test DataLoader.
        criterion: Loss function.
        device: Computing device ('cpu' or 'cuda').

    Returns:
        Dictionary containing average 'loss' and 'accuracy'.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if inputs.dim() > 2:
                inputs = inputs.view(inputs.size(0), -1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = correct / max(total_samples, 1)

    return {"loss": avg_loss, "accuracy": accuracy}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full training loop for a neural network model.

    Args:
        model: PyTorch model.
        train_loader: DataLoader with training data.
        val_loader: Optional DataLoader with validation data.
        epochs: Number of training epochs (default: 10).
        lr: Learning rate for Adam optimizer (default: 0.001).
        device: Target device (defaults to CUDA if available, else CPU).
        save_path: Optional path to save the trained model weights.

    Returns:
        Dictionary containing training history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print(f"Training on device: {device} for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])

        log_msg = f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}"

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            log_msg += f" | Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"

        print(log_msg)

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved successfully to '{save_path}'")

    return history

