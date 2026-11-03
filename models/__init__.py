"""
MiniCert - Models Module
========================

This module provides modular PyTorch neural network architectures, training
routines, and weight extraction/loading utilities suitable for formal verification
and robustness certification.

Responsibilities:
- Defining feedforward architectures (e.g., Fully Connected Networks with ReLU activations).
- Exposing linear layer weights and biases for interval propagation and LP formulation.
- Training models on standard datasets (e.g., MNIST, synthetic benchmarks).
- Model serialization, deserialization, and parameter inspection.

Assigned Developer:
- Neural Network & Abstract Interpretation Engineer (Developer 1)
"""

from models.network import FCNet
from models.train import train_model, evaluate
from models.model_loader import save_model, load_model, extract_layer_parameters

__all__ = [
    "FCNet",
    "train_model",
    "evaluate",
    "save_model",
    "load_model",
    "extract_layer_parameters",
]

