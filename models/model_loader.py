"""
Model Loading, Serialization, and Parameter Extraction
======================================================

Utilities for saving and loading PyTorch models, as well as extracting layer
matrices (weights and biases) in formats ready for abstract domain propagation
and LP optimization engines.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
import torch
import torch.nn as nn
import numpy as np


def save_model(
    model: nn.Module,
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model state dictionary along with optional architecture metadata.

    Args:
        model: PyTorch model to save.
        file_path: Target destination path (.pt or .pth).
        metadata: Optional dictionary containing architecture config (e.g. input_dim, hidden_dims).
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    print(f"Saved model checkpoint to '{path}'")


def load_model(
    model_class: Type[nn.Module],
    file_path: str,
    *args: Any,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Instantiate and load weights into a PyTorch model.

    Args:
        model_class: Neural network class (e.g. FCNet).
        file_path: Path to the saved checkpoint.
        *args: Positional arguments for model_class constructor.
        device: Target device to map tensors onto.
        **kwargs: Keyword arguments for model_class constructor.

    Returns:
        Instantiated model with loaded weights set to eval() mode.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found at '{path}'")

    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(path, map_location=device)

    # Check if checkpoint includes metadata wrapper or is raw state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Merge saved metadata into kwargs if not explicitly provided
        saved_meta = checkpoint.get("metadata", {})
        for k, v in saved_meta.items():
            if k not in kwargs and k in model_class.__init__.__code__.co_varnames:
                kwargs[k] = v
    else:
        state_dict = checkpoint

    model = model_class(*args, **kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def extract_layer_parameters(
    model: nn.Module, as_numpy: bool = False
) -> List[Tuple[Any, Any]]:
    """
    Extract weight and bias matrices for all linear layers in the model.

    This function serves as the bridge between PyTorch neural network representations
    and abstract interpretation / linear programming verification engines.

    Args:
        model: PyTorch model containing linear layers.
        as_numpy: If True, returns NumPy ndarrays; if False, returns PyTorch Tensors.

    Returns:
        List of tuples: [(W_0, b_0), (W_1, b_1), ..., (W_{L-1}, b_{L-1})].
        - Weight shape: (out_features, in_features)
        - Bias shape: (out_features,)
    """
    parameters: List[Tuple[Any, Any]] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.detach()
            bias = (
                module.bias.detach()
                if module.bias is not None
                else torch.zeros(module.out_features, device=weight.device)
            )

            if as_numpy:
                parameters.append((weight.cpu().numpy(), bias.cpu().numpy()))
            else:
                parameters.append((weight, bias))

    return parameters


def inspect_model_architecture(model: nn.Module) -> Dict[str, Any]:
    """
    Inspect the architecture and count trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with architecture summary.
    """
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    layer_shapes = [
        f"Linear({layer.in_features} -> {layer.out_features})"
        for layer in linear_layers
    ]

    return {
        "num_linear_layers": len(linear_layers),
        "total_trainable_parameters": total_params,
        "layers": layer_shapes,
    }

