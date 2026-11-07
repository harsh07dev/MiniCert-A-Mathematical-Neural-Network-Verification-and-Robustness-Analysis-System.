"""Safe checkpoint loading and mathematical extraction for supported networks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import inspect
from pathlib import Path
import pickle
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

from models.network import MiniCertNet


FloatArray: TypeAlias = NDArray[np.float64]


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be loaded or fails strict validation."""


class UnsupportedModelError(TypeError):
    """Raised when a model's forward execution path is not explicitly supported."""


class UnsupportedLayerError(TypeError):
    """Raised when a sequential path contains an unsupported operation."""


class LayerCompatibilityError(ValueError):
    """Raised when consecutive Linear layer dimensions are incompatible."""


@dataclass(frozen=True)
class LinearLayer:
    """Affine operation y = Wx + b at its original sequence position."""

    position: int
    weight: FloatArray
    bias: FloatArray
    input_features: int
    output_features: int

    def apply(self, inputs: FloatArray) -> FloatArray:
        return inputs @ self.weight.T + self.bias


@dataclass(frozen=True)
class ReLULayer:
    """Elementwise ReLU operation at its original sequence position."""

    position: int
    activation: str = "relu"

    def apply(self, inputs: FloatArray) -> FloatArray:
        return np.maximum(inputs, 0.0)


Layer: TypeAlias = LinearLayer | ReLULayer


@dataclass(frozen=True)
class MathematicalNetwork:
    """Ordered float64 representation of a supported feature-based network.

    Inputs must have shape (input_features,) or (batch_size, input_features).
    Preprocessing in a containing model, including MiniCertNet image flattening,
    must be performed before calling forward().
    """

    layers: tuple[Layer, ...]
    input_features: int
    output_features: int
    source_model: str

    def forward(self, inputs: np.ndarray) -> FloatArray:
        values = np.asarray(inputs, dtype=np.float64)
        single_input = values.ndim == 1
        if single_input:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.input_features:
            raise ValueError(
                f"Expected shape ({self.input_features},) or "
                f"(batch_size, {self.input_features}), got {values.shape}"
            )

        for layer in self.layers:
            values = layer.apply(values)
        return values[0] if single_input else values


def save_model(
    model: nn.Module,
    file_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a state dictionary with optional constructor metadata."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "metadata": metadata or {}}
    torch.save(payload, path)


def _read_checkpoint(path: Path, device: torch.device) -> Any:
    load_options: dict[str, Any] = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = True
    try:
        return torch.load(path, **load_options)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise CheckpointError(f"Invalid or unsafe checkpoint '{path}': {exc}") from exc


def _unpack_checkpoint(
    checkpoint: Any,
    model_class: type[nn.Module],
    kwargs: dict[str, Any],
) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = checkpoint.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise CheckpointError("Checkpoint metadata must be a mapping")
        constructor_parameters = inspect.signature(model_class.__init__).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in constructor_parameters.values()
        )
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise CheckpointError("Checkpoint metadata keys must be strings")
            if key not in kwargs and (accepts_kwargs or key in constructor_parameters):
                kwargs[key] = value
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise CheckpointError("Checkpoint must contain a non-empty state dictionary")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise CheckpointError("State dictionary entries must map strings to tensors")
    return state_dict


def load_model(
    model_class: type[nn.Module],
    file_path: str | Path,
    *args: Any,
    device: str | torch.device | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate a model and strictly load a tensor-only checkpoint.

    Both raw state dictionaries and save_model() payloads are supported. Missing,
    unexpected, or mismatched parameters cause CheckpointError.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found at '{path}'")
    target_device = torch.device("cpu") if device is None else torch.device(device)
    checkpoint = _read_checkpoint(path, target_device)
    state_dict = _unpack_checkpoint(checkpoint, model_class, kwargs)

    model = model_class(*args, **kwargs)
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise CheckpointError(f"Strict checkpoint validation failed for '{path}': {exc}") from exc
    model.to(target_device)
    model.eval()
    return model


def _get_sequential_path(model: nn.Module) -> nn.Sequential:
    if isinstance(model, nn.Sequential):
        if type(model).forward is not nn.Sequential.forward:
            raise UnsupportedModelError(
                f"Unsupported Sequential subclass {type(model).__name__}: "
                "its forward method overrides standard sequential execution"
            )
        return model
    if type(model) is MiniCertNet:
        if not isinstance(model.network, nn.Sequential):
            raise UnsupportedModelError("MiniCertNet.network must be nn.Sequential")
        return model.network
    raise UnsupportedModelError(
        f"Unsupported model {type(model).__name__}: extraction supports nn.Sequential "
        "and MiniCertNet's explicitly known flatten-then-network forward path. "
        "Arbitrary forward methods cannot be reconstructed from child modules."
    )


def extract_network(model: nn.Module) -> MathematicalNetwork:
    """Extract an ordered Linear/ReLU network with float64 NumPy parameters."""
    sequence = _get_sequential_path(model)
    layers: list[Layer] = []
    input_features: int | None = None
    current_features: int | None = None

    for position, module in enumerate(sequence):
        if type(module) is nn.Linear:
            if module.bias is None:
                raise UnsupportedLayerError(
                    f"Linear layer at position {position} has no bias; y = Wx + b requires one"
                )
            if current_features is not None and module.in_features != current_features:
                raise LayerCompatibilityError(
                    f"Linear layer at position {position} expects {module.in_features} "
                    f"features, but the preceding operation outputs {current_features}"
                )
            if input_features is None:
                input_features = module.in_features
            weight = module.weight.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
            bias = module.bias.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
            layers.append(
                LinearLayer(
                    position=position,
                    weight=weight,
                    bias=bias,
                    input_features=module.in_features,
                    output_features=module.out_features,
                )
            )
            current_features = module.out_features
        elif type(module) is nn.ReLU:
            layers.append(ReLULayer(position=position))
        else:
            raise UnsupportedLayerError(
                f"Unsupported operation {type(module).__name__} at sequence position {position}; "
                "only nn.Linear and nn.ReLU are supported"
            )

    if input_features is None or current_features is None:
        raise UnsupportedModelError("Sequential path must contain at least one Linear layer")

    return MathematicalNetwork(
        layers=tuple(layers),
        input_features=input_features,
        output_features=current_features,
        source_model=type(model).__name__,
    )


def extract_layer_parameters(
    model: nn.Module, as_numpy: bool = False
) -> list[tuple[Any, Any]]:
    """Return Linear parameters in execution order after full path validation."""
    representation = extract_network(model)
    if as_numpy:
        return [
            (layer.weight, layer.bias)
            for layer in representation.layers
            if isinstance(layer, LinearLayer)
        ]

    sequence = _get_sequential_path(model)
    parameters: list[tuple[Any, Any]] = []
    for module in sequence:
        if type(module) is nn.Linear:
            parameters.append((module.weight.detach(), module.bias.detach()))
    return parameters


def inspect_model_architecture(model: nn.Module) -> dict[str, Any]:
    """Return a validated architectural summary."""
    representation = extract_network(model)
    linear_layers = [
        layer for layer in representation.layers if isinstance(layer, LinearLayer)
    ]
    return {
        "num_linear_layers": len(linear_layers),
        "total_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "layers": [
            f"Linear({layer.input_features} -> {layer.output_features})"
            for layer in linear_layers
        ],
    }

