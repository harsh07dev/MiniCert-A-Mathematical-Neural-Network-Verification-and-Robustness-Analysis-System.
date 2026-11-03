"""
Neural Network Architecture for Formal Verification
===================================================

This module implements clean, modular feedforward neural network architectures
tailored for formal verification. It features explicit layer decomposition,
allowing downstream abstract transformers (such as interval bound propagation)
and optimization engines (like linear programming relaxations) to easily inspect
and traverse the network layer by layer.
"""

from typing import List, Tuple, Type
import torch
import torch.nn as nn


class MiniCertNet(nn.Module):
    """Reference fully connected network for 28x28 grayscale images.

    The sequential container exposes only Linear and ReLU operations so that
    verification engines can directly extract parameters and activations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return 10 raw logits for flattened or unflattened image batches.

        Args:
            x: Tensor of shape (batch_size, 784) or (batch_size, 1, 28, 28).

        Returns:
            Tensor of raw logits with shape (batch_size, 10).
        """
        return self.network(x.flatten(start_dim=1))


class FCNet(nn.Module):
    """
    Fully Connected Feedforward Neural Network (Multi-Layer Perceptron).

    Designed specifically with piecewise linear activations (default: ReLU) to
    support abstract interpretation and linear programming relaxations.

    Attributes:
        input_dim (int): Number of input features.
        hidden_dims (List[int]): Number of neurons in each hidden layer.
        output_dim (int): Number of output classes / logits.
        layers (nn.ModuleList): Sequential container of Linear layers.
        activation (nn.Module): Activation function applied between layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        """
        Initialize the fully connected network.

        Args:
            input_dim: Dimensionality of the input space.
            hidden_dims: List of dimensions for hidden layers (e.g. [64, 32]).
            output_dim: Dimensionality of the output layer (e.g. 10 for MNIST).
            activation: Activation class to use (default: nn.ReLU).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.activation = activation()

        self.linear_layers = nn.ModuleList()

        # Build layer dimensions: [input_dim, h1, h2, ..., output_dim]
        all_dims = [input_dim] + self.hidden_dims + [output_dim]

        for in_d, out_d in zip(all_dims[:-1], all_dims[1:]):
            self.linear_layers.append(nn.Linear(in_d, out_d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,).

        Returns:
            Output logits tensor of shape (batch_size, output_dim).
        """
        # Ensure 2D tensor (batch_size, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            # Apply activation to all layers except the final output layer
            if i < len(self.linear_layers) - 1:
                x = self.activation(x)

        return x

    def forward_with_intermediates(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass recording intermediate pre-activation and post-activation values.
        Useful for verification debugging, counterexample validation, and neuron state analysis.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            pre_activations: List of tensors before activation function at each hidden layer.
            post_activations: List of tensors after activation function at each hidden layer.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        pre_activations: List[torch.Tensor] = []
        post_activations: List[torch.Tensor] = [x]

        current = x
        for i, layer in enumerate(self.linear_layers):
            pre = layer(current)
            pre_activations.append(pre)
            if i < len(self.linear_layers) - 1:
                current = self.activation(pre)
                post_activations.append(current)
            else:
                current = pre

        return pre_activations, post_activations

    def get_layer(self, index: int) -> nn.Linear:
        """
        Retrieve a linear layer by index.

        Args:
            index: Zero-based layer index (0 to num_layers - 1).

        Returns:
            The nn.Linear layer.
        """
        return self.linear_layers[index]

    @property
    def num_layers(self) -> int:
        """Returns the total number of linear transformations."""
        return len(self.linear_layers)

    def get_weights_and_biases(
        self,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract weights and biases for all layers as a list of tuples.
        Directly consumed by abstract transformers and LP optimizers.

        Returns:
            List of (weight, bias) tuples where:
                weight shape: (out_features, in_features)
                bias shape: (out_features,)
        """
        params = []
        for layer in self.linear_layers:
            w = layer.weight.detach()
            b = layer.bias.detach() if layer.bias is not None else torch.zeros(layer.out_features)
            params.append((w, b))
        return params

