import pytest
import torch
import torch.nn as nn

from models.network import MiniCertNet


def test_model_initialization() -> None:
    model = MiniCertNet()

    assert isinstance(model, nn.Module)
    assert isinstance(model.network, nn.Sequential)
    assert len(model.network) == 5


@pytest.mark.parametrize(
    ("input_shape", "batch_size"),
    [
        ((1, 1, 28, 28), 1),
        ((1, 784), 1),
        ((16, 1, 28, 28), 16),
    ],
)
def test_forward_output_shape_and_finite_values(
    input_shape: tuple[int, ...], batch_size: int
) -> None:
    model = MiniCertNet()

    output = model(torch.randn(input_shape))

    assert output.shape == (batch_size, 10)
    assert torch.isfinite(output).all()


def test_total_trainable_parameters() -> None:
    model = MiniCertNet()

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    assert trainable_parameters == 52_650


@pytest.mark.parametrize(
    ("layer_index", "weight_shape", "bias_shape"),
    [
        (0, (64, 784), (64,)),
        (2, (32, 64), (32,)),
        (4, (10, 32), (10,)),
    ],
)
def test_linear_layer_dimensions(
    layer_index: int, weight_shape: tuple[int, int], bias_shape: tuple[int]
) -> None:
    layer = MiniCertNet().network[layer_index]

    assert isinstance(layer, nn.Linear)
    assert layer.weight.shape == weight_shape
    assert layer.bias is not None
    assert layer.bias.shape == bias_shape


def test_relu_layers_execution_order() -> None:
    layer_types = [type(layer) for layer in MiniCertNet().network]

    assert layer_types == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
