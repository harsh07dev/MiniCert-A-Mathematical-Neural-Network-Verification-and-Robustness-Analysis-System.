from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.model_loader import (
    CheckpointError,
    LayerCompatibilityError,
    LinearLayer,
    ReLULayer,
    UnsupportedLayerError,
    UnsupportedModelError,
    extract_network,
    load_model,
)
from models.network import MiniCertNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_extract_minicertnet() -> None:
    representation = extract_network(MiniCertNet())

    assert representation.source_model == "MiniCertNet"
    assert representation.input_features == 784
    assert representation.output_features == 10
    assert len(representation.layers) == 5
    assert all(
        layer.weight.dtype == np.float64 and layer.bias.dtype == np.float64
        for layer in representation.layers
        if isinstance(layer, LinearLayer)
    )


def test_layer_sequence_preserves_execution_positions() -> None:
    representation = extract_network(MiniCertNet())

    assert [type(layer) for layer in representation.layers] == [
        LinearLayer,
        ReLULayer,
        LinearLayer,
        ReLULayer,
        LinearLayer,
    ]
    assert [layer.position for layer in representation.layers] == [0, 1, 2, 3, 4]
    assert [
        layer.activation
        for layer in representation.layers
        if isinstance(layer, ReLULayer)
    ] == ["relu", "relu"]


def test_weight_bias_and_feature_dimensions() -> None:
    representation = extract_network(MiniCertNet())
    linear_layers = [
        layer for layer in representation.layers if isinstance(layer, LinearLayer)
    ]
    expected_dimensions = [(784, 64), (64, 32), (32, 10)]

    for layer, (input_features, output_features) in zip(
        linear_layers, expected_dimensions, strict=True
    ):
        assert layer.input_features == input_features
        assert layer.output_features == output_features
        assert layer.weight.shape == (output_features, input_features)
        assert layer.bias.shape == (output_features,)


def test_extracted_values_match_torch_parameters() -> None:
    model = MiniCertNet()
    representation = extract_network(model)
    extracted = [
        layer for layer in representation.layers if isinstance(layer, LinearLayer)
    ]
    torch_layers = [module for module in model.network if isinstance(module, nn.Linear)]

    for extracted_layer, torch_layer in zip(extracted, torch_layers, strict=True):
        np.testing.assert_array_equal(
            extracted_layer.weight, torch_layer.weight.detach().numpy()
        )
        np.testing.assert_array_equal(
            extracted_layer.bias, torch_layer.bias.detach().numpy()
        )


def test_strict_checkpoint_reload(tmp_path: Path) -> None:
    original = MiniCertNet()
    checkpoint = tmp_path / "model.pth"
    torch.save(original.state_dict(), checkpoint)

    loaded = load_model(MiniCertNet, checkpoint)

    assert isinstance(loaded, MiniCertNet)
    assert not loaded.training
    for expected, actual in zip(
        original.state_dict().values(), loaded.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_trained_minicert_checkpoint_reload() -> None:
    checkpoint = PROJECT_ROOT / "saved_model.pth"
    if not checkpoint.is_file():
        pytest.skip("Trained MiniCertNet checkpoint is not available")

    loaded = load_model(MiniCertNet, checkpoint)
    representation = extract_network(loaded)

    assert isinstance(loaded, MiniCertNet)
    assert representation.output_features == 10


def test_smaller_sequential_network() -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

    representation = extract_network(model)

    assert representation.input_features == 4
    assert representation.output_features == 2
    assert [
        (layer.input_features, layer.output_features)
        for layer in representation.layers
        if isinstance(layer, LinearLayer)
    ] == [(4, 3), (3, 2)]


@pytest.mark.parametrize(
    "operation",
    [nn.Flatten(), nn.Dropout(), nn.BatchNorm1d(4), nn.Sigmoid()],
)
def test_unsupported_layers_are_rejected(operation: nn.Module) -> None:
    model = nn.Sequential(nn.Linear(4, 4), operation)

    with pytest.raises(UnsupportedLayerError, match=type(operation).__name__):
        extract_network(model)


def test_incompatible_linear_dimensions_are_rejected() -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(2, 2))

    with pytest.raises(LayerCompatibilityError, match="expects 2.*outputs 3"):
        extract_network(model)


def test_missing_checkpoint_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        load_model(MiniCertNet, tmp_path / "missing.pth")


def test_invalid_checkpoint_has_clear_error(tmp_path: Path) -> None:
    checkpoint = tmp_path / "invalid.pth"
    checkpoint.write_bytes(b"not a PyTorch checkpoint")

    with pytest.raises(CheckpointError, match="Invalid or unsafe checkpoint"):
        load_model(MiniCertNet, checkpoint)


def test_invalid_state_dictionary_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "invalid_state.pth"
    torch.save({"weight": "not a tensor"}, checkpoint)

    with pytest.raises(CheckpointError, match="strings to tensors"):
        load_model(MiniCertNet, checkpoint)


def test_strict_reload_rejects_missing_parameters(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing_parameters.pth"
    state_dict = MiniCertNet().state_dict()
    del state_dict["network.4.bias"]
    torch.save(state_dict, checkpoint)

    with pytest.raises(CheckpointError, match="Strict checkpoint validation failed"):
        load_model(MiniCertNet, checkpoint)


def test_extracted_network_reproduces_torch_logits() -> None:
    torch.manual_seed(7)
    model = MiniCertNet().eval()
    representation = extract_network(model)
    images = torch.randn(4, 1, 28, 28)

    with torch.no_grad():
        expected = model(images).numpy()
    actual = representation.forward(images.flatten(start_dim=1).numpy())

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_arbitrary_wrapper_forward_is_rejected() -> None:
    class ScaledSequential(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(nn.Linear(4, 2))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return 2 * self.network(inputs)

    with pytest.raises(UnsupportedModelError, match="Arbitrary forward methods"):
        extract_network(ScaledSequential())


def test_sequential_subclass_with_custom_forward_is_rejected() -> None:
    class ScaledSequential(nn.Sequential):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return 2 * super().forward(inputs)

    with pytest.raises(UnsupportedModelError, match="overrides standard sequential"):
        extract_network(ScaledSequential(nn.Linear(4, 2)))
