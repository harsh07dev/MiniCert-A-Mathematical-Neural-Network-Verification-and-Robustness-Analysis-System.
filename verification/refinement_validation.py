"""Empirical validation of refined neuron bounds against sampled network activations.

This module provides empirical testing tools to evaluate proposed
:class:`~abstract_domain.bounds.RefinedBounds` against actual concrete intermediate
activations. It samples inputs within an :class:`~verification.regions.InputRegion`,
computes the layer-by-layer activations through a
:class:`~models.model_loader.MathematicalNetwork`, and reports whether any sample
violates the proposed refined intervals.

Important Note:
    Empirical sampling can identify unsound refinements (by demonstrating a
    concrete counterexample where an activation falls outside the interval), but
    cannot prove mathematical soundness across all uncountably many inputs in the
    region. Passing empirical validation is a necessary consistency check, not a
    formal certification.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from abstract_domain.bounds import NeuronRef, RefinedBounds
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer
from verification.regions import InputRegion


class ActivationTrace(tuple):
    """Tuple of per-layer activation dictionaries, indexable by layer index or NeuronRef."""

    def __getitem__(self, key: int | slice | NeuronRef) -> Any:
        if isinstance(key, NeuronRef):
            if 0 <= key.layer_index < len(self):
                layer_dict = super().__getitem__(key.layer_index)
                if key in layer_dict:
                    return layer_dict[key]
                raise KeyError(f"NeuronRef {key} not found in layer {key.layer_index}")
            raise KeyError(
                f"NeuronRef layer_index {key.layer_index} out of range for trace with {len(self)} layers"
            )
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, NeuronRef):
            if 0 <= key.layer_index < len(self):
                return key in super().__getitem__(key.layer_index)
            return False
        return super().__contains__(key)

    def get(self, neuron: NeuronRef, default: Any = None) -> Any:
        """Safe lookup by NeuronRef returning default if not present."""
        try:
            return self[neuron]
        except (KeyError, IndexError):
            return default


def sample_region_inputs(
    input_region: InputRegion,
    num_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate reproducible uniform random samples within an input perturbation region.

    Args:
        input_region: The bounded InputRegion to sample from.
        num_samples: Number of concrete input samples to generate (must be > 0).
        seed: Random seed for deterministic reproducibility.

    Returns:
        Two-dimensional NumPy float64 array of shape (num_samples, input_dimension).

    Raises:
        TypeError: If input_region is not an InputRegion, or num_samples/seed is not an integer.
        ValueError: If num_samples <= 0 or the region has unbounded infinite limits.
    """
    if not isinstance(input_region, InputRegion):
        raise TypeError(f"input_region must be an InputRegion, got {type(input_region).__name__}")

    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        raise TypeError(f"num_samples must be an integer, got {type(num_samples).__name__}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")

    if np.any(np.isinf(input_region.lower)) or np.any(np.isinf(input_region.upper)):
        raise ValueError("Cannot sample uniformly from an unbounded region with infinite limits")

    rng = np.random.default_rng(seed)
    samples = rng.uniform(
        input_region.lower, input_region.upper, size=(num_samples, input_region.dimension)
    )
    return np.asarray(samples, dtype=np.float64)


def evaluate_intermediate_activations(
    network: MathematicalNetwork,
    inputs: np.ndarray,
) -> tuple[dict[NeuronRef, np.ndarray], ...]:
    """Evaluate concrete intermediate layer and neuron activations for a batch of inputs.

    Preserves actual network layer indices and phase designations:
    - LinearLayer: Records affine output (inputs @ W^T + b) under phase "pre".
    - ReLULayer: Records incoming values under phase "pre" and ReLU outputs (max(0, x))
      under phase "post".

    Args:
        network: MathematicalNetwork containing ordered Linear and ReLU layers.
        inputs: 1-D vector of shape (input_features,) or 2-D array of shape
            (num_samples, input_features).

    Returns:
        Tuple of per-layer dictionaries mapping each NeuronRef to a 1-D NumPy array
        of activation values across the sample batch. Also indexable directly by NeuronRef.

    Raises:
        TypeError: If network is not a MathematicalNetwork or an unsupported layer is encountered.
        ValueError: If inputs has invalid shape, mismatched feature dimension, or contains NaN.
    """
    if not isinstance(network, MathematicalNetwork):
        raise TypeError(f"network must be a MathematicalNetwork, got {type(network).__name__}")

    values = np.asarray(inputs, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(
            f"inputs must be 1-D or 2-D array, got ndim={values.ndim} (shape={values.shape})"
        )
    if values.shape[0] == 0:
        raise ValueError("inputs must not be empty")
    if values.shape[1] != network.input_features:
        raise ValueError(
            f"inputs feature dimension {values.shape[1]} does not match "
            f"network.input_features {network.input_features}"
        )
    if np.any(np.isnan(values)):
        raise ValueError("inputs cannot contain NaN")

    layer_dicts: list[dict[NeuronRef, np.ndarray]] = []
    current_values = values.copy()

    for layer_idx, layer in enumerate(network.layers):
        layer_dict: dict[NeuronRef, np.ndarray] = {}

        if isinstance(layer, LinearLayer):
            affine_out = layer.apply(current_values)
            for j in range(layer.output_features):
                ref = NeuronRef(layer_index=layer_idx, phase="pre", neuron_index=j)
                layer_dict[ref] = affine_out[:, j].copy()
            layer_dicts.append(layer_dict)
            current_values = affine_out

        elif isinstance(layer, ReLULayer):
            for j in range(current_values.shape[1]):
                pre_ref = NeuronRef(layer_index=layer_idx, phase="pre", neuron_index=j)
                layer_dict[pre_ref] = current_values[:, j].copy()

            post_values = np.maximum(current_values, 0.0)
            for j in range(post_values.shape[1]):
                post_ref = NeuronRef(layer_index=layer_idx, phase="post", neuron_index=j)
                layer_dict[post_ref] = post_values[:, j].copy()

            layer_dicts.append(layer_dict)
            current_values = post_values

        else:
            layer_name = type(layer).__name__
            pos = getattr(layer, "position", layer_idx)
            raise TypeError(
                f"Unsupported layer type '{layer_name}' at sequence position {pos}; "
                "only LinearLayer and ReLULayer are supported"
            )

    return ActivationTrace(layer_dicts)


def validate_refined_bounds_empirically(
    network: MathematicalNetwork,
    input_region: InputRegion,
    refined_bounds: tuple[RefinedBounds, ...] | list[RefinedBounds],
    num_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Empirically test whether proposed refined bounds enclose actual network activations.

    Draws reproducible uniform samples within the input region, evaluates intermediate
    activations across the entire network, and checks each refined neuron's interval.

    Args:
        network: MathematicalNetwork to evaluate.
        input_region: Input perturbation region to sample from.
        refined_bounds: Sequence of RefinedBounds proposals to validate.
        num_samples: Number of random input samples to evaluate (default: 1000).
        seed: Random seed for sampling reproducibility (default: 42).

    Returns:
        Structured JSON-serializable dictionary reporting results:
        {
            "num_samples": int,
            "seed": int,
            "all_refinements_passed": bool,
            "results": [
                {
                    "neuron": {"layer_index": int, "phase": str, "neuron_index": int},
                    "samples_checked": int,
                    "violations": int,
                    "max_lower_violation": float,
                    "max_upper_violation": float,
                    "passed": bool,
                },
                ...
            ]
        }

    Raises:
        TypeError: If inputs fail type validation.
        ValueError: If num_samples <= 0 or a NeuronRef is out of range.
    """
    if not isinstance(network, MathematicalNetwork):
        raise TypeError(f"network must be a MathematicalNetwork, got {type(network).__name__}")
    if not isinstance(input_region, InputRegion):
        raise TypeError(f"input_region must be an InputRegion, got {type(input_region).__name__}")
    if not isinstance(refined_bounds, (tuple, list)):
        raise TypeError(
            "refined_bounds must be a tuple or list of RefinedBounds, "
            f"got {type(refined_bounds).__name__}"
        )
    for idx, item in enumerate(refined_bounds):
        if not isinstance(item, RefinedBounds):
            raise TypeError(f"item at index {idx} in refined_bounds must be a RefinedBounds")

    samples = sample_region_inputs(input_region, num_samples=num_samples, seed=seed)
    trace = evaluate_intermediate_activations(network, samples)

    results: list[dict[str, Any]] = []

    for item in refined_bounds:
        neuron = item.neuron
        if not (0 <= neuron.layer_index < len(network.layers)):
            raise ValueError(
                f"layer_index {neuron.layer_index} is out of range for network with "
                f"{len(network.layers)} layers"
            )

        if neuron not in trace:
            layer = network.layers[neuron.layer_index]
            if isinstance(layer, LinearLayer) and neuron.phase != "pre":
                raise ValueError(
                    f"linear layer {neuron.layer_index} does not support phase {neuron.phase!r}"
                )
            raise ValueError(
                f"neuron reference {neuron} is invalid or out of range for layer {neuron.layer_index}"
            )

        activations = trace[neuron]
        lower_violations = activations < item.lower
        upper_violations = activations > item.upper
        any_violations = lower_violations | upper_violations
        num_violations = int(np.sum(any_violations))

        max_lower_viol = (
            float(np.max(item.lower - activations[lower_violations]))
            if np.any(lower_violations)
            else 0.0
        )
        max_upper_viol = (
            float(np.max(activations[upper_violations] - item.upper))
            if np.any(upper_violations)
            else 0.0
        )

        results.append(
            {
                "neuron": {
                    "layer_index": neuron.layer_index,
                    "phase": neuron.phase,
                    "neuron_index": neuron.neuron_index,
                },
                "samples_checked": num_samples,
                "violations": num_violations,
                "max_lower_violation": max_lower_viol,
                "max_upper_violation": max_upper_viol,
                "passed": num_violations == 0,
            }
        )

    all_passed = all(r["passed"] for r in results) if results else True

    return {
        "num_samples": num_samples,
        "seed": seed,
        "all_refinements_passed": all_passed,
        "results": results,
    }


__all__ = [
    "sample_region_inputs",
    "evaluate_intermediate_activations",
    "validate_refined_bounds_empirically",
]

