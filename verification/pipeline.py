"""End-to-end neural network robustness verification orchestration pipeline.

This module connects the vector abstract-domain Interval Bound Propagation (IBP)
engine with formal safety checking to certify adversarial robustness over an
input perturbation region.
"""

from __future__ import annotations

from dataclasses import replace

from abstract_domain.propagation import propagate_network
from models.model_loader import MathematicalNetwork
from verification import VerificationResult
from verification.regions import InputRegion
from verification.safety import SafetyChecker


def verify_robustness(
    network: MathematicalNetwork,
    region: InputRegion,
    target_class: int,
) -> VerificationResult:
    """Certify whether a neural network robustly predicts target_class over an InputRegion.

    Executes network-level Interval Bound Propagation (IBP) to compute certified
    lower and upper output bounds, then queries Member 2's SafetyChecker to evaluate
    strict class dominance:

        for every competitor j != target_class: lower[target_class] > upper[j]

    Args:
        network: Extracted float64 MathematicalNetwork containing ordered Linear and ReLU layers.
        region: Input perturbation region (L-infinity ball around a nominal center point).
        target_class: Protected ground-truth class index (non-negative integer).

    Returns:
        VerificationResult with status CERTIFIED (if dominance holds strictly) or
        INCONCLUSIVE (if bounds overlap or fail dominance), carrying the certified margin,
        explanation, and attached perturbation_epsilon.

    Raises:
        TypeError: If network is not a MathematicalNetwork, region is not an InputRegion,
            or target_class is not an integer.
        ValueError: If region dimension does not match network.input_features or
            target_class is out of range for network.output_features.
    """
    if not isinstance(network, MathematicalNetwork):
        raise TypeError(f"network must be a MathematicalNetwork, got {type(network).__name__}")

    if not isinstance(region, InputRegion):
        raise TypeError(f"region must be an InputRegion, got {type(region).__name__}")

    if isinstance(target_class, bool) or not isinstance(target_class, int):
        raise TypeError("target_class must be an integer class index")

    if not (0 <= target_class < network.output_features):
        raise ValueError(
            f"target_class {target_class} is out of range for {network.output_features} classes"
        )

    if region.dimension != network.input_features:
        raise ValueError(
            f"region dimension {region.dimension} does not match network.input_features {network.input_features}"
        )

    output_box = propagate_network(network, region)
    result = SafetyChecker().check(output_box.lower, output_box.upper, target_class)

    return replace(result, perturbation_epsilon=region.radius)

