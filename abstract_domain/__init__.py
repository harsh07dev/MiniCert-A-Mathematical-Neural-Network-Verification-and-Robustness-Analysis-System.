"""
MiniCert - Abstract Domain Module
=================================

This module implements mathematical abstract domains used in Abstract
Interpretation for sound neural network verification.

Responsibilities:
- Defining the Box/Interval domain representation for multi-dimensional sets.
- Implementing sound abstract transformers for affine (linear) layers.
- Implementing sound abstract transformers for activation functions (e.g. ReLU).
- Providing domain operations such as joins, meets, and domain splits for refinement.

Assigned Developer:
- Neural Network & Abstract Interpretation Engineer (Developer 1)
"""

from abstract_domain.interval import Interval
from abstract_domain.box import Box
from abstract_domain.bounds import (
    IntermediateBounds,
    LayerBounds,
    NeuronRef,
    RefinedBounds,
)
from abstract_domain.propagation import get_intermediate_bounds, propagate_network
from abstract_domain.refinement import propagate_with_refinements

__all__ = [
    "Interval",
    "Box",
    "LayerBounds",
    "IntermediateBounds",
    "NeuronRef",
    "RefinedBounds",
    "propagate_network",
    "get_intermediate_bounds",
    "propagate_with_refinements",
]

