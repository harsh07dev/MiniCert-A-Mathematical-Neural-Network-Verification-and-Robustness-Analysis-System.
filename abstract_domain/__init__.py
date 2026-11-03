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

__all__ = ["Interval"]

