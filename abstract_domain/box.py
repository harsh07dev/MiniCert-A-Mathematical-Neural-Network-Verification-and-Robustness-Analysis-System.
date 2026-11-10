"""Multidimensional Box (vector interval) abstract domain.

This module provides the :class:`Box` class representing hyper-rectangular
subsets of R^n:

    B = { x in R^n | lower <= x <= upper }

It structurally conforms to :class:`~verification.regions.BoxProtocol` by exposing
float64 1-D `lower` and `upper` arrays. It implements sound interval bound
propagation (IBP) transformers for affine operations (y = Wx + b) and elementwise
ReLU activations (y = max(0, x)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _safe_matmul(matrix: FloatArray, vector: FloatArray) -> FloatArray:
    """Matrix-vector product adhering to the 0 * inf = 0 bound convention."""
    if not np.any(np.isinf(vector)):
        return matrix @ vector
    nonzero = matrix != 0.0
    products = np.zeros_like(matrix)
    broadcast_vec = np.broadcast_to(vector, matrix.shape)
    products[nonzero] = matrix[nonzero] * broadcast_vec[nonzero]
    return np.sum(products, axis=1)


@dataclass(frozen=True, eq=False)
class Box:
    """Immutable multidimensional hyper-rectangle over real vector spaces.

    A Box represents a Cartesian product of intervals:
        B = [l_1, u_1] x [l_2, u_2] x ... x [l_n, u_n]

    Attributes:
        lower: 1-D float64 array of component-wise lower bounds.
        upper: 1-D float64 array of component-wise upper bounds.
    """

    lower: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        try:
            lower_arr = np.array(self.lower, dtype=np.float64, copy=True)
            upper_arr = np.array(self.upper, dtype=np.float64, copy=True)
        except (ValueError, TypeError) as exc:
            raise TypeError(f"Box bounds must be convertible to float64 arrays: {exc}") from exc

        if lower_arr.ndim != 1:
            raise ValueError(f"lower bound must be a 1-D vector, got ndim={lower_arr.ndim} (shape={lower_arr.shape})")
        if upper_arr.ndim != 1:
            raise ValueError(f"upper bound must be a 1-D vector, got ndim={upper_arr.ndim} (shape={upper_arr.shape})")

        if lower_arr.size == 0 or upper_arr.size == 0:
            raise ValueError("Box bounds must not be empty")

        if lower_arr.shape != upper_arr.shape:
            raise ValueError(
                f"Bound shape mismatch: lower shape {lower_arr.shape} does not match upper shape {upper_arr.shape}"
            )

        if np.any(np.isnan(lower_arr)) or np.any(np.isnan(upper_arr)):
            raise ValueError("Box bounds cannot contain NaN")

        if np.any(lower_arr == np.inf) or np.any(upper_arr == -np.inf):
            raise ValueError("Infinite bounds must be -inf below or +inf above")

        if np.any(lower_arr > upper_arr):
            raise ValueError("Lower bound cannot exceed upper bound component-wise")

        lower_arr.flags.writeable = False
        upper_arr.flags.writeable = False

        object.__setattr__(self, "lower", lower_arr)
        object.__setattr__(self, "upper", upper_arr)

    def __eq__(self, other: object) -> bool:
        """Structural equality on lower and upper bound vectors."""
        if not isinstance(other, Box):
            return NotImplemented
        return bool(
            np.array_equal(self.lower, other.lower)
            and np.array_equal(self.upper, other.upper)
        )

    @property
    def dimension(self) -> int:
        """Return the feature dimension n of the box."""
        return int(self.lower.shape[0])

    def __len__(self) -> int:
        """Return the feature dimension n of the box."""
        return len(self.lower)

    def contains(self, point: Any) -> bool:
        """Return whether a concrete point belongs to this Box."""
        arr = np.asarray(point, dtype=np.float64)
        if arr.shape != self.lower.shape:
            raise ValueError(
                f"Point shape {arr.shape} does not match Box shape {self.lower.shape}"
            )
        if np.any(np.isnan(arr)):
            raise ValueError("Point cannot contain NaN")
        return bool(np.all((self.lower <= arr) & (arr <= self.upper)))

    def affine(self, weight: Any, bias: Any) -> Box:
        """Sound affine abstract transformer y = Wx + b via Interval Bound Propagation.

        Computes:
            W_positive = max(W, 0)
            W_negative = min(W, 0)
            output_lower = W_positive @ lower + W_negative @ upper + bias
            output_upper = W_positive @ upper + W_negative @ lower + bias

        Args:
            weight: Two-dimensional weight matrix W of shape (output_dim, input_dim).
            bias: One-dimensional bias vector b of shape (output_dim,).

        Returns:
            New Box enclosing the affine transformation image.

        Raises:
            TypeError: If weights or biases cannot be coerced to float64 arrays.
            ValueError: If weights are not 2D, biases are not 1D, dimensions mismatch,
                or weights/biases contain NaN or infinite values.
        """
        try:
            w = np.array(weight, dtype=np.float64, copy=True)
            b = np.array(bias, dtype=np.float64, copy=True)
        except (ValueError, TypeError) as exc:
            raise TypeError(f"Weight and bias must be convertible to float64 arrays: {exc}") from exc

        if w.ndim != 2:
            raise ValueError(f"Weight matrix must be 2-D, got ndim={w.ndim} (shape={w.shape})")
        if w.shape[1] != self.dimension:
            raise ValueError(
                f"Weight input dimension {w.shape[1]} does not match Box dimension {self.dimension}"
            )
        if b.ndim != 1:
            raise ValueError(f"Bias vector must be 1-D, got ndim={b.ndim} (shape={b.shape})")
        if b.shape[0] != w.shape[0]:
            raise ValueError(
                f"Bias dimension {b.shape[0]} does not match weight output dimension {w.shape[0]}"
            )
        if np.any(np.isnan(w)) or np.any(np.isnan(b)):
            raise ValueError("Weights and biases cannot contain NaN")
        if np.any(np.isinf(w)) or np.any(np.isinf(b)):
            raise ValueError("Weights and biases must be finite")

        w_pos = np.maximum(w, 0.0)
        w_neg = np.minimum(w, 0.0)

        out_lower = _safe_matmul(w_pos, self.lower) + _safe_matmul(w_neg, self.upper) + b
        out_upper = _safe_matmul(w_pos, self.upper) + _safe_matmul(w_neg, self.lower) + b

        return Box(out_lower, out_upper)

    def relu(self) -> Box:
        """Sound elementwise ReLU transformer y = max(0, x).

        Computes:
            output_lower = max(lower, 0)
            output_upper = max(upper, 0)

        Returns:
            New Box enclosing the ReLU transformation image.
        """
        return Box(np.maximum(self.lower, 0.0), np.maximum(self.upper, 0.0))

