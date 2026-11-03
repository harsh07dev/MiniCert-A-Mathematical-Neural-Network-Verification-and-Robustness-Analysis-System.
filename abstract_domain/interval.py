"""
Interval (Box) Abstract Domain
==============================

Implements the multi-dimensional Interval (Box) abstract domain for sound
Abstract Interpretation of neural networks.

Mathematical Foundation:
------------------------
An element of the interval domain represents a hyper-rectangle:
    X = { x in R^n | l_i <= x_i <= u_i, for all i = 1, ..., n }

Soundness Guarantee:
    If a concrete input x in [l_in, u_in], then the propagated abstract interval
    [l_out, u_out] is guaranteed to contain the concrete output f(x):
        f(x) in [l_out, u_out]

Abstract Transformers:
----------------------
1. Affine / Linear Transformer (y = W x + b):
   Decomposing weight matrix W into positive and negative components:
       W^+ = max(W, 0)
       W^- = min(W, 0)
   The sound output bounds are computed as:
       l_out = W^+ l + W^- u + b
       u_out = W^+ u + W^- l + b

2. ReLU Transformer:
   Since ReLU(z) = max(0, z) is monotonic:
       l_out = max(0, l)
       u_out = max(0, u)
"""

from typing import Tuple, Union
import torch


class Interval:
    """
    Multi-dimensional Interval (Box) abstraction.

    Encapsulates lower and upper bound tensors with sound arithmetic operations
    and neural network abstract transformers.

    Attributes:
        lower (torch.Tensor): Tensor representing the component-wise lower bounds (l).
        upper (torch.Tensor): Tensor representing the component-wise upper bounds (u).
    """

    def __init__(
        self,
        lower: Union[torch.Tensor, float, int],
        upper: Union[torch.Tensor, float, int],
    ) -> None:
        """
        Initialize an interval domain element.

        Args:
            lower: Lower bound(s). Converted to torch.FloatTensor.
            upper: Upper bound(s). Converted to torch.FloatTensor.
        """
        if not isinstance(lower, torch.Tensor):
            lower = torch.tensor(lower, dtype=torch.float32)
        else:
            lower = lower.to(dtype=torch.float32)

        if not isinstance(upper, torch.Tensor):
            upper = torch.tensor(upper, dtype=torch.float32)
        else:
            upper = upper.to(dtype=torch.float32)

        # Broadcast shapes if needed
        if lower.shape != upper.shape:
            lower, upper = torch.broadcast_tensors(lower, upper)

        # Numerical sanity check (allowing negligible float epsilon tolerance)
        if (lower > upper + 1e-6).any():
            raise ValueError(
                "Invalid interval bounds: Lower bound cannot strictly exceed upper bound."
            )

        # Clamp tiny numerical discrepancies where lower slightly exceeds upper
        self.lower = torch.minimum(lower, upper)
        self.upper = torch.maximum(lower, upper)

    @classmethod
    def from_center_and_epsilon(
        cls, center: torch.Tensor, epsilon: float
    ) -> "Interval":
        """
        Factory method to construct an L-infinity perturbation ball.

        Given a concrete point x_0 and perturbation radius eps:
            B_inf(x_0, eps) = [x_0 - eps, x_0 + eps]

        Args:
            center: Nominal concrete input vector (e.g. an image).
            epsilon: Maximum allowed L-infinity perturbation radius (eps >= 0).

        Returns:
            Interval enclosing the L-infinity ball.
        """
        if epsilon < 0:
            raise ValueError(f"Epsilon must be non-negative, got {epsilon}")

        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=torch.float32)

        eps_tensor = torch.tensor(epsilon, dtype=torch.float32, device=center.device)
        return cls(center - eps_tensor, center + eps_tensor)

    @property
    def shape(self) -> torch.Size:
        """Returns the shape of the interval tensor."""
        return self.lower.shape

    @property
    def device(self) -> torch.device:
        """Returns the device on which bounds reside."""
        return self.lower.device

    @property
    def width(self) -> torch.Tensor:
        """Calculates component-wise width: u - l."""
        return self.upper - self.lower

    @property
    def radius(self) -> torch.Tensor:
        """Calculates component-wise radius: (u - l) / 2."""
        return (self.upper - self.lower) / 2.0

    @property
    def center(self) -> torch.Tensor:
        """Calculates component-wise midpoint / center: (l + u) / 2."""
        return (self.lower + self.upper) / 2.0

    def contains(self, x: torch.Tensor, tol: float = 1e-6) -> bool:
        """
        Check whether a concrete point x lies entirely within this interval.

        Args:
            x: Concrete tensor to check.
            tol: Numerical tolerance for floating-point comparisons.

        Returns:
            True if l - tol <= x <= u + tol for all components, False otherwise.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device=self.device, dtype=self.lower.dtype)
        return bool(torch.all(x >= self.lower - tol) and torch.all(x <= self.upper + tol))

    def affine(self, weight: torch.Tensor, bias: torch.Tensor) -> "Interval":
        """
        Sound linear / affine abstract transformer: y = W x + b.

        Uses classic Interval Bound Propagation (IBP):
            W^+ = max(W, 0)
            W^- = min(W, 0)
            l_out = l @ (W^+)^T + u @ (W^-)^T + b
            u_out = u @ (W^+)^T + l @ (W^-)^T + b

        Args:
            weight: Weight matrix W of shape (out_features, in_features).
            bias: Bias vector b of shape (out_features,).

        Returns:
            New Interval enclosing the affine output space.
        """
        weight = weight.to(device=self.device, dtype=self.lower.dtype)
        bias = bias.to(device=self.device, dtype=self.lower.dtype)

        w_pos = torch.clamp(weight, min=0.0)
        w_neg = torch.clamp(weight, max=0.0)

        # Support both 1D (features,) and 2D (batch, features) intervals
        l_in = self.lower if self.lower.dim() > 1 else self.lower.unsqueeze(0)
        u_in = self.upper if self.upper.dim() > 1 else self.upper.unsqueeze(0)

        # Perform matrix multiplications: (batch, in_f) @ (in_f, out_f)
        l_out = torch.matmul(l_in, w_pos.t()) + torch.matmul(u_in, w_neg.t()) + bias
        u_out = torch.matmul(u_in, w_pos.t()) + torch.matmul(l_in, w_neg.t()) + bias

        if self.lower.dim() == 1:
            l_out = l_out.squeeze(0)
            u_out = u_out.squeeze(0)

        return Interval(l_out, u_out)

    def relu(self) -> "Interval":
        """
        Sound abstract transformer for the ReLU non-linear activation.

        Because ReLU(x) = max(0, x) is monotonically non-decreasing:
            l_out = max(0, l)
            u_out = max(0, u)

        Returns:
            New Interval enclosing the post-activation output space.
        """
        zero = torch.tensor(0.0, device=self.device, dtype=self.lower.dtype)
        return Interval(torch.maximum(self.lower, zero), torch.maximum(self.upper, zero))

    def intersection(self, other: "Interval") -> "Interval":
        """
        Compute the meet (intersection) of two intervals: [max(l1, l2), min(u1, u2)].

        Args:
            other: Interval to intersect with.

        Returns:
            Intersected Interval.
        """
        new_lower = torch.maximum(self.lower, other.lower.to(self.device))
        new_upper = torch.minimum(self.upper, other.upper.to(self.device))
        return Interval(new_lower, new_upper)

    def hull(self, other: "Interval") -> "Interval":
        """
        Compute the interval hull (join) of two intervals: [min(l1, l2), max(u1, u2)].

        Args:
            other: Interval to join with.

        Returns:
            Joined Interval enclosing both sets.
        """
        new_lower = torch.minimum(self.lower, other.lower.to(self.device))
        new_upper = torch.maximum(self.upper, other.upper.to(self.device))
        return Interval(new_lower, new_upper)

    def split(self, dim: int) -> Tuple["Interval", "Interval"]:
        """
        Bisect this interval into two sub-intervals along a given dimension.
        Fundamental operation for Branch-and-Bound (BaB) refinement.

        Args:
            dim: Dimension index along which to split.

        Returns:
            Tuple of (left_child_interval, right_child_interval).
        """
        mid = (self.lower[dim] + self.upper[dim]) / 2.0

        upper_left = self.upper.clone()
        upper_left[dim] = mid
        left_interval = Interval(self.lower.clone(), upper_left)

        lower_right = self.lower.clone()
        lower_right[dim] = mid
        right_interval = Interval(lower_right, self.upper.clone())

        return left_interval, right_interval

    # Arithmetic Operator Overloads
    def __add__(self, other: Union["Interval", torch.Tensor, float]) -> "Interval":
        if isinstance(other, Interval):
            return Interval(self.lower + other.lower, self.upper + other.upper)
        if not isinstance(other, torch.Tensor):
            other = torch.tensor(other, device=self.device, dtype=self.lower.dtype)
        return Interval(self.lower + other, self.upper + other)

    def __radd__(self, other: Union["Interval", torch.Tensor, float]) -> "Interval":
        return self.__add__(other)

    def __sub__(self, other: Union["Interval", torch.Tensor, float]) -> "Interval":
        if isinstance(other, Interval):
            # [l1, u1] - [l2, u2] = [l1 - u2, u1 - l2]
            return Interval(self.lower - other.upper, self.upper - other.lower)
        if not isinstance(other, torch.Tensor):
            other = torch.tensor(other, device=self.device, dtype=self.lower.dtype)
        return Interval(self.lower - other, self.upper - other)

    def __mul__(self, scalar: Union[float, int, torch.Tensor]) -> "Interval":
        """Scalar multiplication: c * [l, u]."""
        if isinstance(scalar, (float, int)):
            scalar = torch.tensor(scalar, device=self.device, dtype=self.lower.dtype)

        cand1 = self.lower * scalar
        cand2 = self.upper * scalar
        return Interval(torch.minimum(cand1, cand2), torch.maximum(cand1, cand2))

    def __rmul__(self, scalar: Union[float, int, torch.Tensor]) -> "Interval":
        return self.__mul__(scalar)

    def __neg__(self) -> "Interval":
        """Negation: -[l, u] = [-u, -l]."""
        return Interval(-self.upper, -self.lower)

    def __repr__(self) -> str:
        return f"Interval(lower={self.lower.tolist()}, upper={self.upper.tolist()})"

