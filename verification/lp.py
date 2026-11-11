"""LP constraint generation for sound neuron bound refinement.

This module implements the constraint-generation layer of the Phase 3 LP
refinement pipeline. It converts an extracted float64
:class:`~models.model_loader.MathematicalNetwork`, a bounded input region, and
a baseline Interval Bound Propagation (IBP) trace into a Linear Program in the
matrix form expected by ``scipy.optimize.linprog``::

    minimize    objective @ xi
    subject to  A_ub @ xi <= b_ub
                A_eq @ xi == b_eq
                variable_bounds[k][0] <= xi[k] <= variable_bounds[k][1]

The resulting feasible set is a sound outer approximation of the network's
concrete executions over the input region:

- Linear layers are encoded by their exact affine equations ``z = W x + b``.
- ReLU layers with stable IBP pre-activation bounds (all negative or all
  positive) are encoded exactly (``y = 0`` or ``y = z``).
- ReLU neurons whose IBP pre-activation bounds cross zero are relaxed with the
  convex Planet/triangle relaxation::

      y >= 0,   y >= z,   y <= (u / (u - l)) * (z - l)

  where ``[l, u]`` is the IBP pre-activation interval. Because ReLU is convex
  and its graph lies below the chord through ``(l, 0)`` and ``(u, u)``, every
  concrete execution satisfies the relaxation; the LP therefore never excludes
  a concrete execution.
- Baseline IBP boxes are imposed as variable bounds. IBP is a sound abstract
  transformer, so these bounds are satisfied by every concrete execution and
  only tighten the LP without endangering soundness.

Soundness also requires that the supplied IBP trace was produced for a region
containing the queried input region: a trace from a different region could
classify a ReLU neuron as stable when it is not, which would cut off concrete
executions. :func:`build_lp_problem` validates this containment and rejects
mismatched traces.

This module builds and validates problems only; it does not solve them. Solver
integration (``solve_lp``) and the full refinement algorithm
(``refine_neuron_bounds``) are separate deliverables.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from abstract_domain.bounds import ALLOWED_PHASES, IntermediateBounds, NeuronRef
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer
from verification.regions import BoxProtocol

FloatArray: TypeAlias = NDArray[np.float64]

# Numerical tolerance for the trace-containment guard. It is small enough that
# meaningfully mismatched traces are still rejected rather than silently
# accepted or clipped.
_TOLERANCE = 1e-12


def _positive_int(value: object, name: str) -> int:
    """Coerce a positive integer, rejecting booleans, non-integers and zero."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _index(value: object, name: str, size: int) -> int:
    """Coerce an integer index into the range [0, size)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if not 0 <= value < size:
        raise ValueError(f"{name} must be in range [0, {size}), got {value}")
    return value


def _finite_scalar(value: object, name: str) -> float:
    """Coerce a finite real scalar, rejecting NaN and infinities."""
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, got {type(value).__name__}")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar}")
    return scalar


def _coerce_finite_vector(value: object, name: str) -> FloatArray:
    """Coerce a finite 1-D float64 vector to an independent read-only array."""
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"{name} must be convertible to a float64 array: {exc}") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got shape {array.shape}")
    if np.any(np.isnan(array)) or np.any(np.isinf(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.flags.writeable = False
    return array


def _coerce_bound_vector(value: object, name: str, length: int) -> FloatArray:
    """Coerce a NaN-free 1-D bound vector of the required length.

    Infinite entries are allowed and interpreted as unbounded sides; the
    caller maps them to ``None`` variable bounds.
    """
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"{name} must be convertible to a float64 array: {exc}") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got shape {array.shape}")
    if array.shape[0] != length:
        raise ValueError(f"{name} must have {length} entries, got {array.shape[0]}")
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} cannot contain NaN")
    return array


def _coerce_int_tuple(value: object, name: str) -> tuple[int, ...]:
    """Coerce a tuple of nonnegative integers."""
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple, got {type(value).__name__}")
    entries: list[int] = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise TypeError(f"{name} entries must be integers, got {type(entry).__name__}")
        if entry < 0:
            raise ValueError(f"{name} entries must be nonnegative, got {entry}")
        entries.append(entry)
    return tuple(entries)


def _coerce_optional_int_tuple(value: object, name: str) -> tuple[int | None, ...]:
    """Coerce a tuple whose entries are nonnegative integers or None."""
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple, got {type(value).__name__}")
    entries: list[int | None] = []
    for entry in value:
        if entry is None:
            entries.append(None)
            continue
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise TypeError(
                f"{name} entries must be integers or None, got {type(entry).__name__}"
            )
        if entry < 0:
            raise ValueError(f"{name} entries must be nonnegative, got {entry}")
        entries.append(entry)
    return tuple(entries)


@dataclass(frozen=True)
class VariableLayout:
    """Offsets of the variable groups inside the LP column vector.

    The LP variable vector concatenates, in order: the network inputs, then
    for every sequence layer index the pre-activation group (``"pre"``; the
    affine output of a Linear layer or the ReLU argument), followed, for ReLU
    layers only, by the post-activation group (``"post"``). All groups are
    contiguous; ``pre_offsets[i]`` therefore always equals the running extent
    of every previous group and ``post_offsets[i]`` is ``None`` exactly for
    linear layers.

    Args:
        input_dim: Number of network input variables.
        pre_offsets: Start offset of the pre-activation group per layer index.
        pre_dims: Number of pre-activation variables per layer index.
        post_offsets: Start offset of the post-activation group per layer
            index; ``None`` for linear layers.
        post_dims: Number of post-activation variables per layer index;
            ``None`` for linear layers.
        total_variables: Total number of LP variables.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If tuples disagree in length, a dimension is zero, an
            offset breaks group contiguity, or the final extent does not equal
            ``total_variables``.
    """

    input_dim: int
    pre_offsets: tuple[int, ...]
    pre_dims: tuple[int, ...]
    post_offsets: tuple[int | None, ...]
    post_dims: tuple[int | None, ...]
    total_variables: int

    def __post_init__(self) -> None:
        input_dim = _positive_int(self.input_dim, "input_dim")
        pre_offsets = _coerce_int_tuple(self.pre_offsets, "pre_offsets")
        pre_dims = _coerce_int_tuple(self.pre_dims, "pre_dims")
        post_offsets = _coerce_optional_int_tuple(self.post_offsets, "post_offsets")
        post_dims = _coerce_optional_int_tuple(self.post_dims, "post_dims")
        total_variables = _positive_int(self.total_variables, "total_variables")

        layer_count = len(pre_offsets)
        if not all(
            len(entries) == layer_count
            for entries in (pre_dims, post_offsets, post_dims)
        ):
            raise ValueError(
                "pre_offsets, pre_dims, post_offsets and post_dims must have "
                "equal lengths"
            )
        if any(dim == 0 for dim in pre_dims):
            raise ValueError("pre_dims entries must be positive")
        if any(dim == 0 for dim in post_dims if dim is not None):
            raise ValueError("post_dims entries must be positive")

        cursor = input_dim
        for index in range(layer_count):
            if (post_offsets[index] is None) != (post_dims[index] is None):
                raise ValueError(
                    f"layer {index}: post_offsets and post_dims must both be "
                    "set for ReLU layers and both None for linear layers"
                )
            if pre_offsets[index] != cursor:
                raise ValueError(
                    f"layer {index}: expected pre_offsets {cursor} for a "
                    f"contiguous layout, got {pre_offsets[index]}"
                )
            cursor += pre_dims[index]
            if post_offsets[index] is not None:
                if post_offsets[index] != cursor:
                    raise ValueError(
                        f"layer {index}: expected post_offsets {cursor} for a "
                        f"contiguous layout, got {post_offsets[index]}"
                    )
                cursor += post_dims[index]
        if cursor != total_variables:
            raise ValueError(
                f"variable groups must end exactly at total_variables "
                f"{total_variables}, got {cursor}"
            )

        object.__setattr__(self, "input_dim", input_dim)
        object.__setattr__(self, "pre_offsets", pre_offsets)
        object.__setattr__(self, "pre_dims", pre_dims)
        object.__setattr__(self, "post_offsets", post_offsets)
        object.__setattr__(self, "post_dims", post_dims)
        object.__setattr__(self, "total_variables", total_variables)

    def variable_slice(self, layer_index: int, phase: str) -> slice:
        """Return the column slice selecting one layer-phase variable group.

        Args:
            layer_index: Zero-based sequence index of the network layer.
            phase: ``"pre"`` or ``"post"``.

        Raises:
            TypeError: If ``layer_index`` is not an integer or ``phase`` is
                not a string.
            ValueError: If ``layer_index`` is out of range, ``phase`` is not
                allowed, or ``"post"`` is requested for a linear layer.
        """
        index = _index(layer_index, "layer_index", len(self.pre_offsets))
        if not isinstance(phase, str):
            raise TypeError(f"phase must be a string, got {type(phase).__name__}")
        if phase not in ALLOWED_PHASES:
            raise ValueError(f"phase must be one of {ALLOWED_PHASES}, got {phase!r}")
        if phase == "pre":
            start = self.pre_offsets[index]
            return slice(start, start + self.pre_dims[index])
        post_offset = self.post_offsets[index]
        if post_offset is None:
            raise ValueError(
                f"layer {index} has no post-activation variables; only ReLU "
                "layers support phase 'post'"
            )
        return slice(post_offset, post_offset + self.post_dims[index])

    def variable_index(self, neuron: NeuronRef) -> int:
        """Return the LP column index selected by a neuron reference.

        The mapping follows the shared Phase 3 phase convention: ``"pre"``
        selects the pre-activation variable of the layer (affine output for a
        Linear layer, ReLU argument for a ReLU layer) and ``"post"`` selects
        the post-activation variable of a ReLU layer.

        Raises:
            TypeError: If ``neuron`` is not a NeuronRef.
            ValueError: If the layer index, phase, or neuron index is invalid
                for this layout.
        """
        if not isinstance(neuron, NeuronRef):
            raise TypeError(f"neuron must be a NeuronRef, got {type(neuron).__name__}")
        index = _index(neuron.layer_index, "layer_index", len(self.pre_offsets))
        if neuron.phase == "post":
            post_offset = self.post_offsets[index]
            if post_offset is None:
                raise ValueError(
                    f"layer {index} has no post-activation variables; only "
                    "ReLU layers support phase 'post'"
                )
            offset, dimension = post_offset, self.post_dims[index]
        else:
            offset, dimension = self.pre_offsets[index], self.pre_dims[index]
        neuron_index = _index(neuron.neuron_index, "neuron_index", dimension)
        return offset + neuron_index

    def objective_vector(self, neuron: NeuronRef) -> FloatArray:
        """Return the unit vector selecting the variable of ``neuron``."""
        vector = np.zeros(self.total_variables, dtype=np.float64)
        vector[self.variable_index(neuron)] = 1.0
        return vector


@dataclass(frozen=True)
class LPProblem:
    """A Linear Program in ``scipy.optimize.linprog`` matrix form.

    The problem minimizes ``objective @ xi`` over variables ``xi`` subject to
    ``A_ub @ xi <= b_ub``, ``A_eq @ xi == b_eq`` and per-variable bounds.
    ``objective`` is always the unit vector selecting ``objective_neuron``;
    maximization is expressed by negating the objective at solve time (the
    feasible set is identical for both directions).

    Args:
        objective: Unit vector of length ``layout.total_variables``.
        objective_neuron: The neuron whose variable the objective selects.
        layout: The variable group offsets the matrices refer to.
        variable_bounds: Per-variable ``(lower, upper)`` pairs; ``None`` marks
            an unbounded side. Interpreted directly as the ``bounds`` argument
            of ``scipy.optimize.linprog``.
        A_ub: Inequality constraint matrix of shape ``(m_ub, n)`` or ``None``.
        b_ub: Inequality right-hand side of shape ``(m_ub,)`` or ``None``.
        A_eq: Equality constraint matrix of shape ``(m_eq, n)`` or ``None``.
        b_eq: Equality right-hand side of shape ``(m_eq,)`` or ``None``.

    Raises:
        TypeError: If a field has the wrong type or cannot be coerced.
        ValueError: If shapes disagree, coefficients are not finite, bounds
            are inverted, or the objective is not the unit vector of
            ``objective_neuron``.
    """

    objective: FloatArray
    objective_neuron: NeuronRef
    layout: VariableLayout
    variable_bounds: tuple[tuple[float | None, float | None], ...]
    A_ub: FloatArray | None = None
    b_ub: FloatArray | None = None
    A_eq: FloatArray | None = None
    b_eq: FloatArray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.objective_neuron, NeuronRef):
            raise TypeError(
                f"objective_neuron must be a NeuronRef, got {type(self.objective_neuron).__name__}"
            )
        layout = self.layout
        if not isinstance(layout, VariableLayout):
            raise TypeError(f"layout must be a VariableLayout, got {type(layout).__name__}")
        n = layout.total_variables

        objective = _coerce_finite_vector(self.objective, "objective")
        if objective.shape[0] != n:
            raise ValueError(f"objective must have length {n}, got {objective.shape[0]}")
        expected = layout.objective_vector(self.objective_neuron)
        if not np.array_equal(objective, expected):
            raise ValueError("objective must be the unit vector selecting objective_neuron")

        variable_bounds = _coerce_variable_bounds(self.variable_bounds, n)
        A_ub, b_ub = _coerce_constraint_pair(self.A_ub, self.b_ub, "A_ub", "b_ub", n)
        A_eq, b_eq = _coerce_constraint_pair(self.A_eq, self.b_eq, "A_eq", "b_eq", n)

        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "variable_bounds", variable_bounds)
        object.__setattr__(self, "A_ub", A_ub)
        object.__setattr__(self, "b_ub", b_ub)
        object.__setattr__(self, "A_eq", A_eq)
        object.__setattr__(self, "b_eq", b_eq)

    @property
    def num_inequality_constraints(self) -> int:
        """Return the number of ``<=`` rows (0 when there are none)."""
        return 0 if self.A_ub is None else int(self.A_ub.shape[0])

    @property
    def num_equality_constraints(self) -> int:
        """Return the number of equality rows (0 when there are none)."""
        return 0 if self.A_eq is None else int(self.A_eq.shape[0])


def _coerce_variable_bounds(
    bounds: object, n: int
) -> tuple[tuple[float | None, float | None], ...]:
    """Coerce and validate a linprog-style variable bounds tuple."""
    if not isinstance(bounds, tuple):
        raise TypeError(f"variable_bounds must be a tuple, got {type(bounds).__name__}")
    if len(bounds) != n:
        raise ValueError(f"variable_bounds must have {n} entries, got {len(bounds)}")
    cleaned: list[tuple[float | None, float | None]] = []
    for index, entry in enumerate(bounds):
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(
                f"variable_bounds[{index}] must be a (lower, upper) pair, got {entry!r}"
            )
        lower, upper = entry
        low = None if lower is None else _finite_scalar(lower, f"variable_bounds[{index}] lower")
        high = None if upper is None else _finite_scalar(upper, f"variable_bounds[{index}] upper")
        if low is not None and high is not None and low > high:
            raise ValueError(
                f"variable_bounds[{index}] lower {low} must not exceed upper {high}"
            )
        cleaned.append((low, high))
    return tuple(cleaned)


def _coerce_constraint_pair(
    A: object,
    b: object,
    a_name: str,
    b_name: str,
    n: int,
) -> tuple[FloatArray | None, FloatArray | None]:
    """Coerce one constraint matrix/rhs pair to independent read-only arrays."""
    if A is None and b is None:
        return None, None
    if A is None or b is None:
        raise ValueError(f"{a_name} and {b_name} must both be provided or both be None")
    try:
        matrix = np.array(A, dtype=np.float64, copy=True)
        rhs = np.array(b, dtype=np.float64, copy=True)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"{a_name} and {b_name} must be convertible to float64 arrays: {exc}"
        ) from exc
    if matrix.ndim != 2:
        raise ValueError(f"{a_name} must be 2-D, got shape {matrix.shape}")
    if matrix.shape[1] != n:
        raise ValueError(f"{a_name} must have {n} columns, got {matrix.shape[1]}")
    if matrix.shape[0] == 0:
        raise ValueError(
            f"{a_name} must contain at least one row; pass None for an empty constraint set"
        )
    if rhs.ndim != 1:
        raise ValueError(f"{b_name} must be 1-D, got shape {rhs.shape}")
    if rhs.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"{b_name} must have {matrix.shape[0]} entries, got {rhs.shape[0]}"
        )
    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        raise ValueError(f"{a_name} coefficients must be finite")
    if np.any(np.isnan(rhs)) or np.any(np.isinf(rhs)):
        raise ValueError(f"{b_name} must be finite")
    matrix.flags.writeable = False
    rhs.flags.writeable = False
    return matrix, rhs


class LPConstraintBuilder:
    """Accumulates linear constraints for one LP over a fixed variable vector.

    The builder stores sparse rows internally and materializes dense float64
    matrices in the :class:`LPProblem` returned by :meth:`build`. All rows are
    written directly in the ``linprog`` orientation: equalities as
    ``coefficients @ xi == rhs`` and inequalities as ``coefficients @ xi <= rhs``.

    Args:
        n_variables: Total number of LP variables (columns).

    Raises:
        TypeError: If ``n_variables`` is not an integer.
        ValueError: If ``n_variables`` is not positive.
    """

    def __init__(self, n_variables: int) -> None:
        self._n_variables = _positive_int(n_variables, "n_variables")
        self._bounds: list[list[float | None]] = [
            [None, None] for _ in range(self._n_variables)
        ]
        self._ub_rows: list[tuple[dict[int, float], float]] = []
        self._eq_rows: list[tuple[dict[int, float], float]] = []

    def _slice_range(self, variables: slice, name: str) -> range:
        """Validate a slice and return its index range within the variables."""
        if not isinstance(variables, slice):
            raise TypeError(f"{name} must be a slice, got {type(variables).__name__}")
        if variables.step not in (None, 1):
            raise ValueError(f"{name} must use step 1 or no step, got step {variables.step}")
        start, stop = variables.start, variables.stop
        if start is None or stop is None:
            raise ValueError(f"{name} must specify both start and stop")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(stop, bool)
            or not isinstance(stop, int)
        ):
            raise TypeError(f"{name} start and stop must be integers")
        if not 0 <= start < stop <= self._n_variables:
            raise ValueError(
                f"{name} range {start}:{stop} is out of bounds for "
                f"{self._n_variables} variables"
            )
        return range(start, stop)

    def add_box_bounds(self, variables: slice, lower: Any, upper: Any) -> None:
        """Assign per-variable bounds for the selected column range.

        Infinite sides are mapped to ``None`` (unbounded), matching
        ``scipy.optimize.linprog`` bounds semantics.

        Args:
            variables: Column slice selecting the variables to bound.
            lower: 1-D lower bound vector with one entry per variable.
            upper: 1-D upper bound vector with one entry per variable.

        Raises:
            TypeError: If ``variables`` is not a slice or the vectors cannot
                be coerced to float64.
            ValueError: If the slice or vector lengths are invalid, bounds
                contain NaN or are inverted, or a variable was already bound.
        """
        indices = self._slice_range(variables, "variables")
        lower_arr = _coerce_bound_vector(lower, "lower", len(indices))
        upper_arr = _coerce_bound_vector(upper, "upper", len(indices))
        for position, index in enumerate(indices):
            low = lower_arr[position]
            high = upper_arr[position]
            if low == np.inf or high == -np.inf:
                raise ValueError("infinite bounds must be -inf below or +inf above")
            if low > high:
                raise ValueError(f"lower {low} must not exceed upper {high} for variable {index}")
            if self._bounds[index] != [None, None]:
                raise ValueError(
                    f"variable {index} bounds were already assigned; "
                    "overlapping box constraints are not allowed"
                )
            self._bounds[index] = [
                None if low == -np.inf else float(low),
                None if high == np.inf else float(high),
            ]

    def add_linear_layer(
        self,
        weight: Any,
        bias: Any,
        input_vars: slice,
        output_vars: slice,
    ) -> None:
        """Add the exact affine equalities ``z = W a + b`` of a Linear layer.

        One equality row ``z_j - sum_k W[j, k] a_k == b[j]`` is added per
        output neuron.

        Args:
            weight: 2-D weight matrix of shape (outputs, inputs).
            bias: 1-D bias vector of shape (outputs,).
            input_vars: Column slice selecting the incoming variables ``a``.
            output_vars: Column slice selecting the output variables ``z``.

        Raises:
            TypeError: If the weight or bias cannot be coerced to float64.
            ValueError: If shapes disagree with the selected slices or the
                weight or bias contains NaN or infinite values.
        """
        input_range = self._slice_range(input_vars, "input_vars")
        output_range = self._slice_range(output_vars, "output_vars")
        try:
            w = np.array(weight, dtype=np.float64, copy=True)
            b = np.array(bias, dtype=np.float64, copy=True)
        except (ValueError, TypeError) as exc:
            raise TypeError(
                f"weight and bias must be convertible to float64 arrays: {exc}"
            ) from exc
        if w.ndim != 2:
            raise ValueError(f"weight must be 2-D, got shape {w.shape}")
        if b.ndim != 1:
            raise ValueError(f"bias must be 1-D, got shape {b.shape}")
        if w.shape != (len(output_range), len(input_range)):
            raise ValueError(
                f"weight shape {w.shape} does not match the selected input "
                f"({len(input_range)}) and output ({len(output_range)}) variables"
            )
        if b.shape[0] != w.shape[0]:
            raise ValueError(
                f"bias dimension {b.shape[0]} does not match weight output dimension {w.shape[0]}"
            )
        if np.any(np.isnan(w)) or np.any(np.isinf(w)) or np.any(np.isnan(b)) or np.any(np.isinf(b)):
            raise ValueError("weight and bias must be finite")

        for row, output_index in enumerate(output_range):
            coefficients = {output_index: 1.0}
            for column, input_index in enumerate(input_range):
                value = w[row, column]
                if value != 0.0:
                    coefficients[input_index] = -value
            self._eq_rows.append((coefficients, float(b[row])))

    def add_relu_layer(
        self,
        input_vars: slice,
        pre_vars: slice,
        post_vars: slice,
        pre_lower: Any,
        pre_upper: Any,
    ) -> None:
        """Add the constraints of a ReLU layer classified by IBP pre-bounds.

        First the exact coupling ``z_j - a_j == 0`` links every pre-activation
        variable to its incoming variable. Then each neuron is classified by
        its IBP pre-activation interval ``[pre_lower[j], pre_upper[j]]``:

        - ``upper <= 0`` (stable inactive): ``y_j == 0`` (exact).
        - ``lower >= 0`` (stable active): ``y_j == z_j`` (exact).
        - ``lower < 0 < upper`` (unstable): the convex triangle relaxation
          ``-y_j <= 0``, ``z_j - y_j <= 0`` and
          ``-u_j z_j + (u_j - l_j) y_j <= -u_j l_j``.

        Args:
            input_vars: Column slice selecting the incoming variables.
            pre_vars: Column slice selecting the pre-activation variables.
            post_vars: Column slice selecting the post-activation variables.
            pre_lower: 1-D IBP pre-activation lower bounds per neuron.
            pre_upper: 1-D IBP pre-activation upper bounds per neuron.

        Raises:
            TypeError: If a slice is invalid or the bound vectors cannot be
                coerced to float64.
            ValueError: If the slices disagree in length, bounds are inverted,
                or an unstable neuron has infinite pre-bounds (no sound
                triangle relaxation exists for them).
        """
        input_range = self._slice_range(input_vars, "input_vars")
        pre_range = self._slice_range(pre_vars, "pre_vars")
        post_range = self._slice_range(post_vars, "post_vars")
        count = len(pre_range)
        if len(input_range) != count or len(post_range) != count:
            raise ValueError(
                f"input_vars, pre_vars and post_vars must select the same "
                f"number of neurons, got {len(input_range)}, {count} and "
                f"{len(post_range)}"
            )
        lower_arr = _coerce_bound_vector(pre_lower, "pre_lower", count)
        upper_arr = _coerce_bound_vector(pre_upper, "pre_upper", count)

        for position in range(count):
            input_index = input_range[position]
            pre_index = pre_range[position]
            post_index = post_range[position]
            low = lower_arr[position]
            high = upper_arr[position]
            if low > high:
                raise ValueError(
                    f"ReLU neuron {position} pre lower {low} must not exceed "
                    f"pre upper {high}"
                )
            self._eq_rows.append(({input_index: -1.0, pre_index: 1.0}, 0.0))
            if high <= 0.0:
                self._eq_rows.append(({post_index: 1.0}, 0.0))
            elif low >= 0.0:
                self._eq_rows.append(({pre_index: -1.0, post_index: 1.0}, 0.0))
            else:
                if not (np.isfinite(low) and np.isfinite(high)):
                    raise ValueError(
                        f"ReLU neuron {position} has unstable pre bounds "
                        f"[{low}, {high}]; a sound triangle relaxation "
                        "requires finite bounds"
                    )
                self._ub_rows.append(({post_index: -1.0}, 0.0))
                self._ub_rows.append(({pre_index: 1.0, post_index: -1.0}, 0.0))
                self._ub_rows.append(
                    ({pre_index: -high, post_index: high - low}, -high * low)
                )

    def build(
        self,
        objective: Any,
        objective_neuron: NeuronRef,
        layout: VariableLayout,
    ) -> LPProblem:
        """Materialize the accumulated constraints into an LPProblem.

        Args:
            objective: Unit vector of length ``n_variables`` selecting
                ``objective_neuron``.
            objective_neuron: The neuron the objective selects.
            layout: The variable layout matching the builder's column count.

        Raises:
            TypeError: If ``objective_neuron`` or ``layout`` has the wrong type.
            ValueError: If the layout column count differs from the builder's
                column count, or the objective is invalid for the layout.
        """
        if not isinstance(objective_neuron, NeuronRef):
            raise TypeError(
                f"objective_neuron must be a NeuronRef, got {type(objective_neuron).__name__}"
            )
        if not isinstance(layout, VariableLayout):
            raise TypeError(f"layout must be a VariableLayout, got {type(layout).__name__}")
        if layout.total_variables != self._n_variables:
            raise ValueError(
                f"layout has {layout.total_variables} variables but the "
                f"builder was created for {self._n_variables}"
            )

        A_ub = b_ub = A_eq = b_eq = None
        if self._ub_rows:
            A_ub = np.zeros((len(self._ub_rows), self._n_variables), dtype=np.float64)
            b_ub = np.zeros(len(self._ub_rows), dtype=np.float64)
            for row, (coefficients, rhs) in enumerate(self._ub_rows):
                for index, value in coefficients.items():
                    A_ub[row, index] = value
                b_ub[row] = rhs
        if self._eq_rows:
            A_eq = np.zeros((len(self._eq_rows), self._n_variables), dtype=np.float64)
            b_eq = np.zeros(len(self._eq_rows), dtype=np.float64)
            for row, (coefficients, rhs) in enumerate(self._eq_rows):
                for index, value in coefficients.items():
                    A_eq[row, index] = value
                b_eq[row] = rhs

        return LPProblem(
            objective=objective,
            objective_neuron=objective_neuron,
            layout=layout,
            variable_bounds=tuple((low, high) for low, high in self._bounds),
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
        )


def _coerce_input_bounds(
    network: MathematicalNetwork, input_region: BoxProtocol
) -> tuple[FloatArray, FloatArray]:
    """Validate the network and input region, returning float64 bound vectors."""
    if not isinstance(network, MathematicalNetwork):
        raise TypeError(f"network must be a MathematicalNetwork, got {type(network).__name__}")

    if (
        not isinstance(input_region, BoxProtocol)
        or callable(input_region.lower)
        or callable(input_region.upper)
    ):
        raise TypeError(
            f"input_region must satisfy BoxProtocol, got {type(input_region).__name__}"
        )
    try:
        lower = np.asarray(input_region.lower, dtype=np.float64)
        upper = np.asarray(input_region.upper, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"input_region bounds must be convertible to float64 arrays: {exc}") from exc

    if lower.ndim != 1 or upper.ndim != 1:
        raise ValueError(
            f"input_region bounds must be 1-D vectors, got lower.ndim={lower.ndim}, "
            f"upper.ndim={upper.ndim}"
        )
    if lower.shape != upper.shape:
        raise ValueError(
            f"input_region bound shape mismatch: lower shape {lower.shape} != "
            f"upper shape {upper.shape}"
        )
    if lower.size == 0:
        raise ValueError("input_region bounds must not be empty")
    if lower.shape[0] != network.input_features:
        raise ValueError(
            f"input_region dimension {lower.shape[0]} does not match "
            f"network.input_features {network.input_features}"
        )
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("input_region bounds cannot contain NaN")
    if np.any(lower > upper):
        raise ValueError("input_region lower bound cannot exceed upper bound")
    return lower, upper


def _validate_trace(
    network: MathematicalNetwork,
    trace: IntermediateBounds,
    region_lower: FloatArray,
    region_upper: FloatArray,
) -> None:
    """Validate that the IBP trace matches the network and covers the region.

    The trace is the sole source of the ReLU stability classification and the
    relaxation parameters, so a mismatched trace (different layers, different
    dimensions, or a different input region) would silently produce an unsound
    LP. This function rejects every such mismatch.

    Raises:
        TypeError: If ``trace`` is not an IntermediateBounds or the network
            contains an unsupported layer type.
        ValueError: If the trace disagrees with the network structure or its
            input box does not contain the queried input region.
    """
    if not isinstance(trace, IntermediateBounds):
        raise TypeError(
            f"intermediate_bounds must be an IntermediateBounds, got {type(trace).__name__}"
        )
    layers = network.layers
    entries = trace.layer_bounds
    if len(entries) != len(layers):
        raise ValueError(
            f"intermediate_bounds has {len(entries)} layer entries but the "
            f"network has {len(layers)} layers"
        )
    if trace.input_bounds.dimension != network.input_features:
        raise ValueError(
            f"intermediate_bounds input dimension {trace.input_bounds.dimension} "
            f"does not match network.input_features {network.input_features}"
        )
    if trace.output_bounds.dimension != network.output_features:
        raise ValueError(
            f"intermediate_bounds output dimension {trace.output_bounds.dimension} "
            f"does not match network.output_features {network.output_features}"
        )

    if not (
        np.all(region_lower >= trace.input_bounds.lower - _TOLERANCE)
        and np.all(region_upper <= trace.input_bounds.upper + _TOLERANCE)
    ):
        raise ValueError(
            "intermediate_bounds.input_bounds must contain the input region "
            "bounds; a trace computed from a different input region would "
            "invalidate the ReLU relaxation"
        )

    for index, layer in enumerate(layers):
        entry = entries[index]
        if entry.layer_index != index:
            raise ValueError(
                f"intermediate_bounds entry at position {index} has layer_index "
                f"{entry.layer_index}"
            )
        if isinstance(layer, LinearLayer):
            if entry.layer_type != "linear":
                raise ValueError(
                    f"intermediate_bounds entry {index} has layer_type "
                    f"{entry.layer_type!r} but network layer {index} is a LinearLayer"
                )
            if entry.pre_activation_bounds is None or entry.post_activation_bounds is not None:
                raise ValueError(
                    f"intermediate_bounds entry {index} must carry exactly a "
                    "pre-activation box for a linear layer"
                )
            if entry.pre_activation_bounds.dimension != layer.output_features:
                raise ValueError(
                    f"intermediate_bounds entry {index} pre-activation dimension "
                    f"{entry.pre_activation_bounds.dimension} does not match "
                    f"layer output features {layer.output_features}"
                )
            if entry.input_bounds.dimension != layer.input_features:
                raise ValueError(
                    f"intermediate_bounds entry {index} input dimension "
                    f"{entry.input_bounds.dimension} does not match layer "
                    f"input features {layer.input_features}"
                )
        elif isinstance(layer, ReLULayer):
            if entry.layer_type != "relu":
                raise ValueError(
                    f"intermediate_bounds entry {index} has layer_type "
                    f"{entry.layer_type!r} but network layer {index} is a ReLULayer"
                )
            pre_box = entry.pre_activation_bounds
            post_box = entry.post_activation_bounds
            if pre_box is None or post_box is None:
                raise ValueError(
                    f"intermediate_bounds entry {index} must carry pre- and "
                    "post-activation boxes for a ReLU layer"
                )
            if (
                entry.input_bounds.dimension != pre_box.dimension
                or pre_box.dimension != post_box.dimension
            ):
                raise ValueError(
                    f"intermediate_bounds entry {index} ReLU input, pre and "
                    "post dimensions must agree"
                )
        else:
            layer_name = type(layer).__name__
            position = getattr(layer, "position", index)
            raise TypeError(
                f"Unsupported layer type '{layer_name}' at sequence position "
                f"{position}; only LinearLayer and ReLULayer are supported"
            )


def _validate_target_neuron(
    network: MathematicalNetwork,
    trace: IntermediateBounds,
    target_neuron: NeuronRef,
) -> None:
    """Validate that the target neuron is real and phase-compatible.

    Mirrors the shared Phase 3 convention: linear layers only expose the
    ``"pre"`` phase, ReLU layers expose ``"pre"`` and ``"post"``.

    Raises:
        TypeError: If ``target_neuron`` is not a NeuronRef.
        ValueError: If the layer index, phase, or neuron index is invalid.
    """
    if not isinstance(target_neuron, NeuronRef):
        raise TypeError(
            f"target_neuron must be a NeuronRef, got {type(target_neuron).__name__}"
        )
    layer_index = target_neuron.layer_index
    if not 0 <= layer_index < len(network.layers):
        raise ValueError(
            f"layer_index {layer_index} is out of range for a network with "
            f"{len(network.layers)} layers"
        )
    layer = network.layers[layer_index]
    entry = trace.layer_bounds[layer_index]
    if isinstance(layer, LinearLayer):
        if target_neuron.phase != "pre":
            raise ValueError(
                "linear layers only support 'pre' phase objectives; got phase "
                f"{target_neuron.phase!r} for layer {layer_index}"
            )
        dimension = layer.output_features
    else:
        reference_box = (
            entry.pre_activation_bounds
            if target_neuron.phase == "pre"
            else entry.post_activation_bounds
        )
        if reference_box is None:
            raise ValueError(
                f"layer {layer_index} has no {target_neuron.phase!r} "
                "activation variables to target"
            )
        dimension = reference_box.dimension
    if not 0 <= target_neuron.neuron_index < dimension:
        raise ValueError(
            f"neuron_index {target_neuron.neuron_index} is out of range for "
            f"layer {layer_index} with {dimension} neurons"
        )


def build_lp_problem(
    network: MathematicalNetwork,
    input_region: BoxProtocol,
    intermediate_bounds: IntermediateBounds,
    target_neuron: NeuronRef,
) -> LPProblem:
    """Build the LP encoding a sound outer approximation of the network.

    The LP's feasible set contains every concrete network execution over
    ``input_region`` (see the module docstring for the soundness argument), so
    minimizing and maximizing the target neuron's variable over it yields
    conservative outer bounds. The returned problem uses the matrix layout
    expected by ``scipy.optimize.linprog``; solving it is outside this
    function's scope.

    Args:
        network: An extracted float64 MathematicalNetwork with Linear and
            ReLU layers in execution order.
        input_region: The input perturbation region (an InputRegion or Box;
            anything conforming to BoxProtocol).
        intermediate_bounds: The baseline IBP trace produced by
            ``get_intermediate_bounds`` for a region containing
            ``input_region``.
        target_neuron: The neuron whose variable becomes the LP objective.

    Returns:
        An LPProblem whose objective is the unit vector selecting the target
        neuron's variable.

    Raises:
        TypeError: If an argument has the wrong type or the network contains
            an unsupported layer type.
        ValueError: If dimensions disagree, the trace does not match the
            network, the trace input box does not contain the input region,
            the target neuron is invalid, or an unstable ReLU neuron has
            infinite pre-activation bounds.
    """
    region_lower, region_upper = _coerce_input_bounds(network, input_region)
    _validate_trace(network, intermediate_bounds, region_lower, region_upper)
    _validate_target_neuron(network, intermediate_bounds, target_neuron)

    layers = network.layers
    n_input = network.input_features
    pre_offsets: list[int] = []
    pre_dims: list[int] = []
    post_offsets: list[int | None] = []
    post_dims: list[int | None] = []
    cursor = n_input
    for index, layer in enumerate(layers):
        entry = intermediate_bounds.layer_bounds[index]
        dimension = (
            layer.output_features
            if isinstance(layer, LinearLayer)
            else entry.pre_activation_bounds.dimension
        )
        pre_offsets.append(cursor)
        pre_dims.append(dimension)
        cursor += dimension
        if isinstance(layer, ReLULayer):
            post_offsets.append(cursor)
            post_dims.append(dimension)
            cursor += dimension
        else:
            post_offsets.append(None)
            post_dims.append(None)

    layout = VariableLayout(
        input_dim=n_input,
        pre_offsets=tuple(pre_offsets),
        pre_dims=tuple(pre_dims),
        post_offsets=tuple(post_offsets),
        post_dims=tuple(post_dims),
        total_variables=cursor,
    )
    builder = LPConstraintBuilder(cursor)
    builder.add_box_bounds(slice(0, n_input), region_lower, region_upper)

    for index, layer in enumerate(layers):
        entry = intermediate_bounds.layer_bounds[index]
        pre_slice = layout.variable_slice(index, "pre")
        if index == 0:
            incoming_slice = slice(0, n_input)
        elif isinstance(layers[index - 1], ReLULayer):
            incoming_slice = layout.variable_slice(index - 1, "post")
        else:
            incoming_slice = layout.variable_slice(index - 1, "pre")

        if isinstance(layer, LinearLayer):
            builder.add_linear_layer(layer.weight, layer.bias, incoming_slice, pre_slice)
            pre_box = entry.pre_activation_bounds
            builder.add_box_bounds(pre_slice, pre_box.lower, pre_box.upper)
        else:
            post_slice = layout.variable_slice(index, "post")
            pre_box = entry.pre_activation_bounds
            post_box = entry.post_activation_bounds
            builder.add_relu_layer(
                incoming_slice, pre_slice, post_slice, pre_box.lower, pre_box.upper
            )
            builder.add_box_bounds(pre_slice, pre_box.lower, pre_box.upper)
            builder.add_box_bounds(post_slice, post_box.lower, post_box.upper)

    objective = layout.objective_vector(target_neuron)
    return builder.build(objective, target_neuron, layout)


__all__ = ["VariableLayout", "LPProblem", "LPConstraintBuilder", "build_lp_problem"]
