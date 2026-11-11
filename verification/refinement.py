"""LP-based neuron bound refinement orchestration.

This module implements the Phase 3 refinement engine that combines the LP
constraint builder (:mod:`verification.lp`) with the SciPy/HiGHS solver
(:mod:`verification.lp_solver`) to tighten IBP intermediate bounds.

Interface
---------
``refine_neuron_bounds(network, input_region, intermediate_bounds,
target_neurons) -> RefinementResult``

For every requested neuron the engine builds one shared LP relaxation of the
network (the constraint set is objective-independent) and solves two LPs per
neuron: the minimization and the maximization of that neuron's variable. A
refined interval is accepted only when BOTH directions return an optimal,
numerically validated result (:attr:`LPSolverResult.is_certified`), the
certified bounds stay inside the IBP envelope up to the documented tolerance,
and the certified lower bound does not exceed the certified upper bound.
Accepted refinements are exposed as
:class:`~abstract_domain.bounds.RefinedBounds` records in
:attr:`RefinementResult.refined_bounds`, ready for Member 1's
``propagate_with_refinements``.

Soundness
---------
The LP feasible set contains every concrete network execution over the input
region (see :mod:`verification.lp`), so the conservative sides of the two
solves -- the dual lower bound of the minimization and the negated dual
objective of the maximization, both exposed through ``certified_bound`` --
are conservative outer bounds whenever the solver's numerical validation
passes. This module only accepts ``is_certified`` results: it never uses a
primal witness, an unsuccessful solve, or a raw IBP value in place of a
certified LP bound, and a neuron whose optimization fails simply keeps its
original IBP interval (it does not appear in ``refined_bounds``).

The validation behind ``is_certified`` is tolerance-based numerical checking
(primal/dual feasibility at a 1e-8 relative tolerance, see
:mod:`verification.lp_solver`), not a rigorous interval-arithmetic
certificate. This module therefore does NOT claim formal soundness: accepted
refinements are numerically validated conservative bounds of the LP
relaxation, which is itself a sound outer approximation of the concrete
network. That limitation is represented explicitly: every per-neuron outcome
carries both solver results, so an unvalidated or failed direction is never
hidden.

Behavior contract
-----------------
- The returned refined interval never widens the IBP interval. An accepted
  refinement must lie inside the IBP envelope up to ``_TOLERANCE`` (1e-12,
  the same tolerance Member 1's containment check uses); certified values
  beyond that tolerance reject the neuron.
- A neuron whose minimization or maximization fails, returns unvalidated
  marginals, or yields crossed certified bounds is rejected with its
  original IBP bounds preserved and the reason recorded.
- Duplicate target references are deduplicated (first occurrence wins) so
  the returned collection can always be consumed by
  ``propagate_with_refinements``, which rejects duplicates. Invalid
  references raise ``ValueError``/``TypeError`` before any LP is solved.
- An empty target list returns an ``EMPTY_TARGETS`` result.

Status convention
-----------------
``RefinementStatus`` is derived from the outcomes and enforced on
construction (constructing a result whose status disagrees with its outcomes
raises ``ValueError``):

- ``TIGHTENED``: every target accepted and at least one strictly tighter
  than IBP on one side (beyond ``_TOLERANCE``);
- ``NO_IMPROVEMENT``: every target accepted but none strictly tighter;
- ``PARTIAL``: some targets accepted and some rejected;
- ``FAILED``: every requested target rejected;
- ``EMPTY_TARGETS``: no targets requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
import time
from typing import Optional

from abstract_domain.bounds import IntermediateBounds, NeuronRef, RefinedBounds
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer
from verification.lp import LPProblem, build_lp_problem
from verification.lp_solver import LPSolverResult, solve_lp
from verification.regions import InputRegion

# Tolerance used for the IBP-envelope containment checks and for classifying a
# refinement as a strict improvement. It mirrors the containment tolerance of
# ``abstract_domain.refinement`` so that every refinement accepted here is
# also accepted by Member 1's ``propagate_with_refinements``.
_TOLERANCE = 1e-12

__all__ = [
    "NeuronRefinementOutcome",
    "RefinementResult",
    "RefinementStatus",
    "refine_neuron_bounds",
]


class RefinementStatus(str, Enum):
    """Aggregate outcome of one ``refine_neuron_bounds`` call.

    The status is always consistent with the per-neuron outcomes; the
    mapping is enforced by :class:`RefinementResult`.
    """

    TIGHTENED = "TIGHTENED"            # all targets accepted, at least one strictly tighter
    NO_IMPROVEMENT = "NO_IMPROVEMENT"  # all targets accepted, none strictly tighter
    PARTIAL = "PARTIAL"                # some targets accepted, some rejected
    FAILED = "FAILED"                  # every requested target rejected
    EMPTY_TARGETS = "EMPTY_TARGETS"    # no targets requested


def _finite_real(value: object, name: str) -> float:
    """Coerce a finite real scalar, rejecting NaN and infinities."""
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, got {type(value).__name__}")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {scalar}")
    return scalar


def _coerce_targets(value: object) -> tuple[NeuronRef, ...]:
    """Coerce the target collection to a tuple of NeuronRef entries."""
    if not isinstance(value, (tuple, list)):
        raise TypeError(
            "target_neurons must be a tuple or list of NeuronRef, "
            f"got {type(value).__name__}"
        )
    items = tuple(value)
    for position, item in enumerate(items):
        if not isinstance(item, NeuronRef):
            raise TypeError(
                f"target_neurons[{position}] must be a NeuronRef, "
                f"got {type(item).__name__}"
            )
    return items


def _deduplicate(targets: tuple[NeuronRef, ...]) -> tuple[NeuronRef, ...]:
    """Drop repeated references, preserving first-occurrence order.

    Duplicate targets would solve the same LP twice and would produce a
    collection that Member 1's ``propagate_with_refinements`` rejects.
    """
    seen: set[tuple[int, str, int]] = set()
    unique: list[NeuronRef] = []
    for neuron in targets:
        key = (neuron.layer_index, neuron.phase, neuron.neuron_index)
        if key not in seen:
            seen.add(key)
            unique.append(neuron)
    return tuple(unique)


def _validate_arguments(
    network: object,
    input_region: object,
    intermediate_bounds: object,
) -> None:
    """Validate the argument types and the region/network dimensions."""
    if not isinstance(network, MathematicalNetwork):
        raise TypeError(
            f"network must be a MathematicalNetwork, got {type(network).__name__}"
        )
    if not isinstance(input_region, InputRegion):
        raise TypeError(
            f"input_region must be an InputRegion, got {type(input_region).__name__}"
        )
    if input_region.dimension != network.input_features:
        raise ValueError(
            f"input_region dimension {input_region.dimension} does not match "
            f"network.input_features {network.input_features}"
        )
    if not isinstance(intermediate_bounds, IntermediateBounds):
        raise TypeError(
            "intermediate_bounds must be an IntermediateBounds, "
            f"got {type(intermediate_bounds).__name__}"
        )


def _validate_target(
    network: MathematicalNetwork,
    trace: IntermediateBounds,
    neuron: NeuronRef,
) -> None:
    """Validate that one target reference addresses a real, supported neuron.

    Replicates the shared Phase 3 phase convention locally (linear layers
    only expose the ``"pre"`` phase, ReLU layers expose both) instead of
    importing private helpers from other modules.
    """
    layer_index = neuron.layer_index
    if not 0 <= layer_index < len(network.layers):
        raise ValueError(
            f"layer_index {layer_index} is out of range for a network with "
            f"{len(network.layers)} layers"
        )
    layer = network.layers[layer_index]
    if isinstance(layer, LinearLayer):
        if neuron.phase != "pre":
            raise ValueError(
                "linear layers only support 'pre' phase targets; got phase "
                f"{neuron.phase!r} for layer {layer_index}"
            )
        dimension = layer.output_features
    elif isinstance(layer, ReLULayer):
        entry = trace.layer_bounds[layer_index]
        reference = (
            entry.pre_activation_bounds
            if neuron.phase == "pre"
            else entry.post_activation_bounds
        )
        if reference is None:
            raise ValueError(
                f"layer {layer_index} has no {neuron.phase!r} activation bounds"
            )
        dimension = reference.dimension
    else:
        layer_name = type(layer).__name__
        position = getattr(layer, "position", layer_index)
        raise TypeError(
            f"Unsupported layer type '{layer_name}' at sequence position "
            f"{position}; only LinearLayer and ReLULayer are supported"
        )
    if not 0 <= neuron.neuron_index < dimension:
        raise ValueError(
            f"neuron_index {neuron.neuron_index} is out of range for layer "
            f"{layer_index} with {dimension} neurons"
        )


def _ibp_interval(
    trace: IntermediateBounds,
    neuron: NeuronRef,
) -> tuple[float, float]:
    """Return the trace's IBP interval for the neuron.

    Mirrors Member 1's containment reference: the ``"pre"`` phase reads
    ``pre_activation_bounds`` and the ``"post"`` phase reads
    ``output_bounds``.
    """
    entry = trace.layer_bounds[neuron.layer_index]
    reference = (
        entry.pre_activation_bounds if neuron.phase == "pre" else entry.output_bounds
    )
    index = neuron.neuron_index
    return float(reference.lower[index]), float(reference.upper[index])


@dataclass(frozen=True)
class NeuronRefinementOutcome:
    """Per-neuron record of one refinement attempt.

    Args:
        neuron: The target reference.
        ibp_lower: The original IBP lower bound (preserved on rejection).
        ibp_upper: The original IBP upper bound (preserved on rejection).
        lower_solve: The minimization solve result.
        upper_solve: The maximization solve result.
        refined: The accepted refined interval, or ``None`` when the neuron
            was rejected (its IBP bounds are then preserved by simply not
            being included in ``RefinementResult.refined_bounds``).
        message: Human-readable account of the outcome.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the IBP interval is invalid, the solve directions are
            swapped, or an accepted refinement is inconsistent with the
            solver results or extends outside the IBP envelope.
    """

    neuron: NeuronRef
    ibp_lower: float
    ibp_upper: float
    lower_solve: LPSolverResult
    upper_solve: LPSolverResult
    refined: Optional[RefinedBounds]
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.neuron, NeuronRef):
            raise TypeError(
                f"neuron must be a NeuronRef, got {type(self.neuron).__name__}"
            )
        ibp_lower = _finite_real(self.ibp_lower, "ibp_lower")
        ibp_upper = _finite_real(self.ibp_upper, "ibp_upper")
        if ibp_lower > ibp_upper:
            raise ValueError(f"ibp_lower {ibp_lower} must not exceed ibp_upper {ibp_upper}")
        if not isinstance(self.lower_solve, LPSolverResult):
            raise TypeError(
                f"lower_solve must be a LPSolverResult, got {type(self.lower_solve).__name__}"
            )
        if self.lower_solve.maximize:
            raise ValueError("lower_solve must be a minimization LPSolverResult")
        if not isinstance(self.upper_solve, LPSolverResult):
            raise TypeError(
                f"upper_solve must be a LPSolverResult, got {type(self.upper_solve).__name__}"
            )
        if not self.upper_solve.maximize:
            raise ValueError("upper_solve must be a maximization LPSolverResult")
        if self.lower_solve.objective_neuron != self.neuron:
            raise ValueError("lower_solve must solve this neuron's objective")
        if self.upper_solve.objective_neuron != self.neuron:
            raise ValueError("upper_solve must solve this neuron's objective")

        refined = self.refined
        if refined is not None:
            if not isinstance(refined, RefinedBounds):
                raise TypeError(
                    f"refined must be a RefinedBounds or None, got {type(refined).__name__}"
                )
            if refined.neuron != self.neuron:
                raise ValueError("refined must target the same neuron as the outcome")
            if not (self.lower_solve.is_certified and self.upper_solve.is_certified):
                raise ValueError(
                    "an accepted refinement requires both certified solves"
                )
            if refined.lower != self.lower_solve.certified_bound:
                raise ValueError(
                    "refined.lower must equal the certified lower bound"
                )
            if refined.upper != self.upper_solve.certified_bound:
                raise ValueError(
                    "refined.upper must equal the certified upper bound"
                )
            if refined.lower < ibp_lower - _TOLERANCE or refined.upper > ibp_upper + _TOLERANCE:
                raise ValueError(
                    "a refinement must lie inside the IBP envelope "
                    f"[{ibp_lower}, {ibp_upper}] up to tolerance {_TOLERANCE}"
                )
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string")

        object.__setattr__(self, "ibp_lower", ibp_lower)
        object.__setattr__(self, "ibp_upper", ibp_upper)

    @property
    def accepted(self) -> bool:
        """Whether a refined interval was produced for this neuron."""
        return self.refined is not None

    @property
    def tightens(self) -> bool:
        """Whether the refinement is strictly tighter than IBP on one side."""
        if self.refined is None:
            return False
        return (
            self.refined.lower > self.ibp_lower + _TOLERANCE
            or self.refined.upper < self.ibp_upper - _TOLERANCE
        )


def _derive_status(outcomes: tuple[NeuronRefinementOutcome, ...]) -> RefinementStatus:
    """Compute the aggregate status the outcome collection implies."""
    if not outcomes:
        return RefinementStatus.EMPTY_TARGETS
    accepted = sum(1 for outcome in outcomes if outcome.accepted)
    if accepted == 0:
        return RefinementStatus.FAILED
    if accepted < len(outcomes):
        return RefinementStatus.PARTIAL
    if any(outcome.tightens for outcome in outcomes):
        return RefinementStatus.TIGHTENED
    return RefinementStatus.NO_IMPROVEMENT


@dataclass(frozen=True)
class RefinementResult:
    """Aggregate result of refining a collection of neurons.

    Args:
        status: Aggregate outcome; must agree with the per-neuron outcomes
            (see the module docstring for the exact mapping).
        target_neurons: The deduplicated, validated targets in request order.
        outcomes: One :class:`NeuronRefinementOutcome` per target, aligned
            one-to-one with ``target_neurons``.
        execution_time_seconds: Total wall-clock time including validation.
        message: Human-readable summary.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If targets are duplicated, outcomes do not align with the
            targets, execution time is invalid, or the stored status
            disagrees with the outcomes.
    """

    status: RefinementStatus
    target_neurons: tuple[NeuronRef, ...]
    outcomes: tuple[NeuronRefinementOutcome, ...]
    execution_time_seconds: float
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, RefinementStatus):
            raise TypeError(
                f"status must be a RefinementStatus member, got {self.status!r}"
            )
        targets = tuple(self.target_neurons)
        for position, item in enumerate(targets):
            if not isinstance(item, NeuronRef):
                raise TypeError(
                    f"target_neurons[{position}] must be a NeuronRef, "
                    f"got {type(item).__name__}"
                )
        seen: set[tuple[int, str, int]] = set()
        for item in targets:
            key = (item.layer_index, item.phase, item.neuron_index)
            if key in seen:
                raise ValueError(
                    "target_neurons must not contain duplicates; got duplicate "
                    f"reference (layer_index={item.layer_index}, phase={item.phase!r}, "
                    f"neuron_index={item.neuron_index})"
                )
            seen.add(key)
        outcomes = tuple(self.outcomes)
        for position, item in enumerate(outcomes):
            if not isinstance(item, NeuronRefinementOutcome):
                raise TypeError(
                    f"outcomes[{position}] must be a NeuronRefinementOutcome, "
                    f"got {type(item).__name__}"
                )
        if [outcome.neuron for outcome in outcomes] != list(targets):
            raise ValueError(
                "outcomes must align one-to-one with target_neurons in order"
            )
        execution_time = float(self.execution_time_seconds)
        if math.isnan(execution_time) or execution_time < 0.0:
            raise ValueError(
                f"execution_time_seconds must be nonnegative, got {execution_time}"
            )
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string")

        expected = _derive_status(outcomes)
        if self.status is not expected:
            raise ValueError(
                f"status {self.status.value} is inconsistent with the outcomes; "
                f"expected {expected.value}"
            )

        object.__setattr__(self, "target_neurons", targets)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "execution_time_seconds", execution_time)

    @property
    def refined_bounds(self) -> tuple[RefinedBounds, ...]:
        """The accepted refinements, ready for ``propagate_with_refinements``.

        Rejected neurons are absent, so applying this tuple to the network
        never widens or overrides any IBP bound.
        """
        return tuple(
            outcome.refined for outcome in self.outcomes if outcome.refined is not None
        )

    @property
    def num_targets(self) -> int:
        """Number of deduplicated targets processed."""
        return len(self.target_neurons)

    @property
    def num_accepted(self) -> int:
        """Number of targets with an accepted refined interval."""
        return sum(1 for outcome in self.outcomes if outcome.accepted)

    @property
    def num_rejected(self) -> int:
        """Number of targets whose IBP bounds were preserved."""
        return self.num_targets - self.num_accepted

    def __str__(self) -> str:
        return (
            f"RefinementResult(status={self.status.value}, "
            f"targets={self.num_targets}, accepted={self.num_accepted}, "
            f"rejected={self.num_rejected}, "
            f"time={self.execution_time_seconds:.4f}s)"
        )


def _refine_one(
    problem: LPProblem,
    trace: IntermediateBounds,
    neuron: NeuronRef,
) -> NeuronRefinementOutcome:
    """Solve both directions for one neuron and assemble its outcome.

    The refined interval is accepted only when both certified conservative
    bounds are available, stay inside the IBP envelope up to ``_TOLERANCE``,
    and do not cross. Any other case preserves the IBP bounds (``refined``
    is ``None``) and records the reason.
    """
    ibp_lower, ibp_upper = _ibp_interval(trace, neuron)
    lower_solve = solve_lp(problem, neuron)
    upper_solve = solve_lp(problem, neuron, maximize=True)

    refined: Optional[RefinedBounds] = None
    problems: list[str] = []

    if not lower_solve.is_certified:
        problems.append(
            f"lower bound not certified ({lower_solve.status.value}: "
            f"{lower_solve.message})"
        )
    elif float(lower_solve.certified_bound) < ibp_lower - _TOLERANCE:
        problems.append(
            f"certified lower {float(lower_solve.certified_bound):.6g} violates "
            f"the IBP lower bound {ibp_lower:.6g}"
        )
    if not upper_solve.is_certified:
        problems.append(
            f"upper bound not certified ({upper_solve.status.value}: "
            f"{upper_solve.message})"
        )
    elif float(upper_solve.certified_bound) > ibp_upper + _TOLERANCE:
        problems.append(
            f"certified upper {float(upper_solve.certified_bound):.6g} violates "
            f"the IBP upper bound {ibp_upper:.6g}"
        )

    if not problems:
        lower = float(lower_solve.certified_bound)
        upper = float(upper_solve.certified_bound)
        if lower > upper:
            problems.append(
                f"certified bounds cross: lower {lower:.6g} exceeds upper {upper:.6g}"
            )
        else:
            refined = RefinedBounds(neuron, lower, upper)

    if refined is not None:
        message = (
            f"refinement [{refined.lower:.6g}, {refined.upper:.6g}] accepted "
            f"(IBP [{ibp_lower:.6g}, {ibp_upper:.6g}])"
        )
    else:
        message = (
            f"refinement rejected: {'; '.join(problems)}; IBP interval "
            f"[{ibp_lower:.6g}, {ibp_upper:.6g}] preserved"
        )
    return NeuronRefinementOutcome(
        neuron=neuron,
        ibp_lower=ibp_lower,
        ibp_upper=ibp_upper,
        lower_solve=lower_solve,
        upper_solve=upper_solve,
        refined=refined,
        message=message,
    )


def refine_neuron_bounds(
    network: MathematicalNetwork,
    input_region: InputRegion,
    intermediate_bounds: IntermediateBounds,
    target_neurons: tuple[NeuronRef, ...] | list[NeuronRef],
) -> RefinementResult:
    """Refine the IBP bounds of the requested neurons with LP relaxation.

    For every requested neuron the engine builds one shared LP relaxation of
    the network and solves the minimization and the maximization of the
    neuron's variable. The refined interval is accepted only when both
    certified conservative bounds are available and consistent (see the
    module docstring for the exact acceptance and soundness contract).
    Rejected neurons keep their original IBP bounds and are absent from
    :attr:`RefinementResult.refined_bounds`, so the result can be handed
    directly to Member 1's ``propagate_with_refinements``.

    Args:
        network: An extracted float64 MathematicalNetwork with Linear and
            ReLU layers in execution order.
        input_region: The L-infinity input perturbation region.
        intermediate_bounds: The baseline IBP trace produced by
            ``get_intermediate_bounds`` for a region containing
            ``input_region``.
        target_neurons: A tuple or list of NeuronRef targets. Duplicates are
            deduplicated; invalid references raise before any LP is solved.

    Returns:
        A RefinementResult with one outcome per deduplicated target; use
        ``result.refined_bounds`` for the accepted RefinedBounds records.

    Raises:
        TypeError: If an argument has the wrong type, the network contains an
            unsupported layer, or a target is not a NeuronRef.
        ValueError: If the region dimension does not match the network, the
            trace does not match the network or does not contain the region,
            or a target reference is invalid.
    """
    start = time.perf_counter()
    _validate_arguments(network, input_region, intermediate_bounds)
    targets = _deduplicate(_coerce_targets(target_neurons))

    if not targets:
        elapsed = time.perf_counter() - start
        return RefinementResult(
            status=RefinementStatus.EMPTY_TARGETS,
            target_neurons=(),
            outcomes=(),
            execution_time_seconds=elapsed,
            message="no target neurons requested",
        )

    for neuron in targets:
        _validate_target(network, intermediate_bounds, neuron)

    # The constraint set does not depend on the objective, so one LP is built
    # and reused for every target; solve_lp accepts per-call objective
    # overrides. build_lp_problem re-validates the trace against the network
    # and the region containment before any solving happens.
    problem = build_lp_problem(network, input_region, intermediate_bounds, targets[0])
    outcomes = tuple(
        _refine_one(problem, intermediate_bounds, neuron) for neuron in targets
    )
    elapsed = time.perf_counter() - start

    accepted = sum(1 for outcome in outcomes if outcome.accepted)
    tightened = sum(1 for outcome in outcomes if outcome.tightens)
    status = _derive_status(outcomes)
    message = (
        f"{len(outcomes)} targets processed: {accepted} accepted "
        f"({tightened} tightened), {len(outcomes) - accepted} rejected "
        f"in {elapsed:.4f}s"
    )
    return RefinementResult(
        status=status,
        target_neurons=targets,
        outcomes=outcomes,
        execution_time_seconds=elapsed,
        message=message,
    )
