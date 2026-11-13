"""Integrated Phase 3 verification pipeline: IBP + LP refinement + safety check.

This module wires the complete Phase 3 workflow together. It consumes
Member 1's propagation components (:func:`abstract_domain.propagation.get_intermediate_bounds`,
:func:`abstract_domain.refinement.propagate_with_refinements`) and Member 2's
LP components (:func:`verification.refinement.refine_neuron_bounds`, which
internally drives :mod:`verification.lp` and :mod:`verification.lp_solver`),
and returns the original IBP verdict, the LP-refined verdict, and the
refinement diagnostics in one immutable record.

Workflow
--------
``verify_robustness_refined(network, region, target_class, target_neurons)``:

1. Validate the arguments (the same contract as ``verify_robustness``).
2. Propagate the region through the network with plain IBP
   (``get_intermediate_bounds``) and keep the full layer trace.
3. Record the IBP-only verdict with the existing ``SafetyChecker``; the
   certification condition is used completely unchanged.
4. Select the target neurons to refine (the explicit list when supplied,
   otherwise the default selector ``select_refinement_targets``).
5. Build one LP relaxation and solve the minimization and maximization for
   every target with ``refine_neuron_bounds``. Only numerically validated,
   IBP-enveloped, non-crossing certified bounds are accepted (see
   :mod:`verification.refinement`).
6. Apply the accepted refinements with ``propagate_with_refinements``, which
   re-validates every refinement against the original IBP trace and
   recomputes all downstream bounds.
7. Re-run the unchanged ``SafetyChecker`` on the refined output bounds.
8. Return a :class:`RefinedVerificationResult` carrying the preserved IBP
   verdict, the refinement diagnostics, the refined verdict, and the refined
   downstream trace.

Behavior
--------
- The original IBP verdict is always preserved for comparison; the refined
  downstream trace only overrides bounds that were accepted.
- When no refinement is accepted (empty target list, solver failure, or
  rejection) the refined downstream trace is the IBP trace itself and the
  refined verdict equals the IBP verdict: an inconclusive result remains
  inconclusive unless the refined bounds establish strict dominance.
- The pipeline never reports a violation. ``SafetyChecker`` only ever
  returns ``CERTIFIED`` or ``INCONCLUSIVE``; a failing sufficient condition
  is reported as ``INCONCLUSIVE``, never as "unsafe".
- Refinements that fail Member 1's downstream validation raise
  ``ValueError`` instead of being applied; the pipeline never overrides an
  IBP bound with an invalid interval.

Soundness
---------
The soundness of the final verdict inherits the limits documented in
:mod:`verification.lp`, :mod:`verification.lp_solver`, and
:mod:`verification.refinement`: the LP feasible set is a sound outer
approximation of the concrete network (Planet/triangle relaxation), and an
accepted refined bound is a conservative dual bound whose numerical
validation is tolerance-based (1e-8 relative) -- not a rigorous
interval-arithmetic certificate. Downstream propagation uses Member 1's
validated application, whose soundness is conditional on the soundness of
the supplied refinements. This module makes no additional formal claims and
reports the verification verdict as sound only under the same assumptions
as the components it composes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from abstract_domain.bounds import IntermediateBounds, NeuronRef, RefinedBounds
from models.model_loader import LinearLayer, MathematicalNetwork, ReLULayer
from verification import VerificationResult, VerificationStatus
from verification.refinement import RefinementResult, refine_neuron_bounds
from verification.regions import InputRegion
from verification.safety import SafetyChecker

# NOTE: ``abstract_domain.propagation`` and ``abstract_domain.refinement`` are
# NOT imported at module level. ``abstract_domain.propagation`` imports
# ``verification.regions`` (running ``verification/__init__.py``), and this
# module is itself imported from that ``__init__``; a module-level import of
# ``propagation`` would therefore recurse whenever ``abstract_domain`` is
# imported first. Both are imported lazily inside
# ``verify_robustness_refined``, where they are guaranteed to be fully
# loaded.

__all__ = [
    "RefinedVerificationResult",
    "select_refinement_targets",
    "verify_robustness_refined",
]


def _validate_query(
    network: object,
    region: object,
    target_class: object,
) -> None:
    """Mirror the argument contract of ``verify_robustness`` exactly."""
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
            f"region dimension {region.dimension} does not match "
            f"network.input_features {network.input_features}"
        )


def _check(box: object, target_class: int, epsilon: float) -> VerificationResult:
    """Run the unchanged SafetyChecker over a Box and attach the epsilon."""
    result = SafetyChecker().check(box.lower, box.upper, target_class)
    return replace(result, perturbation_epsilon=epsilon)


def select_refinement_targets(
    network: MathematicalNetwork,
    intermediate_bounds: IntermediateBounds,
    *,
    include_hidden_relu: bool = False,
) -> tuple[NeuronRef, ...]:
    """Select the default Phase 3 refinement targets for a network trace.

    By default, the selector targets every neuron of the final layer
    (``"pre"`` for a LinearLayer, ``"post"`` for a ReLULayer) so the
    output bounds consumed by the safety check are refined directly without
    spending LP solves on hidden ReLU neurons whose static LP refinements do
    not alter the final output bounds.

    When ``include_hidden_relu=True``, the selector additionally includes
    every hidden unstable ReLU neuron's post-activation bound (pre-activation
    interval spans zero, where the triangle relaxation is active).

    The returned references are unique and ordered by layer index; they are
    directly consumable by :func:`verification.refinement.refine_neuron_bounds`
    and always respect the shared Phase 3 phase conventions (linear layers
    expose only ``"pre"``).

    Args:
        network: An extracted float64 MathematicalNetwork.
        intermediate_bounds: The IBP trace produced by
            ``get_intermediate_bounds`` for the queried region.
        include_hidden_relu: Explicit opt-in flag to also include unstable
            hidden ReLU ``"post"`` neurons. Defaults to ``False``.

    Returns:
        A tuple of unique NeuronRef targets (possibly empty for a network
        with no layers).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the trace does not record one entry per network layer.
    """
    if not isinstance(network, MathematicalNetwork):
        raise TypeError(f"network must be a MathematicalNetwork, got {type(network).__name__}")
    if not isinstance(intermediate_bounds, IntermediateBounds):
        raise TypeError(
            "intermediate_bounds must be an IntermediateBounds, "
            f"got {type(intermediate_bounds).__name__}"
        )
    if not isinstance(include_hidden_relu, bool):
        raise TypeError(
            f"include_hidden_relu must be a bool, got {type(include_hidden_relu).__name__}"
        )
    trace = intermediate_bounds
    if len(trace.layer_bounds) != len(network.layers):
        raise ValueError(
            f"trace records {len(trace.layer_bounds)} layers but the network "
            f"has {len(network.layers)}"
        )

    targets: list[NeuronRef] = []
    seen: set[tuple[int, str, int]] = set()

    def add(neuron: NeuronRef) -> None:
        key = (neuron.layer_index, neuron.phase, neuron.neuron_index)
        if key not in seen:
            seen.add(key)
            targets.append(neuron)

    if include_hidden_relu:
        for index, layer in enumerate(network.layers):
            entry = trace.layer_bounds[index]
            if isinstance(layer, ReLULayer):
                pre = entry.pre_activation_bounds
                for j in range(pre.dimension):
                    if pre.lower[j] < 0.0 < pre.upper[j]:
                        add(NeuronRef(index, "post", j))

    if network.layers:
        last_index = len(network.layers) - 1
        last_layer = network.layers[last_index]
        phase = "pre" if isinstance(last_layer, LinearLayer) else "post"
        output_box = trace.layer_bounds[last_index].output_bounds
        for j in range(output_box.dimension):
            add(NeuronRef(last_index, phase, j))

    return tuple(targets)


@dataclass(frozen=True)
class RefinedVerificationResult:
    """Complete Phase 3 outcome: IBP verdict, refinement diagnostics, refined verdict.

    The record stores both verdicts side by side so the original IBP result
    is preserved for comparison, together with the LP refinement diagnostics
    and the downstream trace the refined verdict was computed from. The
    refined verdict is validated to agree exactly with ``refined_trace``'s
    output bounds, so the two can never disagree.

    Args:
        ibp_result: The original IBP-only VerificationResult (preserved).
        refinement: The RefinementResult diagnostics of the LP stage,
            including every per-neuron solver outcome.
        refined_result: The VerificationResult over the refined downstream
            bounds (equal to ``ibp_result`` when nothing was accepted).
        refined_trace: The IntermediateBounds trace recomputed with the
            accepted refinements (the IBP trace when nothing was accepted).
        message: Human-readable summary of the three stages.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the two verdicts disagree on the target class or the
            perturbation epsilon, the refined verdict is inconsistent with
            ``refined_trace.output_bounds``, or the message is blank.
    """

    ibp_result: VerificationResult
    refinement: RefinementResult
    refined_result: VerificationResult
    refined_trace: IntermediateBounds
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.ibp_result, VerificationResult):
            raise TypeError(
                f"ibp_result must be a VerificationResult, got {type(self.ibp_result).__name__}"
            )
        if not isinstance(self.refined_result, VerificationResult):
            raise TypeError(
                "refined_result must be a VerificationResult, "
                f"got {type(self.refined_result).__name__}"
            )
        if not isinstance(self.refinement, RefinementResult):
            raise TypeError(
                f"refinement must be a RefinementResult, got {type(self.refinement).__name__}"
            )
        if not isinstance(self.refined_trace, IntermediateBounds):
            raise TypeError(
                f"refined_trace must be an IntermediateBounds, got {type(self.refined_trace).__name__}"
            )
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string")
        if self.ibp_result.target_class != self.refined_result.target_class:
            raise ValueError(
                "ibp_result and refined_result must share the same target class"
            )
        if self.ibp_result.perturbation_epsilon != self.refined_result.perturbation_epsilon:
            raise ValueError(
                "ibp_result and refined_result must share the same perturbation epsilon"
            )

        lower = self.refined_trace.output_bounds.lower
        upper = self.refined_trace.output_bounds.upper
        target = self.refined_result.target_class
        if self.refined_result.target_lower_bound != lower[target]:
            raise ValueError(
                "refined_result target lower bound must equal the refined "
                "trace output bound"
            )
        if upper.size > 1:
            competitor_uppers = [
                upper[index] for index in range(upper.size) if index != target
            ]
            if self.refined_result.max_competitor_upper_bound != max(competitor_uppers):
                raise ValueError(
                    "refined_result competitor upper bound must equal the "
                    "refined trace output bound"
                )

    @property
    def status(self) -> VerificationStatus:
        """The final verdict: the refined result's status."""
        return self.refined_result.status

    @property
    def perturbation_epsilon(self) -> float | None:
        """The queried L-infinity radius (shared by both verdicts)."""
        return self.refined_result.perturbation_epsilon

    @property
    def applied_refinements(self) -> tuple[RefinedBounds, ...]:
        """The accepted refinements that were applied downstream."""
        return self.refinement.refined_bounds

    @property
    def margin_delta(self) -> float:
        """Refined margin minus IBP margin (0.0 when both margins are equal)."""
        ibp_margin = self.ibp_result.verification_margin
        refined_margin = self.refined_result.verification_margin
        if ibp_margin == refined_margin:
            return 0.0
        return refined_margin - ibp_margin

    @property
    def improved(self) -> bool:
        """Whether the refined margin strictly exceeds the IBP margin."""
        return self.refined_result.verification_margin > self.ibp_result.verification_margin

    def __str__(self) -> str:
        return (
            f"RefinedVerificationResult(ibp={self.ibp_result.status.value}, "
            f"refinement={self.refinement.status.value}, "
            f"refined={self.refined_result.status.value}, "
            f"margin_delta={self.margin_delta:.6g})"
        )


def verify_robustness_refined(
    network: MathematicalNetwork,
    region: InputRegion,
    target_class: int,
    target_neurons: tuple[NeuronRef, ...] | list[NeuronRef] | None = None,
    *,
    include_hidden_relu: bool = False,
) -> RefinedVerificationResult:
    """Run the complete Phase 3 workflow and return both verdicts plus diagnostics.

    Executes the full workflow documented in the module docstring: original
    IBP propagation, LP-based refinement of the selected neurons, downstream
    re-propagation with the accepted refinements, and the unchanged safety
    check over the refined output bounds. The original IBP verdict is
    preserved alongside the refined verdict for comparison.

    Args:
        network: Extracted float64 MathematicalNetwork containing ordered
            Linear and ReLU layers.
        region: L-infinity input perturbation region.
        target_class: Protected class index (non-negative integer).
        target_neurons: Optional explicit tuple/list of NeuronRef targets;
            when omitted, :func:`select_refinement_targets` chooses the
            default final-layer set. An empty collection performs IBP-only
            verification.
        include_hidden_relu: Explicit opt-in flag forwarded to
            :func:`select_refinement_targets` when ``target_neurons`` is
            omitted. Defaults to ``False``.

    Returns:
        A RefinedVerificationResult with the IBP verdict, the refinement
        diagnostics (per-neuron solver outcomes and aggregate status), the
        refined verdict, and the refined downstream trace.

    Raises:
        TypeError: If network, region, target_class, or include_hidden_relu
            has the wrong type, or target_neurons is not a tuple/list of
            NeuronRef.
        ValueError: If the region dimension does not match the network, the
            target class is out of range, ``include_hidden_relu=True`` is
            combined with explicit ``target_neurons``, a target reference is
            invalid, or an accepted refinement fails the downstream validation
            of ``propagate_with_refinements``.
    """
    _validate_query(network, region, target_class)
    if not isinstance(include_hidden_relu, bool):
        raise TypeError(
            f"include_hidden_relu must be a bool, got {type(include_hidden_relu).__name__}"
        )
    if target_neurons is not None and include_hidden_relu:
        raise ValueError(
            "include_hidden_relu cannot be combined with explicit target_neurons"
        )

    # Deferred imports: see the NOTE at module level. By the time this
    # function runs every import chain has completed, so the lazy imports
    # cannot recurse.
    from abstract_domain.propagation import get_intermediate_bounds
    from abstract_domain.refinement import propagate_with_refinements

    trace = get_intermediate_bounds(network, region)
    ibp_result = _check(trace.output_bounds, target_class, region.radius)

    targets = (
        select_refinement_targets(
            network,
            trace,
            include_hidden_relu=include_hidden_relu,
        )
        if target_neurons is None
        else target_neurons
    )
    refinement = refine_neuron_bounds(network, region, trace, targets)

    if refinement.refined_bounds:
        refined_trace = propagate_with_refinements(
            network, region, refinement.refined_bounds
        )
    else:
        # Nothing was accepted: keep the original IBP trace. The refined
        # verdict then equals the IBP verdict by construction, and an
        # inconclusive result is never upgraded without certified bounds.
        refined_trace = trace

    refined_result = _check(refined_trace.output_bounds, target_class, region.radius)

    message = (
        f"IBP verdict {ibp_result.status.value} "
        f"(margin {ibp_result.verification_margin:.6g}); "
        f"refinement {refinement.status.value} "
        f"({refinement.num_accepted}/{refinement.num_targets} accepted); "
        f"refined verdict {refined_result.status.value} "
        f"(margin {refined_result.verification_margin:.6g})"
    )
    return RefinedVerificationResult(
        ibp_result=ibp_result,
        refinement=refinement,
        refined_result=refined_result,
        refined_trace=refined_trace,
        message=message,
    )
