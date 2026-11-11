# Empirical Soundness Validation of Refined Bounds

## Purpose and Motivation

In formal neural network verification, bounds produced by abstract interpretation or convex relaxations must enclose all reachable activation values. While Interval Bound Propagation (IBP) provides an outer envelope, more precise methods—such as Linear Programming (LP) relaxations or Branch-and-Bound (BaB) search—generate tightened intermediate bounds (`RefinedBounds`).

Before integrating Member 2's LP solver, MiniCert requires an empirical validation layer:
1. **Detecting Unsound Refinements**: If an optimization routine or LP formulation contains formulation flaws, sign errors, or numerical divergence, empirical evaluation can uncover counterexample inputs that violate the proposed bounds.
2. **Experimental Sanity Guard**: Verifying that empirical activations fall strictly within refined intervals provides immediate operational feedback during research and development.
3. **Validating Containment vs. Soundness**: A refinement $[L_{\text{refined}}, U_{\text{refined}}]$ may be structurally contained within original IBP bounds $[L_{\text{IBP}}, U_{\text{IBP}}]$ and therefore accepted by `propagate_with_refinements()`, yet still exclude reachable activations (e.g., when dependencies between cancelable terms are over-constrained). Empirical validation directly surfaces these discrepancies.

> [!IMPORTANT]
> **Empirical Sampling vs. Formal Certification**:
> Empirical testing can demonstrate that a refinement is **unsound** (by finding a counterexample), but **cannot mathematically prove** soundness for all uncountably many inputs in an $L_\infty$ perturbation region. Passing empirical checks is a mandatory diagnostic step, not a formal certificate.

---

## Architectural Components

The validation pipeline is implemented in [`verification/refinement_validation.py`](file:///c:/Users/harsh/MiniCert/verification/refinement_validation.py) and exposes three primary functions:

```
[InputRegion]
      │
      ▼
sample_region_inputs(region, num_samples, seed)
      │
      ▼ (Shape: num_samples, input_features)
evaluate_intermediate_activations(network, samples)
      │
      ▼ (ActivationTrace: dict mapping NeuronRef -> samples)
validate_refined_bounds_empirically(network, region, refined_bounds)
      │
      ▼
Structured Diagnostic JSON Report
```

---

## 1. Input Region Sampling (`sample_region_inputs`)

Given an `InputRegion`, concrete inputs $\mathbf{x} \in \mathbb{R}^n$ are sampled uniformly within the bounded hyper-rectangle:

$$\mathbf{x} \sim \mathcal{U}(\text{region.lower}, \text{region.upper})$$

- **Reproducibility**: Uses NumPy's `np.random.default_rng(seed)` to guarantee bit-exact cross-platform repeatability.
- **Domain Conformance**: Evaluates samples strictly within $[\text{lower}, \text{upper}]$, honoring configured input boundaries (e.g., $[0.0, 1.0]$ image pixel values).
- **Validation**: Rejects non-positive sample counts, non-integer types, and unbounded regions with infinite limits ($-\infty, +\infty$).

---

## 2. Concrete Activation Extraction (`evaluate_intermediate_activations`)

Evaluates the forward propagation of a batch of sampled inputs through a `MathematicalNetwork`:

1. **Topological Order**: Iterates sequentially through `network.layers`.
2. **Linear Layer**:
   - Computes affine transformations: $\mathbf{Y} = \mathbf{X} \mathbf{W}^T + \mathbf{b}$.
   - Records values under phase `"pre"` for each neuron coordinate $j$: `NeuronRef(layer_index, "pre", j)`.
3. **ReLU Layer**:
   - Records incoming arguments under phase `"pre"`: `NeuronRef(layer_index, "pre", j)`.
   - Computes elementwise activations: $\mathbf{Z} = \max(\mathbf{Y}, 0.0)$.
   - Records non-negative outputs under phase `"post"`: `NeuronRef(layer_index, "post", j)`.
4. **Data Structure (`ActivationTrace`)**:
   - Packaged as a tuple of per-layer dictionaries, supporting both layer indexing (`trace[layer_index]`) and direct coordinate lookup (`trace[neuron_ref]`).

---

## 3. Refinement Checking & Violation Measurement

For each proposed `RefinedBounds(neuron, lower, upper)`:

Let $\mathbf{a} \in \mathbb{R}^N$ be the vector of extracted concrete activations for `neuron` across $N$ samples.

- **Lower-Bound Violations**:
  $$\text{lower\_violations} = \{ a_i \mid a_i < L_{\text{refined}} \}$$
  $$\text{max\_lower\_violation} = \begin{cases} \max_{a_i < L_{\text{refined}}} (L_{\text{refined}} - a_i), & \text{if lower violations exist} \\ 0.0, & \text{otherwise} \end{cases}$$

- **Upper-Bound Violations**:
  $$\text{upper\_violations} = \{ a_i \mid a_i > U_{\text{refined}} \}$$
  $$\text{max\_upper\_violation} = \begin{cases} \max_{a_i > U_{\text{refined}}} (a_i - U_{\text{refined}}), & \text{if upper violations exist} \\ 0.0, & \text{otherwise} \end{cases}$$

- **Status**: The proposal passes if and only if $\text{violations} = 0$.

### Diagnostic JSON Report Structure

```json
{
  "num_samples": 1000,
  "seed": 42,
  "all_refinements_passed": true,
  "results": [
    {
      "neuron": {
        "layer_index": 0,
        "phase": "pre",
        "neuron_index": 2
      },
      "samples_checked": 1000,
      "violations": 0,
      "max_lower_violation": 0.0,
      "max_upper_violation": 0.0,
      "passed": true
    }
  ]
}
```

---

## 4. The Containment Pitfall: Why Containment Is Not Soundness

Consider a network where two intermediate linear combinations cancel:
- $y_0 = x_0 + x_1$
- $y_1 = x_0 - x_1$
- $z_0 = y_0 + y_1 = 2 x_0$

For $x_0, x_1 \in [0.0, 1.0]$:
- Standard IBP computes $y_0 \in [0, 2]$, $y_1 \in [-1, 1] \implies y_1 \in [0, 1]$ (after ReLU).
- IBP over-approximates $z_0 = y_0 + y_1 \in [0, 3]$.
- True reachable range of $z_0$ is $[0, 2]$.

If an unsound refinement proposes $z_0 \in [0.0, 1.5]$:
1. `propagate_with_refinements()` checks whether $[0.0, 1.5] \subseteq [0.0, 3.0]$ (original IBP interval). It **passes** and is applied.
2. `validate_refined_bounds_empirically()` draws inputs where $x_0 = 0.9 \implies z_0 = 1.8 > 1.5$. It flags **violations > 0** and reports `max_upper_violation = 0.3`.

This scenario highlights why independent empirical validation is essential before trusting proposed bounds.

---

## 5. Upcoming Integration with LP Solver

When Member 2 introduces the Linear Programming solver (e.g., Planet relaxation via `scipy.optimize.linprog`):
1. The LP solver will formulate constraints and output candidate lower and upper bounds for selected neurons.
2. `validate_refined_bounds_empirically()` will be run on candidate LP solutions across standard perturbation datasets.
3. Once empirically confirmed, bounds will be piped to `propagate_with_refinements()` to recompute the certified output reachability set.

