# Applying Refined Bounds and Downstream Repropagation

## Purpose

`abstract_domain/refinement.py` provides the mechanism to apply externally generated, validated intermediate neuron bounds to a neural network and recompute all downstream reachable intervals.

Standard Interval Bound Propagation (IBP) often yields loose bounds due to the dependency problem (wrapping effect), where correlations between intermediate neurons are discarded. Advanced certification techniques—such as Linear Programming (LP) relaxations (e.g., Planet relaxation) or Branch-and-Bound (BaB) domain bisection—produce strictly tighter intermediate bounds. `propagate_with_refinements()` acts as the deterministic execution engine that incorporates these tightened intervals and repropagates them through subsequent network layers.

The module operates independently of any LP solver: it accepts caller-provided `RefinedBounds` value objects, validates their consistency, overrides the targeted neuron bounds, and recomputes all downstream layer bounds using the existing `Box.affine()` and `Box.relu()` abstract transformers.

---

## Architecture and Data Structures

### 1. Neuron References (`NeuronRef`)

Intermediate activations are identified unambiguously using `NeuronRef`:

```python
NeuronRef(layer_index: int, phase: str, neuron_index: int)
```

- **`layer_index`**: The exact sequence index in `MathematicalNetwork.layers` (0-based integer). This refers to the actual layer execution position (e.g., `0` for the first Linear layer, `1` for the first ReLU layer, `2` for the second Linear layer), avoiding ambiguous "hidden-layer-only" numbering.
- **`phase`**: Specifies whether the targeted value is pre-activation (`"pre"`) or post-activation (`"post"`):
  - For `LinearLayer`: Only `"pre"` is valid, representing the affine transformation output $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$.
  - For `ReLULayer`: Both `"pre"` (the input argument to ReLU) and `"post"` (the non-negative activation output $\max(0, \mathbf{z})$) are valid.
- **`neuron_index`**: The zero-based coordinate $j \in [0, \text{dimension} - 1]$ of the neuron in that layer.

### 2. Refined Bounds (`RefinedBounds`)

A proposal to tighten a specific neuron's interval:

```python
RefinedBounds(neuron: NeuronRef, lower: float, upper: float)
```

Enforces $L_{\text{refined}} \le U_{\text{refined}}$ at construction time.

---

## Validation Pipeline

Before any refinement is applied, `propagate_with_refinements()` enforces strict multi-stage validation:

### 1. Structural and Type Validation
- `network` must be a valid `MathematicalNetwork`.
- `input_region` must be an `InputRegion` whose `dimension` strictly matches `network.input_features`.
- `refined_bounds` must be an iterable collection (`tuple` or `list`) containing exclusively `RefinedBounds` instances.
- **No Duplicate References**: Duplicate `NeuronRef` entries (identical `layer_index`, `phase`, and `neuron_index`) are rejected with `ValueError` to guarantee deterministic, collision-free application.

### 2. Topological Conformance
- Baseline IBP intermediate bounds are computed via `get_intermediate_bounds(network, input_region)`.
- Each `NeuronRef.layer_index` must exist within `network.layers`.
- The layer type must be compatible with the requested phase (`LinearLayer` allows only `"pre"`; `ReLULayer` allows `"pre"` and `"post"`).
- `NeuronRef.neuron_index` must be strictly within that layer's activation dimension.

### 3. Containment Consistency
Every refinement proposal $[L_{\text{refined}}, U_{\text{refined}}]$ is cross-referenced against the original IBP bounds $[L_{\text{IBP}}, U_{\text{IBP}}]$:

$$L_{\text{IBP}} - \epsilon_{\text{tol}} \le L_{\text{refined}} \le U_{\text{refined}} \le U_{\text{IBP}} + \epsilon_{\text{tol}}$$

Refinements that extend outside the original IBP interval are rejected with `ValueError`. A documented numerical tolerance ($\epsilon_{\text{tol}} = 10^{-12}$) guards against minor floating-point differences between solver implementations without permitting meaningfully invalid bounds.

---

## Downstream Propagation Algorithm

Refinement and repropagation proceed sequentially through `network.layers`:

1. **Input Initialization**: The initial activation box is set to `trace.input_bounds`.
2. **Linear Layer**:
   - The affine transformer calculates unrefined pre-activations: $\mathbf{B}_{\text{pre}} = \mathbf{B}_{\text{current}}.\text{affine}(\mathbf{W}, \mathbf{b})$.
   - Any pre-activation refinements for this layer index are applied, overriding the targeted neuron indices while preserving all unrefined neurons.
   - The updated box becomes the input for the next layer.
3. **ReLU Layer**:
   - Pre-activation refinements targeting the ReLU input are applied to the incoming box: $\mathbf{B}_{\text{pre}} = \text{apply\_refinements}(\mathbf{B}_{\text{current}}, \text{pre\_refinements})$.
   - The monotonic ReLU transformer computes intermediate post-activation bounds: $\mathbf{B}_{\text{post}} = \mathbf{B}_{\text{pre}}.\text{relu}()$.
   - **Post-Consistency Check**: Any post-activation refinements targeting this ReLU layer are verified against $\mathbf{B}_{\text{post}}$:
     $$\mathbf{B}_{\text{post}}.\text{lower}[j] - \epsilon_{\text{tol}} \le L_{\text{post\_refined}} \le U_{\text{post\_refined}} \le \mathbf{B}_{\text{post}}.\text{upper}[j] + \epsilon_{\text{tol}}$$
     If a post-refinement contradicts the recomputed ReLU output of the refined pre-activation bounds, it is rejected.
   - Post-activation refinements are applied to $\mathbf{B}_{\text{post}}$, which then feeds into the downstream layer.
4. **Output Integrity**: The final box is validated against `network.output_features` and packaged into a fresh, immutable `IntermediateBounds` trace.

---

## Why Containment Alone Does Not Prove Soundness

It is critical to distinguish between **containment validation** and **mathematical soundness**:

- **Containment is a Necessary, but Not Sufficient, Consistency Guard**:
  Verifying $[L_{\text{refined}}, U_{\text{refined}}] \subseteq [L_{\text{IBP}}, U_{\text{IBP}}]$ confirms that the proposed bounds do not violate known IBP outer limits. It catches obvious transposition errors, sign flips, or solver divergence.
- **Soundness Depends on Upstream Certification**:
  Proving that *no reachable network activation* lies in $[L_{\text{IBP}}, L_{\text{refined}})$ or $(U_{\text{refined}}, U_{\text{IBP}}]$ requires a sound mathematical argument—such as valid dual LP solutions or convex relaxations (Planet relaxation).
- `propagate_with_refinements()` does not verify the mathematical derivations of external proposals; it guarantees only that *if* the supplied refinements are sound, the downstream repropagation preserves that soundness.

---

## Limitations

1. **No Independent Bound Generation**: The module does not formulate optimization problems or generate refined bounds itself; it depends on external callers (such as future LP solvers or Branch-and-Bound search engines).
2. **Supported Operations**: Only `LinearLayer` and `ReLULayer` operations are supported, matching the existing mathematical extraction capabilities of `models/model_loader.py`.
3. **Double Precision**: Calculations use IEEE 754 standard double precision (`float64`) arithmetic with round-to-nearest mode. Hardware-level outward directed rounding is not implemented.

