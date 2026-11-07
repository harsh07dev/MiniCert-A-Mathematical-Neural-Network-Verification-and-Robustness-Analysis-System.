# Model Loader and Mathematical Parameter Extraction

## Purpose

`models/model_loader.py` provides two related operations:

1. Strictly load PyTorch parameters from a checkpoint.
2. Convert supported Linear/ReLU execution paths into an ordered mathematical representation using NumPy arrays.

The module does not train models, propagate intervals, or perform LP refinement. The `MiniCertNet` architecture is unchanged.

## Supported models and operations

### Checkpoint loading

`load_model()` can instantiate a caller-supplied `nn.Module` class and strictly load its state dictionary. Checkpoint loading alone does not imply that the resulting model is supported for mathematical extraction.

Both checkpoint layouts are accepted:

- A raw PyTorch state dictionary produced by `torch.save(model.state_dict(), path)`, including the existing `saved_model.pth`.
- A dictionary produced by `save_model()`, with `state_dict` and optional constructor `metadata` entries.

### Mathematical extraction

`extract_network()` explicitly supports:

- Direct `nn.Sequential` instances with supported layers and any compatible hidden-layer sizes.
- The exact `MiniCertNet` class, whose known forward method flattens inputs and then executes `model.network`, an `nn.Sequential` instance.

Supported sequence operations are:

- `nn.Linear` with a bias.
- `nn.ReLU`, including in-place ReLU, which has the same mathematical function for this representation.

No MiniCertNet dimensions are hardcoded in the extraction algorithm. Dimensions come from each `nn.Linear` instance.

### Unsupported models and operations

The extractor rejects rather than skips:

- `nn.Flatten`, `nn.Dropout`, `nn.BatchNorm1d`, and other batch-normalization layers.
- `nn.Sigmoid`, Softmax, and every other unsupported activation or operation.
- Nested `nn.Sequential` containers unless explicit support is added.
- Linear layers created with `bias=False`.
- Empty sequences and sequences without a Linear layer.
- Arbitrary wrapper modules, even if they contain a Sequential attribute.
- Sequential subclasses that override `forward`, and custom Linear/ReLU subclasses whose execution semantics may differ from the standard classes.

The wrapper restriction is intentional: inspecting child modules does not reveal arbitrary forward-method semantics such as scaling, residual connections, conditional branches, or repeated layer use. In particular, the existing custom-forward `FCNet` is not currently supported by the strict extractor. A future adapter must define and validate its execution semantics before extraction.

## Strict and safe checkpoint loading

`load_model()` performs the following checks:

1. The checkpoint path must exist and be a file.
2. On PyTorch versions that expose the `weights_only` argument, loading uses `weights_only=True` to restrict deserialization to tensor-oriented checkpoint data.
3. The checkpoint must contain a non-empty mapping from string parameter names to tensors.
4. Optional metadata must be a mapping with string keys.
5. `model.load_state_dict(..., strict=True)` rejects missing, unexpected, and shape-incompatible parameters.
6. The loaded model is moved to the requested device and placed in evaluation mode.

Missing files raise `FileNotFoundError`. Invalid or unsafe files, malformed state dictionaries, and strict parameter mismatches raise `CheckpointError` with the underlying reason.

On older PyTorch versions without `weights_only`, the loader uses the available `torch.load` API for compatibility. Such versions do not provide the same deserialization restriction; using a current PyTorch version and trusted checkpoint sources is an integration assumption.

## Mathematical representation

`extract_network()` returns a `MathematicalNetwork` dataclass containing:

- `layers`: an immutable tuple of `LinearLayer` and `ReLULayer` records.
- `input_features`: input dimension of the first Linear layer.
- `output_features`: dimension after the final operation.
- `source_model`: the source class name.

Each record preserves its original zero-based sequence `position`.

### LinearLayer

A `LinearLayer` stores:

| Field | Meaning |
|---|---|
| `weight` | NumPy float64 matrix \(W\) with shape `(output_features, input_features)` |
| `bias` | NumPy float64 vector \(b\) with shape `(output_features,)` |
| `input_features` | Input dimension \(n\) |
| `output_features` | Output dimension \(m\) |
| `position` | Original Sequential execution index |

For one column input vector \(x \in \mathbb{R}^{n}\), it represents

\[
y = Wx + b,
\qquad
W \in \mathbb{R}^{m \times n},
\quad b \in \mathbb{R}^{m},
\quad y \in \mathbb{R}^{m}.
\]

For a row-based batch \(X\), `LinearLayer.apply()` computes \(XW^\mathsf{T}+b\), with bias broadcasting across rows. Extracted arrays are independent CPU copies rather than views of PyTorch storage.

### ReLULayer

A `ReLULayer` stores `activation="relu"` and its sequence position. It applies

\[
\operatorname{ReLU}(x)_j = \max(0,x_j)
\]

elementwise without changing dimensions.

## Execution-order and compatibility validation

Extraction traverses the explicitly supported Sequential path from position zero upward. It never collects parameters from unrelated `named_modules()` order.

For every Linear layer after the first, the extractor checks

\[
\text{layer.in\_features} = \text{features produced by the preceding operation}.
\]

ReLU preserves the current dimension. A mismatch raises `LayerCompatibilityError`; an unsupported operation raises `UnsupportedLayerError`. Thus, a network such as `Linear(4, 3) → Linear(2, 2)` is rejected before mathematical evaluation.

## Using the representation

```python
import torch
from models.model_loader import extract_network, load_model
from models.network import MiniCertNet

model = load_model(MiniCertNet, "saved_model.pth")
network = extract_network(model)

images = torch.randn(8, 1, 28, 28)
with torch.no_grad():
    torch_logits = model(images).numpy()

# MiniCertNet flattening is preprocessing outside the Sequential path.
numpy_logits = network.forward(images.flatten(start_dim=1).numpy())
```

`MathematicalNetwork.forward()` accepts only:

- One feature vector with shape `(input_features,)`.
- A feature batch with shape `(batch_size, input_features)`.

It does not infer arbitrary preprocessing. Callers extracting MiniCertNet must flatten image batches first, as shown above. Direct Sequential models are likewise treated as feature-based networks.

For consumers that need only affine parameters, `extract_layer_parameters(model, as_numpy=True)` returns `(weight, bias)` tuples after performing the same full-path validation. `inspect_model_architecture()` also validates the supported path before returning its summary.

## Numerical arithmetic and tolerance

PyTorch model parameters and execution normally use float32. Extraction converts exact stored float32 values to float64 NumPy arrays, and `MathematicalNetwork.forward()` performs its matrix and activation operations in float64.

The logits test compares extracted float64 evaluation with PyTorch float32 evaluation using:

```python
np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
```

This is a regression-test tolerance, not a universal error bound. Appropriate tolerances can depend on parameter magnitude, input scale, network size, and backend arithmetic.

**Converting to float64 does not by itself guarantee formally sound floating-point bounds.** Future verification engines must account for rounding, overflow, non-finite values, and the semantics of their chosen abstract domain or exact arithmetic. This module only extracts parameters and supplies a conventional NumPy evaluation path.

## Integration assumptions

- Consumers operate on the returned layer tuple in position order.
- Input preprocessing and output postprocessing outside the declared Sequential path remain the consumer's responsibility.
- No Softmax is inserted into the extracted network.
- Parameters are snapshots; later PyTorch training updates do not automatically update extracted arrays.
- GPU parameters are copied to CPU during extraction.
- Future abstract transformers must define their own sound treatment of Linear and ReLU operations; this module does not implement them.
