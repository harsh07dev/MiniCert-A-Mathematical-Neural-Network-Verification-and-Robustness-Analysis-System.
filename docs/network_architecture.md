# MiniCertNet Architecture

## 1. Purpose and network structure

`MiniCertNet` is a fully connected feedforward neural network for inputs representing 28 × 28 grayscale images. It maps each input to 10 raw output scores, called **logits**.

The implementation stores the computational layers in a PyTorch `nn.Sequential` container. Before those layers run, the input is flattened so that every sample is represented by 784 features. The layer sequence is:

| Step | Operation | Input features | Output features |
|---:|---|---:|---:|
| 0 | Flatten non-batch dimensions | 1 × 28 × 28, or already 784 | 784 |
| 1 | Linear | 784 | 64 |
| 2 | ReLU | 64 | 64 |
| 3 | Linear | 64 | 32 |
| 4 | ReLU | 32 | 32 |
| 5 | Linear | 32 | 10 |

There are three Linear layers and two ReLU layers. No activation is applied after the final Linear layer.

## 2. Input and output dimensions

Let \(B\) be the batch size. The forward method accepts either of the following input shapes:

\[
X \in \mathbb{R}^{B \times 1 \times 28 \times 28}
\quad\text{or}\quad
X \in \mathbb{R}^{B \times 784}.
\]

For image-shaped input, flattening converts each sample from shape \(1 \times 28 \times 28\) to a vector of length

\[
1 \cdot 28 \cdot 28 = 784.
\]

Already flattened input remains a \(B \times 784\) matrix. The batch dimension is preserved. The network output has shape

\[
Z^{(3)} \in \mathbb{R}^{B \times 10}.
\]

Thus, row \(i\) of the output contains the 10 logits for sample \(i\). The architecture has fixed feature dimensions, while \(B\) can vary.

## 3. Linear layers

For one sample represented as a column vector \(x \in \mathbb{R}^{n}\), a Linear layer computes

\[
\operatorname{Linear}(x) = Wx + b,
\]

where \(W \in \mathbb{R}^{m \times n}\) is the weight matrix, \(b \in \mathbb{R}^{m}\) is the bias vector, and the output belongs to \(\mathbb{R}^{m}\). Component \(j\) of the output is

\[
(Wx+b)_j = \sum_{k=1}^{n} W_{jk}x_k + b_j.
\]

Each output neuron therefore computes a weighted sum of all inputs to the layer and then adds its own bias.

### 3.1 First Linear layer: 784 → 64

\[
z^{(1)} = W^{(1)}x + b^{(1)},
\]

with

\[
x \in \mathbb{R}^{784}, \qquad
W^{(1)} \in \mathbb{R}^{64 \times 784}, \qquad
b^{(1)} \in \mathbb{R}^{64}, \qquad
z^{(1)} \in \mathbb{R}^{64}.
\]

The stored PyTorch weight tensor has shape `[64, 784]`, and the bias tensor has shape `[64]`.

### 3.2 Second Linear layer: 64 → 32

\[
z^{(2)} = W^{(2)}h^{(1)} + b^{(2)},
\]

with

\[
h^{(1)} \in \mathbb{R}^{64}, \qquad
W^{(2)} \in \mathbb{R}^{32 \times 64}, \qquad
b^{(2)} \in \mathbb{R}^{32}, \qquad
z^{(2)} \in \mathbb{R}^{32}.
\]

The stored PyTorch weight tensor has shape `[32, 64]`, and the bias tensor has shape `[32]`.

### 3.3 Third Linear layer: 32 → 10

\[
z^{(3)} = W^{(3)}h^{(2)} + b^{(3)},
\]

with

\[
h^{(2)} \in \mathbb{R}^{32}, \qquad
W^{(3)} \in \mathbb{R}^{10 \times 32}, \qquad
b^{(3)} \in \mathbb{R}^{10}, \qquad
z^{(3)} \in \mathbb{R}^{10}.
\]

The stored PyTorch weight tensor has shape `[10, 32]`, and the bias tensor has shape `[10]`.

For a batch stored as \(X \in \mathbb{R}^{B \times n}\), with one sample per row, PyTorch evaluates the equivalent matrix expression

\[
XW^\mathsf{T} + \mathbf{1}_B b^\mathsf{T},
\]

where \(\mathbf{1}_B\) is a column vector of \(B\) ones. This broadcasts the same bias vector across all samples.

## 4. ReLU activation

The Rectified Linear Unit (ReLU) is defined for a scalar \(u\) by

\[
\operatorname{ReLU}(u) = \max(0,u) =
\begin{cases}
u, & u \geq 0,\\
0, & u < 0.
\end{cases}
\]

For a vector, ReLU is applied independently to every component:

\[
\operatorname{ReLU}(v)_j = \max(0,v_j).
\]

ReLU does not change the vector's dimensions. It preserves nonnegative values and replaces negative values with zero.

## 5. Complete forward pass

For one flattened input vector \(x \in \mathbb{R}^{784}\), the complete computation is

\[
\begin{aligned}
z^{(1)} &= W^{(1)}x + b^{(1)} &&\in \mathbb{R}^{64},\\
h^{(1)} &= \operatorname{ReLU}(z^{(1)}) &&\in \mathbb{R}^{64},\\
z^{(2)} &= W^{(2)}h^{(1)} + b^{(2)} &&\in \mathbb{R}^{32},\\
h^{(2)} &= \operatorname{ReLU}(z^{(2)}) &&\in \mathbb{R}^{32},\\
z^{(3)} &= W^{(3)}h^{(2)} + b^{(3)} &&\in \mathbb{R}^{10}.
\end{aligned}
\]

Equivalently, as a single composed function \(f:\mathbb{R}^{784}\to\mathbb{R}^{10}\),

\[
f(x)=W^{(3)}
\operatorname{ReLU}\!\left(
W^{(2)}
\operatorname{ReLU}\!\left(W^{(1)}x+b^{(1)}\right)
+b^{(2)}
\right)+b^{(3)}.
\]

For batches, these operations are applied independently to each row. No operation mixes different samples in the batch.

## 6. Logits and the exclusion of Softmax

The 10 values in \(z^{(3)}\) are **logits**: unrestricted real-valued scores produced by the final Linear layer. A logit may be negative, zero, or positive, and the logits do not need to sum to one. They are not probabilities. Under the usual decision rule, the predicted class is the index with the largest logit:

\[
\widehat{y} = \operatorname*{arg\,max}_{j\in\{0,\ldots,9\}} z^{(3)}_j.
\]

Softmax would transform a logit vector \(z\) into a probability vector \(p\) using

\[
p_j = \frac{e^{z_j}}{\sum_{k=1}^{10}e^{z_k}}.
\]

Softmax is not part of `MiniCertNet`; the implementation returns \(z^{(3)}\) directly. This keeps the verification architecture limited to the implemented affine transformations and ReLU operations. It also avoids adding an unnecessary computation when only class comparison is needed: because Softmax is strictly increasing in each logit and uses a common positive denominator, it preserves the ordering and argmax of the logits. Verification can therefore reason directly about logit differences, such as whether a selected logit is greater than every other logit, without constructing probabilities.

## 7. Representation in Abstract Interpretation

Abstract interpretation reasons about a **set of possible tensors**, rather than only one concrete input. An abstract domain represents that set using a chosen mathematical structure. For example, an interval domain can give lower and upper bounds for every component of a tensor.

The implemented layer sequence supports direct layer-by-layer treatment:

1. **Flattening** changes tensor layout while preserving the feature values and their order in the flattened representation. A corresponding layout change is applied to the abstract input.
2. **Linear layers** are affine transformations. Their stored tensors \(W^{(\ell)}\) and \(b^{(\ell)}\) define the transformation exactly. An abstract transformer propagates the input abstraction through \(Wx+b\), using the rules of the selected abstract domain.
3. **ReLU layers** are elementwise transformations. An abstract transformer applies an abstract version of \(\max(0,u)\) to each component. In an interval domain, for example, an input interval \([l,u]\) can be transformed exactly to
   \[
   [\max(0,l),\max(0,u)].
   \]

Starting with an abstract input set \(\mathcal{X}\), the forward-pass equations become a sequence of abstract operations:

\[
\mathcal{Z}^{(1)} = T_W^{(1)}(\mathcal{X}),\qquad
\mathcal{H}^{(1)} = T_R(\mathcal{Z}^{(1)}),\qquad
\mathcal{Z}^{(2)} = T_W^{(2)}(\mathcal{H}^{(1)}),\qquad
\mathcal{H}^{(2)} = T_R(\mathcal{Z}^{(2)}),\qquad
\mathcal{Z}^{(3)} = T_W^{(3)}(\mathcal{H}^{(2)}).
\]

Here, \(T_W^{(\ell)}\) denotes the abstract transformer for Linear layer \(\ell\), and \(T_R\) denotes the abstract ReLU transformer. The final abstract output over-approximates all logits reachable from the abstract input. A future engine can extract the three Linear layers' weights and biases and traverse the two ReLU layers in their stored execution order. The particular abstract domain and its approximation choices remain design decisions for that engine; they are not fixed by this architecture.
