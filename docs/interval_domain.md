# Scalar Interval Abstract Domain

## Abstract domains and values

An **abstract domain** is a mathematical representation used to reason about many concrete values at once. Instead of executing an operation separately for every possible input, abstract interpretation applies an abstract operation to a set representation.

A concrete scalar value is a single real number, such as \(x=1.5\). An abstract interval value describes a nonempty set of possible scalars:

\[
X=[l,u]=\{x\in\mathbb{R}\mid l\leq x\leq u\}.
\]

Here, \(l\) is the lower bound and \(u\) is the upper bound. For example, \([-1,2]\) contains \(-1\), \(0\), \(1.5\), and \(2\), but not \(3\). A degenerate interval \([a,a]\) represents the singleton set \(\{a\}\).

The public implementation is the immutable `Interval` dataclass in `abstract_domain/interval.py`. It stores Python float bounds, is independent of PyTorch and the model loader, and currently supports scalar intervals only. Its small interface leaves room for a separate NumPy-based vector abstraction later without treating a scalar as a special neural-network tensor.

## Construction and validation

Every constructed interval must satisfy

\[
l\leq u.
\]

Construction rejects:

- NaN in either bound.
- A lower bound greater than the upper bound.
- Non-scalar or non-real bound objects.
- The degenerate infinite forms \([-\infty,-\infty]\) and \([+\infty,+\infty]\), which do not describe nonempty sets of real numbers under this module's policy.

### Infinite-bound policy

Unbounded intervals are supported:

- A lower bound may be \(-\infty\).
- An upper bound may be \(+\infty\).
- \([-\infty,u]\), \([l,+\infty]\), and \([-\infty,+\infty]\) are valid when the finite endpoint is valid.

Infinity is a bound marker, not a concrete real value. Width and arithmetic use the extended-bound conventions described below. Every operation constructs a new validated `Interval`; disjoint intersection instead returns `None` because this class represents only nonempty intervals.

## Arithmetic rules

Let

\[
X=[l_X,u_X], \qquad Y=[l_Y,u_Y].
\]

For an operation \(\diamond\), the required real-arithmetic soundness property is

\[
\forall x\in X,\ \forall y\in Y:\quad x\diamond y\in X\boxdiamond Y,
\]

where \(\boxdiamond\) denotes the corresponding interval operation.

### Addition

Interval addition is

\[
X+Y=[l_X+l_Y,\ u_X+u_Y].
\]

The smallest possible sum uses both lower bounds, and the largest uses both upper bounds.

```python
Interval(1, 3) + Interval(-2, 4) == Interval(-1, 7)
```

### Subtraction

Interval subtraction is

\[
X-Y=[l_X-u_Y,\ u_X-l_Y].
\]

The smallest result subtracts the largest possible \(y\); the largest result subtracts the smallest possible \(y\).

```python
Interval(1, 3) - Interval(-2, 4) == Interval(-3, 5)
```

### Multiplication

Multiplication examines all four endpoint products:

\[
P=\{l_Xl_Y,\ l_Xu_Y,\ u_Xl_Y,\ u_Xu_Y\},
\]

and returns

\[
X\cdot Y=[\min P,\ \max P].
\]

Considering all four products is necessary because negative and mixed-sign intervals can attain their extrema at different endpoint combinations.

```python
Interval(-2, 3) * Interval(-4, 5) == Interval(-12, 15)
```

For unbounded endpoints, the implementation evaluates a product as zero whenever either endpoint is zero, including a `0 * infinity` endpoint combination. This is a bound-computation convention, not a claim that \(0\cdot\infty\) is ordinary extended-real arithmetic. It handles cases such as

\[
[0,0]\cdot[-\infty,+\infty]=[0,0],
\qquad
[0,1]\cdot[0,+\infty]=[0,+\infty].
\]

All nonzero infinite products follow their sign. Every candidate set therefore produces ordered, non-NaN bounds under the constructor's policy.

### Division

A denominator interval \(Y\) is accepted only when

\[
0\notin Y.
\]

If \(l_Y\leq 0\leq u_Y\), including cases where zero is an endpoint, division raises `ZeroDivisionError`. This avoids unbounded reciprocal behavior across a pole and follows the task's explicit rejection policy.

For a nonzero-sign denominator, division is implemented using its reciprocal:

\[
\frac{1}{Y}=\left[\frac{1}{u_Y},\frac{1}{l_Y}\right],
\qquad
\frac{X}{Y}=X\cdot\frac{1}{Y}.
\]

The reciprocal bounds are ordered for both entirely positive and entirely negative denominators. An infinite denominator endpoint has reciprocal bound zero.

```python
Interval(6, 8) / Interval(2, 4) == Interval(1.5, 4)
```

## ReLU abstraction

The scalar Rectified Linear Unit is

\[
\operatorname{ReLU}(x)=\max(0,x).
\]

Because it is monotone non-decreasing, the exact interval image is

\[
\operatorname{ReLU}([l,u])=[\max(0,l),\max(0,u)].
\]

Thus:

- Positive intervals are unchanged.
- Entirely negative intervals become \([0,0]\).
- An interval crossing zero has its lower bound replaced by zero.

This transformation is exact over real arithmetic, not merely an over-approximation.

## Containment, intersection, and width

### Scalar containment

`X.contains(value)` implements

\[
l_X\leq\text{value}\leq u_X.
\]

NaN containment queries are rejected because NaN is unordered.

### Interval containment

`X.contains_interval(Y)` implements the subset test

\[
Y\subseteq X
\iff
l_X\leq l_Y\ \text{and}\ u_Y\leq u_X.
\]

### Intersection

`X.intersection(Y)` computes

\[
X\cap Y=[\max(l_X,l_Y),\min(u_X,u_Y)]
\]

when

\[
\max(l_X,l_Y)\leq\min(u_X,u_Y).
\]

Touching intervals produce a singleton. Disjoint intervals return `None`; no invalid interval is constructed.

### Width

The `width` property is

\[
\operatorname{width}(X)=u_X-l_X.
\]

A singleton has width zero. Any unbounded interval has width \(+\infty\).

## Soundness and numerical limitations

The formulas above are sound over mathematical real arithmetic: every concrete result is contained in the resulting interval. The implementation also validates its outputs and avoids NaN-producing extended-bound endpoint cases.

However, Python floats use binary floating-point arithmetic. The implementation does **not** perform directed rounding or outward rounding. A rounded lower endpoint can move upward, a rounded upper endpoint can move downward, and overflow can produce infinity. Consequently, ordinary float arithmetic alone does **not** establish a formally sound floating-point verification bound. Consumers requiring machine-checked floating-point guarantees must add an appropriately directed-rounding or exact-arithmetic layer and define treatment of overflow and non-finite results.

Other limitations are:

- **Dependency problem:** Repeated or correlated occurrences of the same variable are treated independently. For example, interval arithmetic evaluates \(X-X\) using all independent differences rather than recognizing that identical concrete occurrences always give zero.
- **Over-approximation:** Although individual scalar arithmetic images follow the standard interval rules, composed expressions can become much wider than the true correlated result set.
- **Scalar-only scope:** NumPy vector intervals, broadcasting, and tensor operations are not implemented.
- **No propagation or refinement:** Neural-network propagation, affine transformers, LP optimization, joins, and branch-and-bound operations are outside this module.
- **No Galois-connection framework:** The module provides the interval representation and operations only.

## Connection to future abstract transformers

A future abstract transformer can accept scalar intervals, inspect layer parameters supplied by another component, and apply the arithmetic and `relu()` operations in a mathematically defined order. Each transformer will be responsible for proving and documenting its own soundness, handling dependencies, and choosing suitable floating-point or exact arithmetic.

The interval module deliberately does not import neural-network classes or traverse models. This separation keeps the numerical abstraction reusable and prevents model-loading details from becoming part of interval arithmetic.
