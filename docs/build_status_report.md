# MiniCert — Build Status Report

**Scope:** What has actually been built in this repository, with evidence.
**Revision audited:** `5bebfa7` (`main`), working tree clean.
**Date:** 2026-11-09

---

## 1. Executive summary

MiniCert is a research-inspired framework for formal neural-network robustness verification using abstract interpretation (interval bound propagation) and LP relaxations. At the audited revision the project has a **tested reference network, a hardened checkpoint/extraction layer, a trained MNIST model, and a validated scalar interval abstract domain** — roughly half of the theoretical pipeline.

| Area | State | Evidence |
| :--- | :--- | :--- |
| Reference network (`MiniCertNet`) | **Done** | 9 tests, exact topology asserted, 52,650 params |
| Generic `FCNet` | **Partial** | Implemented, but rejected by the extractor |
| Checkpoint loading | **Done** | strict + `weights_only`, 7 failure modes tested |
| Mathematical extraction | **Done** | float64 `MathematicalNetwork`, order preserved |
| MNIST training pipeline | **Done** | 96.44% test accuracy, report generated |
| Scalar interval domain | **Done** | 34 tests passing locally |
| Box/vector domain | **Not built** | Scalar only; no NumPy abstraction |
| Affine (IBP) transformer | **Not built** | No `W⁺l + W⁻u + b` implementation |
| Verification engine / LP | **Contract only** | Enum + dataclass, no solver |
| Refinement / BaB | **Not built** | Docstring only |
| Dashboard | **Not built** | Docstring only |
| Experiments / benchmarks | **Not built** | Docstring only |

**Bottom line:** the foundation is solid and honestly documented, but the headline capability — returning `VERIFIED_ROBUST` / `VIOLATED` / `UNKNOWN` for a perturbed input — **does not exist yet**. Nothing in the repo currently propagates a perturbation box through a network.

---

## 2. Delivery timeline (git history)

| Commit | Date | Deliverable | Churn |
| :--- | :--- | :--- | :--- |
| `167b2ef` | 2026-11-03 | Project skeleton: 7 packages, README, `.gitignore`, `requirements.txt`, `FCNet`, initial loader/trainer, verification contracts | +1,254 |
| `10fe9d4` | 2026-11-03 | Merge of PR #1 (`memebr-1work`) | — |
| `62bd126` | 2026-11-03 | `MiniCertNet` + `tests/test_network.py` + `docs/network_architecture.md` + `docs/implementation_report.md` | +363 |
| `638b88d` | 2026-11-07 | `models/train_mnist.py` + trained checkpoint + `docs/training_report.md` | +209 |
| `0360e7c` | 2026-11-07 | Rewritten `models/model_loader.py` + `tests/test_model_loader.py` + `docs/model_loader.md` | +623 / −102 |
| `ec32841` | 2026-11-08 | `abstract_domain/interval.py` rewritten as scalar domain + `tests/test_interval.py` + `docs/interval_domain.md` | +466 / −281 |
| `5bebfa7` | 2026-11-08 | Merge of PR #2 (`member-1work`) | — |

Five functional commits across two PRs, six days, single author (`Harsh.dev`). The README's two-developer split is a plan, not reflected in history.

---

## 3. Module inventory

**2,430 lines** of Python + Markdown (excluding cached files): 1,085 implementation, 441 tests, 904 documentation.

### 3.1 `models/` — 837 lines

| File | Lines | Built |
| :--- | ---: | :--- |
| `network.py` | 180 | `MiniCertNet` (784→64→32→10 ReLU, flatten-then-network, logits only); `FCNet` (arbitrary `hidden_dims`, pluggable activation, `forward_with_intermediates`, `get_weights_and_biases`) |
| `model_loader.py` | 287 | `save_model`, `load_model`, `extract_network`, `extract_layer_parameters`, `inspect_model_architecture`; dataclasses `LinearLayer`, `ReLULayer`, `MathematicalNetwork`; error hierarchy `CheckpointError`, `UnsupportedModelError`, `UnsupportedLayerError`, `LayerCompatibilityError` |
| `train.py` | 171 | `train_epoch`, `evaluate`, `train_model` (Adam, CrossEntropy, optional val loop, history dict, checkpoint save) |
| `train_mnist.py` | 168 | `argparse` CLI: seeded run, download, train, test-evaluate, save, strict reload, bit-exact prediction equality assertion, Markdown report writer |
| `__init__.py` | 31 | Re-exports `FCNet`, `train_model`, `evaluate`, `save_model`, `load_model`, `extract_layer_parameters` |

Hardening detail worth noting: `load_model` requires `weights_only=True` where available, validates metadata types, binds metadata to constructor parameters via `inspect.signature` (honouring `**kwargs`), and enforces `strict=True`. `_get_sequential_path` refuses `nn.Sequential` subclasses whose `forward` is overridden and any wrapper whose forward semantics cannot be reconstructed.

### 3.2 `abstract_domain/` — 122 lines

`interval.py` (101 lines): frozen `slots` dataclass `Interval` with an explicit extended-real policy — `-inf` only as lower bound, `+inf` only as upper, NaN/reversed/degenerate-infinite rejected. Implements sound `+`, `-`, `*` (four-endpoint products with a documented `0·∞ = 0` bound convention), `/` (rejects any denominator containing zero), exact `relu()`, `contains`, `contains_interval`, `intersection` (returns `None` when disjoint), `width`.

Explicitly **scalar only**: no NumPy, no tensors, no model coupling, no affine transformer, no join/meet/split.

### 3.3 `verification/` — 67 lines (contract, no engine)

`VerificationStatus` (`VERIFIED_ROBUST`, `VIOLATED`, `UNKNOWN`) and `VerificationResult` (`status`, `target_class`, `epsilon`, `lower_bound`, `upper_bound`, `counterexample`, `solve_time_seconds`, `details`). No LP formulation, no `scipy.optimize.linprog` call, no `epsilon`-ball construction, no margin encoding.

### 3.4 Placeholders — 78 lines total

`refinement/` (21), `dashboard/` (19), `experiments/` (19) contain module docstrings describing intended responsibilities and `__all__ = []`. `tests/__init__.py` (15) is a docstring.

---

## 4. Test suite — 62 collected tests

| Suite | Tests | Status at audit |
| :--- | ---: | :--- |
| `tests/test_interval.py` | 34 | **34 passed** (0.03 s) — executed in this workspace |
| `tests/test_network.py` | 9 | Not runnable here — `torch` missing |
| `tests/test_model_loader.py` | 19 | Not runnable here — `numpy`, `torch` missing |

Interval coverage is genuinely rigorous: construction validation, all four arithmetic operators, sign combinations, zero-containing denominators, unbounded intervals, `0·∞`, NaN rejection, ReLU across all three sign regimes, scalar/interval containment, touching and disjoint intersections, width.

Model/loader coverage (verified by reading, not execution) asserts: exact layer type sequence and positions, `(784,64)/(64,32)/(32,10)` shapes, extracted arrays equal to PyTorch parameters, float64 dtype, extracted-vs-PyTorch logit agreement within `rtol=1e-5`, strict reload rejection of missing parameters, and rejection of `Flatten`/`Dropout`/`BatchNorm1d`/`Sigmoid`, dimension mismatches, overridden-`forward` subclasses and arbitrary wrappers.

**Environment blocker:** the local interpreter is **Python 3.14.3** with no dependencies installed and no virtual environment in the workspace, while `requirements.txt` targets **Python 3.11**. `pytest tests/` therefore aborts during collection. The 34 interval tests are the only tests verified in this environment.

---

## 5. Trained artifact

`models/train_mnist.py` produced a real training run, summarized in `docs/training_report.md`:

| Metric | Value |
| :--- | :--- |
| Architecture | 784 → 64 → 32 → 10 ReLU |
| Optimizer / LR / batch | Adam / 0.001 / 128 |
| Epochs | 5 |
| Final train loss / accuracy | 0.120006 / 96.50% |
| Test loss / accuracy | 0.119279 / **96.44%** |
| Wall clock (CPU) | 22.94 s |
| Checkpoint reload | Passed (`strict=True`) |
| Prediction parity after reload | **10,000 / 10,000** |

Caveats: the report was generated on another machine (`C:\Users\harsh\MiniCert\...`), and both `saved_model.pth` and `data/` are gitignored and absent from this repository. `test_trained_minicert_checkpoint_reload` therefore self-skips, and the run is **not reproducible from a fresh clone without retraining**.

---

## 6. Documentation — 904 lines

| Document | Lines | Contents |
| :--- | ---: | :--- |
| `README.md` | 198 | Motivation, threat model formula, package map, team split, IBP math, env setup, smoke tests, 4-phase roadmap |
| `docs/interval_domain.md` | 228 | Full interval semantics, sign cases, infinity policy, ReLU exactness, and an explicit **soundness limitation** section (no directed rounding ⇒ not a formal certificate) |
| `docs/model_loader.md` | 171 | Supported/rejected models, checkpoint safety checks, dtype and tolerance discussion, integration assumptions |
| `docs/network_architecture.md` | 205 | Layer-by-layer dimensions, forward equations, logits vs Softmax, abstract-interpretation hand-off |
| `docs/implementation_report.md` | 62 | Progress narrative for the network phase |
| `docs/training_report.md` | 40 | Auto-generated run report |

Documentation quality is above average for a student/research project: limitations are stated rather than hidden, and the float-rounding caveat is correctly flagged as disqualifying for "certified" claims.

---

## 7. Known gaps and defects

1. **README overstates Phase 1.** It claims a "Box domain with sound affine transformer" and "sound affine (`W⁺l + W⁻u + b`)" transformers; the code has a scalar `Interval` with `relu()` only. No matrix-level transformer exists, so `Interval` cannot consume `MathematicalNetwork`.
2. **`FCNet` is a dead end.** Exported from `models/__init__.py` and documented as verifiable, yet `extract_network()` supports only `nn.Sequential` and the exact `MiniCertNet` class. `docs/model_loader.md` acknowledges this; the README and `models/__init__.py` do not.
3. **No end-to-end verification path.** No `epsilon`-box construction, no margin encoding, no solver invocation, no status decision — the product's core promise is unimplemented.
4. **No certified arithmetic.** Float bounds without outward rounding are not formally sound, as `interval_domain.md` itself states. Must be resolved before any "certified robustness" claim.
5. **Reproducibility gaps.** `requirements.txt` uses `>=` only (README calls it "pinned"); no `pyproject.toml`, lockfile, CI workflow, or `data/`/checkpoint provisioning. Python version mismatch (3.14 local vs 3.11 declared) currently blocks 28 tests.
6. **Minor.** Training flattens inputs in both `train.py` and `network.py`; `train_model` saves a raw `state_dict` while `save_model` writes a metadata payload (both loadable, but two conventions); `models/__init__.py` omits the extraction types consumers need (`MathematicalNetwork`, error classes, `extract_network`).

---

## 8. Recommended next increment

1. **Add a vector `Box` domain and `propagate_intervals(network, x0, epsilon)`** using `l_out = W⁺l + W⁻u + b`, `u_out = W⁺u + W⁻l + b`; then a `verify_ibp()` that returns a real `VerificationResult`. This is the smallest change that makes the project demonstrably a verifier.
2. **Reconcile model support** — either add a parameter-protocol adapter for `FCNet` or remove/mark it deprecated, and correct the README.
3. **Pin the environment and add CI** (`pyproject.toml`, pinned versions, GitHub Actions running `pytest`) so the 28 currently unguarded tests execute on every push.
4. **Replace artifact-dependent tests** with a small deterministic fixture checkpoint generated in-test.
5. **Then** layer on Planet LP relaxation, Branch-and-Bound, and the Streamlit dashboard — the existing `VerificationResult` and `Interval` contracts are adequate extension points.
