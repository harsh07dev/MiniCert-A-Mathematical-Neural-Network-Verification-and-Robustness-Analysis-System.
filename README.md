# MiniCert: A Mathematical Neural Network Verification & Robustness Analysis System

MiniCert is a modular, research-inspired framework for formal neural network verification and certified robustness analysis. Built on the principles of **Abstract Interpretation** and **Linear Programming (LP) Relaxations**, MiniCert evaluates whether a neural network's predictions are mathematically guaranteed to remain invariant under adversarial input perturbations.

---

## 1. Project Motivation & Theoretical Context

Deep neural networks deployed in safety-critical applications (e.g., autonomous systems, medical diagnostics, robotics) are notoriously susceptible to adversarial perturbations—imperceptible input changes that induce arbitrary, high-confidence misclassifications. 

Standard empirical defenses (e.g., adversarial training) and empirical testing methods only prove the *presence* of vulnerabilities, not their *absence*. **Formal Verification** addresses this fundamental limitation by providing mathematical guarantees:

Given a classifier $f: \mathbb{R}^n \to \mathbb{R}^m$, a nominal input point $x_0$, a perturbation bound $\epsilon > 0$, and the correct ground truth class $y$:
$$\forall x \in \mathcal{B}_\infty(x_0, \epsilon) = \{x \mid \|x - x_0\|_\infty \le \epsilon\}, \quad \arg\max_{j} f_j(x) = y$$

MiniCert over-approximates neural network reachable sets using **Interval Bound Propagation (IBP)** and prepares convex relaxations for **Linear Programming Optimization**, enabling certified robustness guarantees.

---

## 2. Architecture & Repository Structure

The repository is organized into isolated, highly cohesive packages to facilitate concurrent, independent development between team members:

```
MiniCert/
├── models/                     # Neural network definitions, training, and parameter serialization
│   ├── __init__.py
│   ├── network.py              # MiniCertNet & modular FCNet PyTorch architectures
│   ├── train.py                # Standard training and evaluation routines
│   ├── train_mnist.py          # MNIST training CLI with checkpoint reload validation
│   └── model_loader.py         # Strict checkpointing and float64 mathematical extraction
├── abstract_domain/            # Abstract Interpretation domain representations
│   ├── __init__.py
│   ├── interval.py             # Scalar interval domain with extended-real bound policy
│   ├── box.py                  # Vector Box (hyper-rectangle) domain + sound IBP transformers
│   └── propagation.py          # Network-level Interval Bound Propagation (IBP)
├── verification/               # Robustness specification, LP relaxations, and verification engine
│   ├── __init__.py             # Verification contracts, status enums, and result types
│   ├── regions.py              # L-infinity input perturbation regions (InputRegion)
│   ├── safety.py               # Safety checking over abstract output bounds
│   └── pipeline.py             # End-to-end verify_robustness() orchestration
├── refinement/                 # Abstraction refinement and Branch-and-Bound (BaB) search
│   └── __init__.py
├── dashboard/                  # Streamlit-based interactive exploration UI
│   └── __init__.py
├── experiments/                # Empirical benchmarks, certified accuracy curves, and scalability studies
│   └── __init__.py
├── tests/                      # Automated unit, integration, and mathematical soundness tests
│   └── __init__.py
├── requirements.txt            # Minimal, pinned Python 3.11 project dependencies
├── README.md                   # Project documentation and architectural guide
└── .gitignore                  # Git ignore rules for Python, PyTorch, and IDE artifacts
```

### Module Responsibilities Breakdown

| Module | Primary Responsibility | Interface Deliverable |
| :--- | :--- | :--- |
| `models/` | PyTorch architectures with accessible linear layers and activation functions; strict checkpoint loading and float64 mathematical extraction. | `MiniCertNet`, `FCNet`, `extract_network()`, `extract_layer_parameters()` |
| `abstract_domain/` | Scalar `Interval` and vector `Box` domains with sound affine ($W^+ l + W^- u + b$) and ReLU transformers, plus network-level IBP. | `Interval`, `Box`, `propagate_network()` |
| `verification/` | L-infinity perturbation regions, safety checking, immutable result records, and end-to-end orchestration. | `InputRegion`, `SafetyChecker`, `VerificationResult`, `verify_robustness()` |
| `refinement/` | Branch-and-Bound search splitting uncertain neurons ($l_i < 0 < u_i$) or input intervals. | Splitting strategies & BaB search queue *(planned)* |
| `dashboard/` | Interactive Streamlit GUI for bound visualization, $\epsilon$-tuning, and adversarial exploration. | Interactive web app *(planned)* |
| `experiments/` | Benchmarking runtime vs. depth/width, certified accuracy vs. $\epsilon$. | Reproducible scripts & figures *(planned)* |
| `tests/` | Pytest suite validating mathematical invariants, arithmetic soundness, and model compatibility. | Test cases |

---

## 3. Team Roles & Division of Responsibilities

MiniCert is developed collaboratively by a team of two engineers:

### Developer 1: Neural Network & Abstract Interpretation Engineering
- **Focus Areas**: Deep Learning architectures and mathematical abstract domains.
- **Assigned Modules**:
  - `models/`: Implement verifiable feedforward networks, layer parameter extractors, training loops, and checkpoint loaders.
  - `abstract_domain/`: Implement the multi-dimensional Box/Interval domain, arithmetic operator overloads, affine transformations, and monotonic activation transformers.
  - Mathematical soundness proofs and interval unit tests.

### Developer 2: Verification, LP Optimization & Integration
- **Focus Areas**: Optimization-based verification algorithms, refinement, and interactive dashboarding.
- **Assigned Modules**:
  - `verification/`: Construct Linear Programming (LP) relaxations of piecewise linear activations (Planet relaxation), solve linear programs using `scipy.optimize.linprog`, and formulate safety specifications.
  - `refinement/`: Design Branch-and-Bound (BaB) splitting heuristics to tighten bounds when interval propagation is inconclusive (`UNKNOWN`).
  - `dashboard/`: Build the interactive Streamlit user interface.

### Shared Collaboration
- `experiments/`: Formulate benchmark test sets (e.g. synthetic data and MNIST), evaluate certified accuracy curves.
- `tests/`: Maintain end-to-end integration tests and property verification tests.

---

## 4. Mathematical Foundations

### 4.1 Box / Interval Abstract Domain
An interval element $\mathcal{X} \in \mathbb{IR}^n$ is defined by component-wise lower and upper bounds:
$$\mathcal{X} = [\mathbf{l}, \mathbf{u}] = \{ \mathbf{x} \in \mathbb{R}^n \mid \mathbf{l} \le \mathbf{x} \le \mathbf{u} \}$$

### 4.2 Sound Affine Transformer (Interval Bound Propagation)
For a linear layer $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$, the weight matrix is decomposed into non-negative and non-positive elements:
$$\mathbf{W}^+ = \max(\mathbf{W}, 0), \quad \mathbf{W}^- = \min(\mathbf{W}, 0)$$

The sound output interval $[\mathbf{l}_{out}, \mathbf{u}_{out}]$ is computed in closed form:
$$\mathbf{l}_{out} = \mathbf{W}^+ \mathbf{l} + \mathbf{W}^- \mathbf{u} + \mathbf{b}$$
$$\mathbf{u}_{out} = \mathbf{W}^+ \mathbf{u} + \mathbf{W}^- \mathbf{l} + \mathbf{b}$$

**Soundness Guarantee**:
$$\forall \mathbf{x} \in [\mathbf{l}, \mathbf{u}], \quad (\mathbf{W}\mathbf{x} + \mathbf{b}) \in [\mathbf{l}_{out}, \mathbf{u}_{out}]$$

### 4.3 Sound ReLU Transformer
Because $\text{ReLU}(z) = \max(0, z)$ is monotonically non-decreasing:
$$\mathbf{l}_{out} = \max(0, \mathbf{l})$$
$$\mathbf{u}_{out} = \max(0, \mathbf{u})$$

### 4.4 Network-Level Interval Bound Propagation
`propagate_network()` walks an extracted `MathematicalNetwork` layer by layer in exact execution order, applying the affine transformer to each `LinearLayer` and the ReLU transformer to each `ReLULayer`, producing a single certified output `Box` bounding the network's reachable set over the input region.

### 4.5 Safety Checking & Certification Margin
Given a target class $t$ and the certified output bounds, the `SafetyChecker` certifies strict class dominance over every competitor $j \neq t$:
$$\forall j \neq t: \quad \mathbf{l}_t > \mathbf{u}_j$$

The **verification margin** is defined as $\text{margin} = \mathbf{l}_t - \max_{j \neq t} \mathbf{u}_j$:
- `CERTIFIED` — margin is strictly positive.
- `INCONCLUSIVE` — margin is non-positive (bounds overlap); this never reports a violation, since overlapping bounds do not prove the network is unsafe.

**Note on soundness:** float64 bounds without outward (directed) rounding do not establish a machine-checked mathematical certificate. Certification is only as sound as the upstream bound computation.

---

## 5. Technical Requirements & Environment Setup

MiniCert requires **Python 3.11**. Follow the instructions below to create an isolated virtual environment and install dependencies.

### Step 1: Clone the Repository & Navigate to Workspace
```bash
cd MiniCert
```

### Step 2: Create a Python 3.11 Virtual Environment

- **On Windows (PowerShell / Command Prompt)**:
  ```powershell
  python -m venv venv
  ```
- **On macOS / Linux**:
  ```bash
  python3.11 -m venv venv
  ```

### Step 3: Activate the Virtual Environment

- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If execution of scripts is restricted in PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once).*

- **Windows (Command Prompt / cmd.exe)**:
  ```cmd
  .\venv\Scripts\activate.bat
  ```

- **macOS / Linux (Bash / Zsh)**:
  ```bash
  source venv/bin/activate
  ```

### Step 4: Upgrade Pip and Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Installation & Verification Commands

To confirm that the project environment is correctly configured and all core dependencies operate as expected, execute the following commands:

### 1. Verify Installed Package Versions
```bash
python -c "import torch, torchvision, numpy, scipy, matplotlib, streamlit, pytest; print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__); print('SciPy:', scipy.__version__); print('Environment is ready!')"
```

### 2. Smoke Test: Model Creation and Parameter Extraction
```bash
python -c "from models.network import MiniCertNet; from models.model_loader import extract_network; model = MiniCertNet(); net = extract_network(model); print('Extracted network:', net.input_features, '->', net.output_features, 'with', len(net.layers), 'layers.')"
```

### 3. Smoke Test: Scalar Interval Arithmetic
```bash
python -c "from abstract_domain import Interval; interval = Interval(-1.0, 2.0); print('Input Interval:', interval); print('After ReLU:', interval.relu()); print('Width:', interval.width)"
```

### 4. Smoke Test: Vector Box Domain & Interval Bound Propagation
```bash
python -c "import numpy as np; from abstract_domain.box import Box; box = Box(np.array([-1.0, 2.0]), np.array([3.0, 5.0])); out = box.affine(np.array([[1.5, -2.0]]), np.array([0.5])); print('Input Box:', box); print('After affine:', out)"
```

### 5. Smoke Test: End-to-End Robustness Verification
```bash
python -c "import numpy as np; from verification.pipeline import verify_robustness; from verification.regions import InputRegion; from models.model_loader import LinearLayer, MathematicalNetwork; w = np.array([[1.0, -1.0], [-1.0, 1.0]]); net = MathematicalNetwork(layers=(LinearLayer(0, w, np.zeros(2), 2, 2),), input_features=2, output_features=2, source_model='TwoClass'); result = verify_robustness(net, InputRegion(np.array([0.5, 0.1]), 0.05), target_class=0); print(result)"
```

### 6. Execute Test Suite
```bash
pytest tests/
```

---

## 7. Modular Development Roadmap

1. **Phase 1: Foundation & Abstract Domain (Done)**
   - Initialized modular architecture and development contracts.
   - Implemented `MiniCertNet` and generic `FCNet` model + strict layer extraction (`models/`).
   - Implemented scalar `Interval` and vector `Box` domains with sound affine and ReLU transformers (`abstract_domain/`).
   - Trained reference network to **96.44% MNIST test accuracy** and validated checkpoint reload.
2. **Phase 2: IBP Verifier Integration (Done)**
   - Implemented network-level Interval Bound Propagation (`propagate_network`).
   - Implemented `InputRegion` (L-infinity perturbation ball), `SafetyChecker`, and `VerificationResult`.
   - Implemented end-to-end `verify_robustness()` returning `CERTIFIED` / `INCONCLUSIVE`.
   - 337 automated tests passing.
3. **Phase 3: LP Relaxation & Refinement (Upcoming)**
   - Formulate Planet linear relaxation for unstable neurons ($l_i < 0 < u_i$).
   - Implement SciPy LP solver integration for margin optimization.
   - Implement Branch-and-Bound (BaB) splitting heuristics to resolve `INCONCLUSIVE` verdicts.
4. **Phase 4: Streamlit Dashboard & Empirical Evaluation**
   - Interactive GUI for real-time reachability exploration.
   - Comprehensive benchmark evaluation across perturbation budgets ($\epsilon$).

