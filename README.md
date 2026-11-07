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
│   ├── network.py              # Modular fully-connected PyTorch architectures (FCNet)
│   ├── train.py                # Standard training and evaluation routines
│   └── model_loader.py         # Checkpointing and weight/bias matrix extraction
├── abstract_domain/            # Abstract Interpretation domain representations
│   ├── __init__.py
│   └── interval.py             # Multi-dimensional Box/Interval domain and sound transformers
├── verification/               # Robustness specification, LP relaxations, and verification engine
│   └── __init__.py             # Verification contracts, status enums, and result types
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
| `models/` | PyTorch architectures with accessible linear layers and activation functions; weight/bias extraction. | `FCNet`, `extract_layer_parameters()` |
| `abstract_domain/` | Mathematical Box domain $[l, u]$ with sound affine ($W^+ l + W^- u + b$) and ReLU transformers. | `Interval` class |
| `verification/` | LP formulations (SciPy `linprog`), property encoding, and verification status outcomes. | `VerificationResult`, `VerificationStatus` |
| `refinement/` | Branch-and-Bound search splitting uncertain neurons ($l_i < 0 < u_i$) or input intervals. | Splitting strategies & BaB search queue |
| `dashboard/` | Interactive Streamlit GUI for bound visualization, $\epsilon$-tuning, and adversarial exploration. | Interactive web app |
| `experiments/` | Benchmarking runtime vs. depth/width, certified accuracy vs. $\epsilon$. | Reproducible scripts & figures |
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
python -c "from models import FCNet, extract_layer_parameters; model = FCNet(input_dim=4, hidden_dims=[16, 8], output_dim=3); params = extract_layer_parameters(model); print('FCNet initialized with', len(params), 'linear layers.'); print('Layer 0 weight shape:', params[0][0].shape)"
```

### 3. Smoke Test: Scalar Interval Arithmetic
```bash
python -c "from abstract_domain import Interval; interval = Interval(-1.0, 2.0); print('Input Interval:', interval); print('After ReLU:', interval.relu()); print('Width:', interval.width)"
```

### 4. Execute Test Suite
```bash
pytest tests/
```

---

## 7. Modular Development Roadmap

1. **Phase 1: Foundation & Abstract Domain (Current)**
   - Initialized modular architecture and development contracts.
   - Implemented `FCNet` model and layer extraction utilities (`models/`).
   - Implemented Box domain with sound affine and ReLU transformers (`abstract_domain/`).
2. **Phase 2: LP Relaxation & Verifier Integration (Upcoming - Developer 2)**
   - Formulate Planet linear relaxation for unstable neurons ($l_i < 0 < u_i$).
   - Implement SciPy LP solver integration for margin optimization.
3. **Phase 3: Refinement & Branch-and-Bound**
   - Implement bisection strategies for ambiguous neurons to resolve `UNKNOWN` verdicts.
4. **Phase 4: Streamlit Dashboard & Empirical Evaluation**
   - Interactive GUI for real-time reachability exploration.
   - Comprehensive benchmark evaluation across perturbation budgets ($\epsilon$).

