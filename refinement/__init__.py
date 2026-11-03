"""
MiniCert - Refinement Module
============================

This module handles abstraction refinement and Branch-and-Bound (BaB) algorithms
to tighten over-approximations when coarse abstract interpretation results in
an 'UNKNOWN' verdict.

Responsibilities:
- Input domain splitting: Partitioning the input perturbation box into smaller sub-boxes.
- Activation space splitting: Branching on ambiguous ReLU neurons (where l_i < 0 < u_i).
- Branch-and-Bound tree search orchestration.
- Counterexample-Guided Abstraction Refinement (CEGAR) workflows.

Collaboration:
- Developer 1 provides splitting primitives on the Interval domain.
- Developer 2 orchestrates the branch-and-bound verification queue and pruning.
"""

__all__ = []

