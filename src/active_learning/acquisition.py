"""Acquisition functions for active learning.

Each function maps (mu, sigma) → scores array; higher score = higher priority.
UCB trades off predicted value against uncertainty; explore is pure uncertainty;
random provides the baseline.
"""
from __future__ import annotations

import numpy as np


def ucb(mu: np.ndarray, sigma: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    """Upper Confidence Bound: balance predicted value and uncertainty."""
    return mu + kappa * sigma


def explore(sigma: np.ndarray) -> np.ndarray:
    """Pure exploration: select the most uncertain compounds."""
    return sigma.copy()


def random_scores(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random scores — the baseline to beat."""
    return rng.uniform(size=n)
