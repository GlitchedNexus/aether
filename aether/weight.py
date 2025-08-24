"""
Aether Weighting Module

This module contains amplitude-weighting functions for different scattering mechanisms.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

__all__ = (
    "WeightParams",
    "apply_polarization_factor",
    "combine_weights",
)

Pol = Literal["VV", "HH", "HV", "VH"]


@dataclass(frozen=True)
class WeightParams:
    """Parameters controlling amplitude combination and polarization.

    The component coefficients scale each mechanism before combination.
    """

    c_specular: float = 1.0
    c_edge: float = 1.0
    c_tip: float = 1.0
    polarization: Pol = "VV"


_POL_FACTORS = {
    "VV": 1.0,
    "HH": 0.8,
    "HV": 0.2,
    "VH": 0.2,
}


def apply_polarization_factor(weights: np.ndarray, pol: Pol = "VV") -> np.ndarray:
    """Apply simple polarization scaling (placeholder for full scattering matrix)."""
    f = float(_POL_FACTORS.get(pol, 1.0))
    return np.asarray(weights, dtype=float) * f


def combine_weights(
    *,
    specular: Optional[np.ndarray] = None,
    edge: Optional[np.ndarray] = None,
    tip: Optional[np.ndarray] = None,
    params: Optional[WeightParams] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Combine mechanism-specific scores into a single non-negative weight vector.

    All provided arrays must be the same length; missing components are treated as 0.

    - Linear combination with coefficients from params (default 1.0 each).
    - Optional min-max normalization to [0,1] for downstream ranking.
    - Polarization factor applied at the end.
    """
    p = params or WeightParams()

    # Determine length
    arrays = [a for a in (specular, edge, tip) if a is not None]
    if not arrays:
        return np.zeros(0, dtype=float)
    n = arrays[0].shape[0]
    if any(a.shape[0] != n for a in arrays):
        raise ValueError("All component arrays must have the same length")

    S = np.zeros(n, dtype=float)
    if specular is not None:
        S += p.c_specular * np.asarray(specular, dtype=float)
    if edge is not None:
        S += p.c_edge * np.asarray(edge, dtype=float)
    if tip is not None:
        S += p.c_tip * np.asarray(tip, dtype=float)

    S = np.maximum(S, 0.0)

    if normalize and S.size:
        s_min = float(np.nanmin(S))
        s_max = float(np.nanmax(S))
        if s_max > s_min:
            S = (S - s_min) / (s_max - s_min)
        else:
            S[:] = 1.0  # constant vector → treat as equally strong

    S = apply_polarization_factor(S, p.polarization)
    return S
