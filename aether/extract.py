"""
Aether Extraction Module

This module contains specular/edge/tip detection logic for radar scattering analysis.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

__all__ = (
    "SpecularParams",
    "EdgeParams",
    "TipParams",
    "detect_specular",
    "detect_edges",
    "detect_tips",
)


@dataclass(frozen=True)
class SpecularParams:
    """Parameters for specular detection using the bisector test.

    threshold_cos: accept faces where |n·b| > threshold_cos, where b is the
    unit bisector between incident (tx→face) and reflection (rx→face) directions.
    """

    threshold_cos: float = 0.985  # ~10° cone → cos ~ 0.985


@dataclass(frozen=True)
class EdgeParams:
    """Parameters for edge detection via dihedral angle (in radians)."""

    dihedral_min: float = np.deg2rad(25.0)  # treat edges sharper than this as diffractive


@dataclass(frozen=True)
class TipParams:
    """Parameters for tip detection via discrete Gaussian curvature (radians)."""

    curvature_min: float = np.deg2rad(120.0)  # large positive curvature ⇒ candidate tip


# ---------- helpers ----------

def _as_unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return v / n


def _require_attrs(mesh, *names):
    for n in names:
        if not hasattr(mesh, n):
            raise AttributeError(f"mesh is missing attribute {n!r}")


# ---------- specular ----------

def detect_specular(
    *,
    face_centroids: np.ndarray,
    face_normals: np.ndarray,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    params: Optional[SpecularParams] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect specular faces.

    Parameters
    ----------
    face_centroids : (F,3)
    face_normals   : (F,3) unit normals (outward)
    tx_pos, rx_pos : (3,) transmitter / receiver positions
    params         : SpecularParams

    Returns
    -------
    mask : (F,) bool
        Faces passing the specular bisector test.
    alignment : (F,) float
        |n·b| score in [0,1], useful for ranking/specular weighting.
    """
    p = params or SpecularParams()
    C = np.asarray(face_centroids, dtype=float)
    N = _as_unit(face_normals)
    tx = np.asarray(tx_pos, dtype=float)
    rx = np.asarray(rx_pos, dtype=float)

    v_tx = _as_unit(tx - C)  # direction from face → tx
    v_rx = _as_unit(rx - C)  # direction from face → rx
    b = _as_unit(v_tx + v_rx)  # bisector
    alignment = np.abs(np.einsum("ij,ij->i", N, b))
    mask = alignment > float(p.threshold_cos)
    return mask, alignment


# ---------- edges ----------

def detect_edges(
    *,
    edges_face: np.ndarray,
    face_normals: np.ndarray,
    params: Optional[EdgeParams] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect diffractive (sharp) edges using dihedral angle between adjacent faces.

    Parameters
    ----------
    edges_face : (E,2) int
        For each edge, the indices of its two incident faces (use -1 for boundary).
    face_normals : (F,3) unit normals
    params : EdgeParams

    Returns
    -------
    edge_mask : (E,) bool
        True for edges considered diffractive (sharp).
    dihedral : (E,) float
        Dihedral angle in radians for each edge (NaN for boundary edges).
    """
    p = params or EdgeParams()

    EF = np.asarray(edges_face, dtype=int)
    N = _as_unit(face_normals)

    E = EF.shape[0]
    dihedral = np.full(E, np.nan, dtype=float)

    valid = (EF[:, 0] >= 0) & (EF[:, 1] >= 0)
    f1 = EF[valid, 0]
    f2 = EF[valid, 1]

    cos_t = np.einsum("ij,ij->i", N[f1], N[f2])
    cos_t = np.clip(cos_t, -1.0, 1.0)
    theta = np.arccos(cos_t)

    dihedral[valid] = theta
    edge_mask = np.zeros(E, dtype=bool)
    edge_mask[valid] = theta >= float(p.dihedral_min)

    return edge_mask, dihedral


# ---------- tips ----------

def detect_tips(
    *,
    vertex_curvature: np.ndarray,
    params: Optional[TipParams] = None,
) -> np.ndarray:
    """Detect tip (corner) vertices by thresholding discrete Gaussian curvature.

    Parameters
    ----------
    vertex_curvature : (V,) float
        Discrete Gaussian curvature estimate at each vertex (in radians).
        (Compute upstream via angle defect: 2π - sum(incident face angles).)
    params : TipParams

    Returns
    -------
    tip_mask : (V,) bool
        True where curvature >= curvature_min.
    """
    p = params or TipParams()
    k = np.asarray(vertex_curvature, dtype=float)
    return k >= float(p.curvature_min)
