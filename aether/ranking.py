"""
Aether Ranking Module

This module handles top-k selection and threshold filtering of scatterers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np

__all__ = (
    "Method",
    "RankingParams",
    "normalize_weights",
    "topk_indices",
    "threshold_mask_abs",
    "threshold_mask_rel",
    "rank_scatterers",
)

Method = Literal["top_k", "abs_threshold", "rel_threshold"]


@dataclass(frozen=True)
class RankingParams:
    """
    Configuration for scatterer ranking.

    Attributes
    ----------
    method : {"top_k", "abs_threshold", "rel_threshold"}
        Ranking strategy.
    k : int, optional
        Number of items to select for the "top_k" method (default: 50).
    abs_threshold : float, optional
        Absolute threshold value for the "abs_threshold" method.
    rel_alpha : float, optional
        Relative fraction (0..1] of the maximum weight for the "rel_threshold" method.
        For example, rel_alpha=0.2 selects all items with weight > 0.2 * max(weight).
    normalize : bool
        If True, normalize weights to [0, 1] before thresholding (does not affect "top_k").
    stable : bool
        If True, break ties using a stable order (original index ascending).
    """

    method: Method = "top_k"
    k: int = 50
    abs_threshold: Optional[float] = None
    rel_alpha: float = 0.2
    normalize: bool = True
    stable: bool = True

    def validate(self) -> None:
        if self.method not in ("top_k", "abs_threshold", "rel_threshold"):
            raise ValueError(f"Unknown method: {self.method!r}")
        if self.method == "top_k":
            if not isinstance(self.k, int) or self.k <= 0:
                raise ValueError("k must be a positive integer for method='top_k'")
        if self.method == "abs_threshold":
            if self.abs_threshold is None or not np.isfinite(self.abs_threshold):
                raise ValueError("abs_threshold must be a finite float for method='abs_threshold'")
        if self.method == "rel_threshold":
            if not (0.0 < self.rel_alpha <= 1.0) or not np.isfinite(self.rel_alpha):
                raise ValueError("rel_alpha must be in (0, 1] for method='rel_threshold'")


def _ensure_1d(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        w = np.ravel(w)
    return w


def normalize_weights(weights: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize weights to [0, 1] robustly, ignoring NaNs and handling constant arrays.

    Parameters
    ----------
    weights : ndarray, shape (N,)
        Raw weights (e.g., RCS-derived amplitudes or powers).
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    ndarray
        Normalized weights in [0, 1] with NaNs replaced by 0.
    """
    w = _ensure_1d(weights).copy()
    finite_mask = np.isfinite(w)
    if not np.any(finite_mask):
        return np.zeros_like(w)

    w_finite = w[finite_mask]
    w_min = np.min(w_finite)
    w_max = np.max(w_finite)

    if np.isclose(w_max, w_min, rtol=0.0, atol=eps):
        normalized = np.zeros_like(w)
        normalized[finite_mask] = 1.0
        return normalized

    normalized = np.zeros_like(w)
    normalized[finite_mask] = (w_finite - w_min) / (w_max - w_min + eps)
    normalized[~finite_mask] = 0.0
    return np.clip(normalized, 0.0, 1.0)


def topk_indices(weights: np.ndarray, k: int, *, stable: bool = True) -> np.ndarray:
    """
    Return indices of the top-k largest weights (descending), with optional stable tie-breaking.

    Parameters
    ----------
    weights : ndarray, shape (N,)
        Weights to rank.
    k : int
        Number of indices to return. If k >= N, returns all indices sorted by weight.
    stable : bool
        If True, break ties by original index (ascending).

    Returns
    -------
    ndarray of dtype int
        Indices of the selected items, sorted by weight descending.
    """
    w = _ensure_1d(weights)
    n = w.size
    if n == 0:
        return np.empty(0, dtype=int)

    k = int(max(0, min(k, n)))
    if k == 0:
        return np.empty(0, dtype=int)

    if 0 < k < n:
        part = np.argpartition(-w, k - 1)[:k]
        if stable:
            order = np.lexsort((part, -w[part]))
            selected = part[order]
        else:
            selected = part[np.argsort(-w[part], kind="quicksort")]
    else:
        idx = np.arange(n)
        if stable:
            order = np.lexsort((idx, -w))
            selected = idx[order]
        else:
            selected = np.argsort(-w, kind="quicksort")

    return selected


def threshold_mask_abs(
    weights: np.ndarray, threshold: float, *, normalize: bool = True
) -> np.ndarray:
    """
    Boolean mask selecting items whose weight exceeds an absolute threshold.

    Parameters
    ----------
    weights : ndarray
        Input weights.
    threshold : float
        Absolute threshold to apply (if `normalize=True`, this is in [0, 1]).
    normalize : bool
        If True, normalize `weights` to [0, 1] before thresholding.

    Returns
    -------
    ndarray of dtype bool
        Mask of selected items.
    """
    w = normalize_weights(weights) if normalize else _ensure_1d(weights)
    thr = float(threshold)
    if not np.isfinite(thr):
        raise ValueError("threshold must be a finite float")
    return w > thr


def threshold_mask_rel(weights: np.ndarray, alpha: float) -> np.ndarray:
    """
    Boolean mask selecting items whose weight exceeds `alpha * max(weights)`.

    Parameters
    ----------
    weights : ndarray
        Input weights.
    alpha : float
        Relative factor in (0, 1].

    Returns
    -------
    ndarray of dtype bool
        Mask of selected items.
    """
    w = _ensure_1d(weights)
    if not (0.0 < alpha <= 1.0) or not np.isfinite(alpha):
        raise ValueError("alpha must be in (0, 1]")
    if w.size == 0:
        return np.zeros(0, dtype=bool)

    finite_mask = np.isfinite(w)
    if not np.any(finite_mask):
        return np.zeros_like(w, dtype=bool)

    max_w = np.max(w[finite_mask])
    return np.where(finite_mask, w > alpha * max_w, False)


def rank_scatterers(
    weights: np.ndarray,
    params: Optional[RankingParams] = None,
    *,
    method: Optional[Method] = None,
    k: Optional[int] = None,
    abs_threshold: Optional[float] = None,
    rel_alpha: Optional[float] = None,
    normalize: Optional[bool] = None,
    stable: Optional[bool] = None,
    return_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Rank and select scatterers according to the chosen strategy.

    This function is the canonical entry point used by the pipeline
    (extract.py → weight.py → ranking.py → export.py). It returns
    indices of selected items (and optionally a boolean mask).

    Parameters
    ----------
    weights : ndarray, shape (N,)
        Weights to rank (e.g., scalar contributions from `weight.py`).
    params : RankingParams, optional
        Preconfigured parameters. Individual keyword arguments override fields from `params`.
    method : {"top_k", "abs_threshold", "rel_threshold"}, optional
        Ranking strategy. Overrides `params.method` if provided.
    k : int, optional
        Number of items for top-k selection. Overrides `params.k`.
    abs_threshold : float, optional
        Absolute threshold for "abs_threshold" strategy. Overrides `params.abs_threshold`.
    rel_alpha : float, optional
        Relative factor for "rel_threshold" strategy. Overrides `params.rel_alpha`.
    normalize : bool, optional
        Normalize before absolute thresholding. Overrides `params.normalize`.
    stable : bool, optional
        Stable tie-breaking. Overrides `params.stable`.
    return_mask : bool
        If True, also return a boolean mask the same length as `weights` indicating selection.

    Returns
    -------
    selected : ndarray of dtype int
        Indices of selected items sorted by descending weight (top first).
    mask : ndarray of dtype bool (optional)
        Selection mask aligned with `weights`.

    Examples
    --------
    >>> import numpy as np
    >>> w = np.array([0.1, 0.7, 0.3, 0.7])
    >>> idx = rank_scatterers(w, method="top_k", k=2)
    >>> idx
    array([1, 3])  # two largest (ties resolved stably by index)
    >>> idx, mask = rank_scatterers(w, method="rel_threshold", rel_alpha=0.9, return_mask=True)
    >>> idx, mask
    (array([1, 3]), array([False,  True, False,  True]))
    """
    p = params or RankingParams()
    method = method if method is not None else p.method
    k = int(k if k is not None else p.k)
    abs_threshold = abs_threshold if abs_threshold is not None else p.abs_threshold
    rel_alpha = float(rel_alpha if rel_alpha is not None else p.rel_alpha)
    normalize = bool(normalize if normalize is not None else p.normalize)
    stable = bool(stable if stable is not None else p.stable)

    tmp = RankingParams(
        method=method,
        k=k,
        abs_threshold=abs_threshold,
        rel_alpha=rel_alpha,
        normalize=normalize,
        stable=stable,
    )
    tmp.validate()

    w = _ensure_1d(weights)
    n = w.size
    if n == 0:
        empty_idx = np.empty(0, dtype=int)
        empty_mask = np.zeros(0, dtype=bool)
        return (empty_idx, empty_mask) if return_mask else empty_idx

    if method == "top_k":
        idx = topk_indices(w, k, stable=stable)
        if return_mask:
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            return idx, mask
        return idx

    if method == "abs_threshold":
        assert abs_threshold is not None
        mask = threshold_mask_abs(w, abs_threshold, normalize=normalize)
        if not np.any(mask):
            idx = topk_indices(w, 1, stable=stable)
            return (idx, mask) if return_mask else idx
        selected = np.flatnonzero(mask)
        order = np.argsort(-w[selected], kind="quicksort")
        idx = selected[order]
        return (idx, mask) if return_mask else idx

    mask = threshold_mask_rel(w, rel_alpha)
    if not np.any(mask):
        idx = topk_indices(w, 1, stable=stable)
        return (idx, mask) if return_mask else idx

    selected = np.flatnonzero(mask)
    order = np.argsort(-w[selected], kind="quicksort")
    idx = selected[order]
    return (idx, mask) if return_mask else idx
