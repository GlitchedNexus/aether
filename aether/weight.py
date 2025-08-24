"""
Aether Weighting Module

This module contains amplitude-weighting functions for different scattering mechanisms.
"""

import numpy as np
from aether.config import RadarConfig


def compute_specular_weight(
    face_normals: np.ndarray,
    face_centers: np.ndarray,
    face_areas: np.ndarray,
    config: RadarConfig
) -> np.ndarray:
    """
    Compute specular reflection weights for mesh faces.

    Args:
        face_normals: Normal vectors for each face [N, 3]
        face_centers: Center coordinates for each face [N, 3]
        face_areas: Areas of each face [N,]
        config: Radar configuration

    Returns:
        Array of weights for each face [N,]
    """
    # Compute direction vectors from object to Tx and Rx
    tx_vec = config.tx_pos_array - face_centers  # [N, 3]
    rx_vec = config.rx_pos_array - face_centers  # [N, 3]
    tx_dir = tx_vec / np.linalg.norm(tx_vec, axis=1, keepdims=True)
    rx_dir = rx_vec / np.linalg.norm(rx_vec, axis=1, keepdims=True)
    # Compute bisector direction
    bisector = tx_dir + rx_dir
    bisector /= np.linalg.norm(bisector, axis=1, keepdims=True)
    # Cosine of angle between face normal and bisector
    cos_theta = np.clip(np.sum(face_normals * bisector, axis=1), 0, 1)
    # Weight: area * (cos_theta)^2 for sharper specular
    weights = face_areas * (cos_theta ** 2)
    return weights


def compute_edge_weight(
    edge_vectors: np.ndarray,
    edge_centers: np.ndarray,
    edge_lengths: np.ndarray,
    config: RadarConfig
) -> np.ndarray:
    """
    Compute edge diffraction weights based on Physical Theory of Diffraction (PTD).

    Args:
        edge_vectors: Vector along each edge [N, 3]
        edge_centers: Center coordinates for each edge [N, 3]
        edge_lengths: Lengths of each edge [N,]
        config: Radar configuration

    Returns:
        Array of weights for each edge [N,]
    """
    # Compute direction vectors from object to Tx and Rx
    tx_vec = config.tx_pos_array - edge_centers  # [N, 3]
    rx_vec = config.rx_pos_array - edge_centers  # [N, 3]
    tx_dir = tx_vec / np.linalg.norm(tx_vec, axis=1, keepdims=True)
    rx_dir = rx_vec / np.linalg.norm(rx_vec, axis=1, keepdims=True)
    # Edge direction unit vectors
    edge_dirs = edge_vectors / np.linalg.norm(edge_vectors, axis=1, keepdims=True)
    # Compute bisector direction
    bisector = tx_dir + rx_dir
    bisector /= np.linalg.norm(bisector, axis=1, keepdims=True)
    cos_theta = np.abs(np.sum(edge_dirs * bisector, axis=1))
    # PTD: weight ~ length * (cos_theta) / sqrt(wavelength)
    weights = edge_lengths * cos_theta / np.sqrt(config.wavelength)
    return weights


def compute_tip_weight(
    vertex_positions: np.ndarray,
    config: RadarConfig
) -> np.ndarray:
    """
    Compute tip diffraction weights based on corner diffraction theory.

    Args:
        vertex_positions: Position of each vertex [N, 3]
        config: Radar configuration

    Returns:
        Array of weights for each vertex [N,]
    """
    # Compute direction vectors from object to Tx and Rx
    tx_vec = config.tx_pos_array - vertex_positions  # [N, 3]
    rx_vec = config.rx_pos_array - vertex_positions  # [N, 3]
    tx_dir = tx_vec / np.linalg.norm(tx_vec, axis=1, keepdims=True)
    rx_dir = rx_vec / np.linalg.norm(rx_vec, axis=1, keepdims=True)
    # Normalize vertex position vectors (relative to object center)
    vertex_dirs = vertex_positions / np.linalg.norm(vertex_positions, axis=1, keepdims=True)
    cos_tx = np.abs(np.sum(vertex_dirs * tx_dir, axis=1))
    cos_rx = np.abs(np.sum(vertex_dirs * rx_dir, axis=1))
    # Corner diffraction: weight ~ (cos_tx * cos_rx) / wavelength
    weights = (cos_tx * cos_rx) / config.wavelength
    return weights


def apply_polarization_factor(weights: np.ndarray, polarization: str = 'VV') -> np.ndarray:
    """
    Apply polarization-dependent scaling to weights.

    Args:
        weights: Input weights [N,]
        polarization: Polarization type ('VV', 'HH', 'HV', 'VH')
                      First letter: Transmit polarization
                      Second letter: Receive polarization
                      V = Vertical, H = Horizontal

    Returns:
        Modified weights [N,]
    """
    # Example factors, can be adjusted based on actual radar config
    factors = {
        'VV': 1.0,
        'HH': 0.9,
        'HV': 0.5,
        'VH': 0.5,
    }
    factor = factors.get(polarization.upper(), 1.0)
    return weights * factor