"""
Aether Weighting Module

This module contains amplitude-weighting functions for different scattering mechanisms.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
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
        face_normals: Normal vectors for each face
        face_centers: Center coordinates for each face
        face_areas: Areas of each face
        config: Radar configuration
        
    Returns:
        Array of weights for each face
    """
    # Get positions as numpy arrays
    tx_pos = config.tx_pos_array
    rx_pos = config.rx_pos_array
    
    # Calculate vectors from faces to transmitter and receiver
    vectors_to_tx = tx_pos - face_centers
    vectors_to_rx = rx_pos - face_centers
    
    # Normalize these vectors
    tx_distances = np.linalg.norm(vectors_to_tx, axis=1)
    rx_distances = np.linalg.norm(vectors_to_rx, axis=1)
    
    vectors_to_tx_norm = vectors_to_tx / tx_distances[:, np.newaxis]
    vectors_to_rx_norm = vectors_to_rx / rx_distances[:, np.newaxis]
    
    # For specular reflection, the angle of incidence equals the angle of reflection
    # Calculate the bisector vector which should align with the normal for perfect reflection
    bisector_vectors = vectors_to_tx_norm + vectors_to_rx_norm
    bisector_norm = np.linalg.norm(bisector_vectors, axis=1)[:, np.newaxis]
    bisector_vectors_norm = bisector_vectors / bisector_norm
    
    # Calculate alignment between normals and bisector vectors (dot product)
    alignment_scores = np.abs(np.sum(face_normals * bisector_vectors_norm, axis=1))
    
    # Apply physical weighting - radar equation components:
    # - alignment squared (directivity)
    # - face area (scattering cross-section)
    # - 1/r² losses (two-way path)
    # - wavelength scaling
    total_distances = tx_distances + rx_distances
    wavelength = config.wavelength
    
    weights = (
        alignment_scores**2 *           # Alignment factor
        face_areas /                    # Proportional to area
        (total_distances**2) *          # Two-way path loss
        (4*np.pi / wavelength**2)       # Wavelength scaling
    )
    
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
        edge_vectors: Vector along each edge
        edge_centers: Center coordinates for each edge
        edge_lengths: Lengths of each edge
        config: Radar configuration

    Returns:
        Array of weights for each edge
    """
    # Get transmitter and receiver positions
    tx_pos = config.tx_pos_array
    rx_pos = config.rx_pos_array
    wavelength = config.wavelength

    # Calculate vectors from edges to transmitter and receiver
    vectors_to_tx = tx_pos - edge_centers
    vectors_to_rx = rx_pos - edge_centers

    # Calculate distances
    tx_distances = np.linalg.norm(vectors_to_tx, axis=1)
    rx_distances = np.linalg.norm(vectors_to_rx, axis=1)
    total_distances = tx_distances + rx_distances

    # Normalize edge vectors and incident/reflection vectors
    edge_vectors_norm = edge_vectors / np.linalg.norm(edge_vectors, axis=1)[:, np.newaxis]
    vectors_to_tx_norm = vectors_to_tx / tx_distances[:, np.newaxis]
    vectors_to_rx_norm = vectors_to_rx / rx_distances[:, np.newaxis]

    # Calculate angles between edge and incident/reflection vectors
    # sin(θ) = |a×b| where a and b are unit vectors
    sin_tx_angles = np.linalg.norm(np.cross(edge_vectors_norm, vectors_to_tx_norm), axis=1)
    sin_rx_angles = np.linalg.norm(np.cross(edge_vectors_norm, vectors_to_rx_norm), axis=1)

    # Edge diffraction is proportional to:
    # - Edge length (normalized by wavelength)
    # - sin(incident angle) * sin(reflection angle)
    # - Inversely proportional to total path distance squared
    # - Proportional to wavelength (longer wavelength = stronger diffraction)

    normalized_edge_lengths = edge_lengths / wavelength

    # PTD weighting formula (simplified)
    weights = (
            normalized_edge_lengths *  # Edge length factor
            sin_tx_angles * sin_rx_angles *  # Angle dependency
            (wavelength / (4 * np.pi)) *  # Wavelength factor
            (1 / total_distances ** 2)  # Two-way path loss
    )

    return weights


def compute_tip_weight(
        vertex_positions: np.ndarray,
        config: RadarConfig
) -> np.ndarray:
    """
    Compute tip diffraction weights based on corner diffraction theory.

    Args:
        vertex_positions: Position of each vertex
        config: Radar configuration

    Returns:
        Array of weights for each vertex
    """
    # Get transmitter and receiver positions
    tx_pos = config.tx_pos_array
    rx_pos = config.rx_pos_array
    wavelength = config.wavelength

    # Calculate vectors from vertices to transmitter and receiver
    vectors_to_tx = tx_pos - vertex_positions
    vectors_to_rx = rx_pos - vertex_positions

    # Calculate distances
    tx_distances = np.linalg.norm(vectors_to_tx, axis=1)
    rx_distances = np.linalg.norm(vectors_to_rx, axis=1)
    total_distances = tx_distances + rx_distances

    # Normalize vectors
    vectors_to_tx_norm = vectors_to_tx / tx_distances[:, np.newaxis]
    vectors_to_rx_norm = vectors_to_rx / rx_distances[:, np.newaxis]

    # Calculate angle between incident and scattered directions
    # cos(angle) = dot product of unit vectors
    cos_angles = np.sum(vectors_to_tx_norm * vectors_to_rx_norm, axis=1)
    scatter_angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))

    # Tip diffraction is:
    # - Stronger at higher frequencies (inversely proportional to wavelength)
    # - Depends on scattering angle (stronger when direct backscatter)
    # - Inversely proportional to total path distance squared

    angle_factor = 0.5 * (1 + np.cos(scatter_angles))  # Maximum at backscatter

    # Corner diffraction formula (simplified)
    weights = (
            (wavelength / (4 * np.pi)) ** 2 *  # Wavelength factor
            angle_factor *  # Angular dependence
            (1 / total_distances ** 2)  # Two-way path loss
    )

    return weights


def apply_polarization_factor(weights: np.ndarray, polarization: str = 'VV') -> np.ndarray:
    """
    Apply polarization-dependent scaling to weights.

    Args:
        weights: Input weights
        polarization: Polarization type ('VV', 'HH', 'HV', 'VH')
                      First letter: Transmit polarization
                      Second letter: Receive polarization
                      V = Vertical, H = Horizontal

    Returns:
        Modified weights
    """
    # Polarization factors based on empirical observations
    # For co-polarized cases (VV, HH), the response is typically stronger
    # For cross-polarized cases (HV, VH), the response is typically weaker

    if polarization == 'VV':
        # Vertical polarization typically gives strongest returns from vertical features
        factor = 1.0
    elif polarization == 'HH':
        # Horizontal polarization typically slightly weaker than VV for most targets
        factor = 0.9
    elif polarization in ['HV', 'VH']:
        # Cross-polarization typically much weaker
        factor = 0.25
    else:
        # Default case, no modification
        factor = 1.0

    return weights * factor