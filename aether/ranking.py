"""
Aether Ranking Module

This module handles top-k selection and threshold filtering of scatterers.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import trimesh
from aether.config import ProcessingConfig


def rank_scatterers(
    mesh: trimesh.Trimesh,
    weights: np.ndarray,
    config: ProcessingConfig
) -> List[Dict[str, Any]]:
    """
    Rank and filter scatterers based on their weights.
    
    Args:
        mesh: Input mesh
        weights: Array of weights for each face
        config: Processing configuration
        
    Returns:
        List of dictionaries with scatterer information
    """
    # Sort faces by weight (descending order)
    sorted_indices = np.argsort(-weights)
    
    # Get the maximum weight for normalization
    max_weight = weights.max() if len(weights) > 0 else 1.0
    
    # Filter and collect results
    scatterers = []
    for i, idx in enumerate(sorted_indices):
        # Stop if we've collected enough scatterers
        if i >= config.num_top_scatterers:
            break
            
        # Stop if the weight is below threshold
        if weights[idx] < config.min_score_threshold * max_weight:
            break
            
        # Add this scatterer
        scatterers.append({
            'position': mesh.triangles_center[idx].tolist(),
            'normal': mesh.face_normals[idx].tolist(),
            'score': float(weights[idx]),
            'type': 'specular',
            'face_idx': int(idx)
        })
    
    return scatterers


def top_k_scatterers(
    weights: np.ndarray, 
    k: int,
    indices: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the top k elements by weight.
    
    Args:
        weights: Array of weights
        k: Number of top elements to return
        indices: Optional array of original indices
        
    Returns:
        Tuple of (top_k_weights, top_k_indices)
    """
    if indices is None:
        indices = np.arange(len(weights))
        
    # Get sorted order (descending)
    sort_idx = np.argsort(-weights)
    
    # Get top k
    k = min(k, len(weights))
    top_k_idx = sort_idx[:k]
    
    return weights[top_k_idx], indices[top_k_idx]


def threshold_scatterers(
    weights: np.ndarray,
    threshold: float,
    indices: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter scatterers by a threshold value.
    
    Args:
        weights: Array of weights
        threshold: Threshold value (absolute)
        indices: Optional array of original indices
        
    Returns:
        Tuple of (filtered_weights, filtered_indices)
    """
    if indices is None:
        indices = np.arange(len(weights))
        
    # Create mask for weights above threshold
    mask = weights >= threshold
    
    return weights[mask], indices[mask]


def relative_threshold_scatterers(
    weights: np.ndarray,
    relative_threshold: float,
    indices: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter scatterers by a threshold relative to maximum weight.
    
    Args:
        weights: Array of weights
        relative_threshold: Threshold value (relative to maximum)
        indices: Optional array of original indices
        
    Returns:
        Tuple of (filtered_weights, filtered_indices)
    """
    if len(weights) == 0:
        return np.array([]), np.array([])
        
    max_weight = weights.max()
    absolute_threshold = max_weight * relative_threshold
    
    return threshold_scatterers(weights, absolute_threshold, indices)
