"""
Aether Extraction Module

This module contains specular/edge/tip detection logic for radar scattering analysis.
"""

import numpy as np
import trimesh
from typing import List, Tuple, Dict, Any
from aether.config import RadarConfig, ProcessingConfig
from aether.weight import compute_specular_weight, compute_edge_weight, compute_tip_weight
from aether.ranking import rank_scatterers

def detect_specular(mesh: trimesh.Trimesh, freq: float, tx: List[float], rx: List[float]) -> List[Dict[str, Any]]:
    """
    Detect specular reflection points on a 3D mesh based on given radar parameters.
    
    Args:
        mesh: A trimesh.Trimesh object representing the 3D model
        freq: Radar frequency in GHz
        tx: Transmitter position as [x, y, z] in meters
        rx: Receiver position as [x, y, z] in meters
    
    Returns:
        A list of dictionaries containing scatterer information:
        [
            {
                'position': [x, y, z],  # Center of the scatterer
                'normal': [nx, ny, nz],  # Surface normal
                'score': float,  # Relative intensity/RCS
                'type': 'specular',  # Type of scattering
                'face_idx': int,  # Index of the triangle in the mesh
            },
            ...
        ]
    """
    # Convert positions to numpy arrays
    tx_pos = np.array(tx)
    rx_pos = np.array(rx)
    
    # Calculate wavelength in meters from frequency in GHz
    # λ = c/f, where c is speed of light (3e8 m/s) and f is frequency in Hz
    wavelength = 3e8 / (freq * 1e9)
    
    # Get face centers and normals
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    
    # Calculate vectors from faces to transmitter and receiver
    vectors_to_tx = tx_pos - face_centers
    vectors_to_rx = rx_pos - face_centers
    
    # Normalize these vectors
    vectors_to_tx_norm = vectors_to_tx / np.linalg.norm(vectors_to_tx, axis=1)[:, np.newaxis]
    vectors_to_rx_norm = vectors_to_rx / np.linalg.norm(vectors_to_rx, axis=1)[:, np.newaxis]
    
    # For specular reflection, the angle of incidence equals the angle of reflection
    # Calculate the bisector vector which should align with the normal for perfect reflection
    bisector_vectors = vectors_to_tx_norm + vectors_to_rx_norm
    bisector_norm = np.linalg.norm(bisector_vectors, axis=1)[:, np.newaxis]
    bisector_vectors_norm = bisector_vectors / bisector_norm
    
    # Calculate alignment between normals and bisector vectors (dot product)
    alignment_scores = np.abs(np.sum(face_normals * bisector_vectors_norm, axis=1))
    
    # Calculate distances for amplitude scaling (using average distance)
    distances_tx = np.linalg.norm(vectors_to_tx, axis=1)
    distances_rx = np.linalg.norm(vectors_to_rx, axis=1)
    total_distances = distances_tx + distances_rx
    
    # Approximate RCS calculation - based on alignment and face area
    face_areas = mesh.area_faces
    
    # Calculate scores considering:
    # - alignment (better alignment = stronger reflection)
    # - face area (larger face = stronger reflection)
    # - distance (greater distance = weaker reflection)
    # - wavelength scaling
    scores = alignment_scores**2 * face_areas / (total_distances**2) * (4*np.pi / wavelength**2)
    
    # Sort faces by scores (highest first)
    sorted_indices = np.argsort(-scores)
    
    # Collect results
    scatterers = []
    for idx in sorted_indices[:min(100, len(sorted_indices))]:  # Limit to top 100 scatterers
        if scores[idx] > 0.01 * scores[sorted_indices[0]]:  # Filter out very weak scatterers
            scatterers.append({
                'position': face_centers[idx].tolist(),
                'normal': face_normals[idx].tolist(),
                'score': float(scores[idx]),
                'type': 'specular',
                'face_idx': int(idx)
            })
    
    return scatterers


def detect_edges(mesh: trimesh.Trimesh, config: RadarConfig) -> List[Dict[str, Any]]:
    """
    Detect edge diffraction points on a 3D mesh.
    
    Args:
        mesh: A trimesh.Trimesh object representing the 3D model
        config: Radar configuration
        
    Returns:
        A list of dictionaries containing edge scatterer information
    """
    # This is a placeholder for edge detection
    # In a full implementation, this would:
    # 1. Identify edges that could create diffraction
    # 2. Calculate edge vectors and centers
    # 3. Compute weights using the edge_weight function
    # 4. Rank and filter the results
    
    # For now, return an empty list
    return []


def detect_tips(mesh: trimesh.Trimesh, config: RadarConfig) -> List[Dict[str, Any]]:
    """
    Detect tip diffraction points on a 3D mesh.
    
    Args:
        mesh: A trimesh.Trimesh object representing the 3D model
        config: Radar configuration
        
    Returns:
        A list of dictionaries containing tip scatterer information
    """
    # This is a placeholder for tip detection
    # In a full implementation, this would:
    # 1. Identify vertices that could create tip diffraction
    # 2. Compute weights using the tip_weight function
    # 3. Rank and filter the results
    
    # For now, return an empty list
    return []


def extract_all_scatterers(
    mesh: trimesh.Trimesh,
    config: RadarConfig,
    proc_config: ProcessingConfig = None
) -> List[Dict[str, Any]]:
    """
    Extract all types of scatterers from a mesh based on configuration.
    
    Args:
        mesh: Input mesh
        config: Radar configuration
        proc_config: Processing configuration (optional)
        
    Returns:
        List of all scatterers (specular, edges, tips)
    """
    if proc_config is None:
        proc_config = ProcessingConfig()
    
    # Start with specular scatterers
    specular_scatterers = detect_specular(
        mesh,
        config.frequency_ghz,
        list(config.tx_position),
        list(config.rx_position)
    )
    
    scatterers = specular_scatterers
    
    # Add edge scatterers if enabled
    if proc_config.edge_detection:
        edge_scatterers = detect_edges(mesh, config)
        scatterers.extend(edge_scatterers)
    
    # Add tip scatterers if enabled
    if proc_config.tip_detection:
        tip_scatterers = detect_tips(mesh, config)
        scatterers.extend(tip_scatterers)
    
    # Sort all scatterers by score (descending)
    scatterers.sort(key=lambda s: s['score'], reverse=True)
    
    # Apply top-k and threshold filtering
    if len(scatterers) > proc_config.num_top_scatterers:
        scatterers = scatterers[:proc_config.num_top_scatterers]
    
    # Filter by minimum score if there are any scatterers
    if scatterers:
        max_score = max(s['score'] for s in scatterers)
        min_score = max_score * proc_config.min_score_threshold
        scatterers = [s for s in scatterers if s['score'] >= min_score]
    
    return scatterers
