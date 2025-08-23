"""
Aether Preprocessing Module

This module handles scale normalization, mesh quality checks, and other preprocessing steps.
"""

import numpy as np
import trimesh
from typing import Tuple, Dict, Any


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Normalize the mesh to a standard scale and center it at the origin.
    
    Args:
        mesh: Input mesh
        
    Returns:
        Normalized mesh
    """
    # Center the mesh at the origin
    mesh.apply_translation(-mesh.centroid)

    # Scale the mesh to fit within a unit sphere
    scale = 1.0 / np.max(mesh.extents)
    mesh.apply_scale(scale)

    return mesh


def check_mesh_quality(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """
    Check the mesh for potential quality issues.
    
    Args:
        mesh: Input mesh
        
    Returns:
        Dictionary with quality metrics and issue flags:
        {
            'vertex_count': int,
            'face_count': int,
            'is_watertight': bool,
            'has_duplicate_vertices': bool,
            'has_degenerate_faces': bool,
        }
    """
    quality_report = {
        'vertex_count': len(mesh.vertices),
        'face_count': len(mesh.faces),
        'is_watertight': mesh.is_watertight,
        'has_duplicate_vertices': np.unique(mesh.vertices, axis=0).shape[0] != mesh.vertices.shape[0],
        'has_degenerate_faces': np.any(
            (mesh.faces[:, 0] == mesh.faces[:, 1]) |
            (mesh.faces[:, 1] == mesh.faces[:, 2]) |
            (mesh.faces[:, 2] == mesh.faces[:, 0])
        ),
    }
    return quality_report


def prepare_mesh(mesh: trimesh.Trimesh, normalize: bool = True) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    Prepare a mesh for analysis by checking quality and optionally normalizing.
    
    Args:
        mesh: Input mesh
        normalize: Whether to normalize the mesh
        
    Returns:
        Tuple of (prepared_mesh, quality_report)
    """
    quality_report = check_mesh_quality(mesh)

    if normalize:
        mesh = normalize_mesh(mesh)

    return mesh, quality_report
