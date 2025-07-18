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
    # Create a copy to avoid modifying the original
    normalized = mesh.copy()
    
    # Center at origin
    normalized.vertices -= normalized.centroid
    
    # Scale to fit in a unit cube
    scale_factor = 1.0 / max(normalized.extents)
    normalized.vertices *= scale_factor
    
    return normalized


def check_mesh_quality(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """
    Check the mesh for potential quality issues.
    
    Args:
        mesh: Input mesh
        
    Returns:
        Dictionary with quality metrics and issue flags
    """
    quality = {
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "is_watertight": mesh.is_watertight,
        "has_duplicate_vertices": False,  # Will be checked
        "has_degenerate_faces": False,    # Will be checked
    }
    
    # Check for duplicate vertices (simplified)
    # In a production environment, use mesh.remove_duplicate_vertices()
    quality["has_duplicate_vertices"] = len(mesh.vertices) > len(np.unique(mesh.vertices, axis=0))
    
    # Check for degenerate faces (very small area)
    areas = mesh.area_faces
    quality["has_degenerate_faces"] = np.any(areas < 1e-10)
    
    return quality


def prepare_mesh(mesh: trimesh.Trimesh, normalize: bool = True) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    Prepare a mesh for analysis by checking quality and optionally normalizing.
    
    Args:
        mesh: Input mesh
        normalize: Whether to normalize the mesh
        
    Returns:
        Tuple of (prepared_mesh, quality_report)
    """
    # Check quality first
    quality_report = check_mesh_quality(mesh)
    
    # Create a working copy
    prepared = mesh.copy()
    
    # Fix issues if needed
    if quality_report["has_duplicate_vertices"]:
        prepared.merge_vertices()
        print("Warning: Duplicate vertices were merged")
    
    # Remove degenerate faces if needed
    if quality_report["has_degenerate_faces"]:
        mask = prepared.area_faces > 1e-10
        prepared.update_faces(mask)
        print("Warning: Degenerate faces were removed")
    
    # Normalize if requested
    if normalize:
        prepared = normalize_mesh(prepared)
    
    return prepared, quality_report
