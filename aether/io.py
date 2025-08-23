"""
Aether IO Module

This module provides mesh loading and cleaning routines.
"""

import trimesh
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional
import scipy

def calculate_face_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Calculate face normals if they're not provided by trimesh.
    
    Args:
        mesh: Input mesh
        
    Returns:
        Array of face normal vectors
    """

    if mesh.face_normals is None:
        # Trimesh usually computes face_normals automatically, but if not,
        # calculate them manually using the vertices of each face.
        faces = mesh.faces
        vertices = mesh.vertices

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        normals = np.cross(v1 - v0, v2 - v0)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        mesh.face_normals = normals

    return mesh.face_normals


def calculate_face_reflection_direction(
    face_normals: np.ndarray,
    incident_direction: np.ndarray
) -> np.ndarray:
    """
    Calculate reflection direction from face normals using perfect reflection law.
    
    Args:
        face_normals: Array of face normal vectors [N, 3]
        incident_direction: Direction of incident ray [3,] or [N, 3]
        
    Returns:
        Array of reflection directions [N, 3]
    """

    incident_direction = incident_direction / np.linalg.norm(incident_direction, axis=-1, keepdims=True)
    reflection_directions = incident_direction - 2 * np.sum(incident_direction * face_normals, axis=-1, keepdims=True) * face_normals

    return reflection_directions

def find_mesh_intersections(
    mesh: trimesh.Trimesh,
    ray_origins: np.ndarray,
    ray_directions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find intersections between rays and mesh faces.
    
    Args:
        mesh: Input mesh
        ray_origins: Starting points of rays [N, 3]
        ray_directions: Direction vectors of rays [N, 3]
        
    Returns:
        Tuple of (intersection_points, face_indices, distances)
    """
    # Use trimesh's ray intersection capabilities
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False
    )

    if len(locations) == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=int), np.empty((0,))

    # Calculate distances from ray origins to intersection points
    distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)

    return locations, index_tri, distances

def render_transmitter_receiver_positions(
    mesh: trimesh.Trimesh,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    scale_factor: float = 0.05
) -> trimesh.Scene:
    """
    Create a visualization scene with mesh and transmitter/receiver positions.
    
    Args:
        mesh: Input mesh
        tx_position: Transmitter position [x, y, z]
        rx_position: Receiver position [x, y, z]
        scale_factor: Size of TX/RX markers relative to mesh size
        
    Returns:
        Scene with mesh and positioned markers
    """
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.add_geometry(trimesh.creation.icosphere(radius=scale_factor, center=tx_position))
    scene.add_geometry(trimesh.creation.icosphere(radius=scale_factor, center=rx_position))
    return scene

def load_mesh(path: str) -> trimesh.Trimesh:
    """
    Load a mesh from `path` and return a trimesh.Trimesh object.
    
    Args:
        path: Path to the mesh file (STL, OBJ, PLY, etc.)
    
    Returns:
        A trimesh.Trimesh object representing the loaded mesh
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid triangle mesh
        RuntimeError: If loading fails for other reasons
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = trimesh.load_mesh(path)

    if not mesh.is_watertight:
        raise ValueError(f"Mesh is not a valid triangle mesh: {path}")

    return mesh