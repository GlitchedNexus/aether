"""
Aether Extraction Module

This module contains specular/edge/tip detection logic for radar scattering analysis.
"""

import numpy as np
import trimesh
from typing import List, Tuple, Dict, Any, Optional
from aether.config import RadarConfig, ProcessingConfig


def detect_specular_reflection(
    mesh: trimesh.Trimesh, 
    freq: float, 
    tx: List[float], 
    rx: List[float]
) -> List[Dict[str, Any]]:
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
    mesh = mesh.copy()
    mesh.apply_scale(1 / 100)  # Convert to meters
    scatterers = []

    for face_idx, face in enumerate(mesh.faces):
        # Get the vertices of the face
        vertices = mesh.vertices[face]
        # Compute the face normal
        normal = mesh.face_normals[face_idx]
        # Compute the center of the triangle
        center = vertices.mean(axis=0)

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)

        # Compute incident and reflected directions
        tx_vec = np.array(center) - np.array(tx)
        rx_vec = np.array(center) - np.array(rx)
        tx_vec /= np.linalg.norm(tx_vec)
        rx_vec /= np.linalg.norm(rx_vec)

        # Specular reflection condition: reflected TX should align with RX
        # Reflection direction: r = d - 2*(d·n)*n
        reflected = tx_vec - 2 * np.dot(tx_vec, normal) * normal
        alignment = np.dot(reflected, rx_vec)
        # Clamp alignment to [-1, 1] to avoid numerical issues
        alignment = np.clip(alignment, -1.0, 1.0)

        # Score: how well the specular condition is met (1 = perfect)
        score = max(0.0, alignment)

        # Optionally, threshold score to filter out non-specular faces
        if score > 0.95:  # Adjustable threshold
            scatterers.append({
                'position': center.tolist(),
                'normal': normal.tolist(),
                'score': float(score),
                'type': 'specular',
                'face_idx': int(face_idx)
            })

    return scatterers


def detect_edge_diffraction(mesh: trimesh.Trimesh, config: RadarConfig) -> List[Dict[str, Any]]:
    """
    Detect edge diffraction points on a 3D mesh.

    Args:
        mesh: A trimesh.Trimesh object representing the 3D model
        config: Radar configuration

    Returns:
        A list of dictionaries containing edge scatterer information:
        [
            {
                'position': [x, y, z],  # Center of the edge
                'direction': [dx, dy, dz],  # Edge direction vector
                'length': float,  # Length of the edge
                'score': float,  # Relative intensity/RCS
                'type': 'edge',  # Type of scattering
                'edge_idx': int,  # Index of the edge
            },
            ...
        ]
    """
    mesh = mesh.copy()
    mesh.apply_scale(1 / 100)
    scatterers = []

    # Use edge_faces for robust face adjacency
    edge_faces = mesh.edges_face
    min_length = getattr(config, "min_edge_length", 1e-3)

    for edge_idx, (edge, faces) in enumerate(zip(mesh.edges, edge_faces)):
        v0, v1 = mesh.vertices[edge]
        center = (v0 + v1) / 2.0
        direction_vec = v1 - v0
        length = np.linalg.norm(direction_vec)
        if length < min_length or length == 0:
            continue
        direction = direction_vec / length

        # Only consider edges with exactly two adjacent faces (manifold)
        if np.any(faces == -1) or len(faces) != 2:
            continue

        n0 = mesh.face_normals[faces[0]]
        n1 = mesh.face_normals[faces[1]]
        cos_theta = np.clip(np.dot(n0, n1), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        # Sharper edges (smaller theta) have higher score
        score = 1.0 - (theta / np.pi)
        if score < 0.05:
            continue

        scatterers.append({
            'position': center.tolist(),
            'direction': direction.tolist(),
            'length': float(length),
            'score': float(score),
            'type': 'edge',
            'edge_idx': int(edge_idx)
        })

    return scatterers


def detect_tip_diffraction(mesh: trimesh.Trimesh, config: RadarConfig) -> List[Dict[str, Any]]:
    """
    Detect tip diffraction points on a 3D mesh.

    Args:
        mesh: A trimesh.Trimesh object representing the 3D model
        config: Radar configuration

    Returns:
        A list of dictionaries containing tip scatterer information:
        [
            {
                'position': [x, y, z],  # Position of the tip
                'normal': [nx, ny, nz],  # Average normal at tip
                'curvature': float,  # Curvature measure
                'score': float,  # Relative intensity/RCS
                'type': 'tip',  # Type of scattering
                'vertex_idx': int,  # Index of the vertex
            },
            ...
        ]
    """
    mesh = mesh.copy()
    mesh.apply_scale(1 / 100)
    scatterers = []

    min_curvature = getattr(config, "min_tip_curvature", 0.1)

    for vertex_idx, vertex in enumerate(mesh.vertices):
        faces = mesh.vertex_faces[vertex_idx]
        faces = faces[faces != -1]
        if len(faces) < 2:
            continue

        normals = mesh.face_normals[faces]
        avg_normal = np.mean(normals, axis=0)
        norm = np.linalg.norm(avg_normal)
        if norm == 0:
            continue
        avg_normal /= norm

        # Curvature: mean deviation of face normals from average
        curvature = np.mean(np.linalg.norm(normals - avg_normal, axis=1))
        if curvature < min_curvature:
            continue

        # Score: how much the average normal opposes the vertex normal (sharpness)
        v_normal = mesh.vertex_normals[vertex_idx]
        v_normal_norm = np.linalg.norm(v_normal)
        if v_normal_norm == 0:
            continue
        v_normal /= v_normal_norm
        score = max(0.0, 1.0 - np.dot(avg_normal, v_normal))

        scatterers.append({
            'position': vertex.tolist(),
            'normal': avg_normal.tolist(),
            'curvature': float(curvature),
            'score': float(score),
            'type': 'tip',
            'vertex_idx': int(vertex_idx)
        })

    return scatterers


def extract_all_scatterers(
    mesh: trimesh.Trimesh,
    radar_config: RadarConfig,
    proc_config: Optional[ProcessingConfig] = None
) -> List[Dict[str, Any]]:
    """
    Extract all types of scatterers from a mesh based on configuration.
    
    Args:
        mesh: Input mesh
        radar_config: Radar configuration containing frequency and TX/RX positions
        proc_config: Processing configuration (optional)
        
    Returns:
        List of all scatterers (specular, edges, tips) sorted by importance
    """
    if proc_config is None:
        proc_config = ProcessingConfig()

    scatterers = []

    if proc_config.face_detection:
        speculars = detect_specular_reflection(
            mesh, radar_config.frequency_ghz, list(radar_config.tx_position), list(radar_config.rx_position)
        )
        scatterers.extend(speculars)

    if proc_config.edge_detection:
        edges = detect_edge_diffraction(mesh, radar_config)
        scatterers.extend(edges)

    if proc_config.tip_detection:
        tips = detect_tip_diffraction(mesh, radar_config)
        scatterers.extend(tips)

    # Normalize scores and filter based on config
    if scatterers:
        max_score = max(s['score'] for s in scatterers)
        for s in scatterers:
            s['score'] /= max_score  # Normalize to [0, 1]

        # Filter by minimum score threshold
        scatterers = [s for s in scatterers if s['score'] >= proc_config.min_score_threshold]

        # Sort by score descending
        scatterers.sort(key=lambda x: x['score'], reverse=True)

        # Keep only top K scatterers
        scatterers = scatterers[:proc_config.num_top_scatterers]

    return scatterers
