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
    # 1. Get edges from the mesh
    edges = mesh.edges
    
    # Skip if no edges (should never happen in a valid mesh)
    if len(edges) == 0:
        return []
    
    # 2. Calculate edge centers (midpoint of vertices)
    edge_vertices = mesh.vertices[edges]
    edge_centers = np.mean(edge_vertices, axis=1)
    
    # 3. Calculate edge vectors (direction along edge)
    edge_vectors = edge_vertices[:, 1, :] - edge_vertices[:, 0, :]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    
    # Normalize edge vectors
    with np.errstate(divide='ignore', invalid='ignore'):
        edge_vectors_norm = np.where(
            edge_lengths[:, np.newaxis] > 0,
            edge_vectors / edge_lengths[:, np.newaxis],
            np.zeros_like(edge_vectors)
        )
    
    # 4. Simplified approach: Directly find sharp edges using dihedral angles
    face_adjacency = mesh.face_adjacency  # Pairs of adjacent faces
    face_adjacency_edges = mesh.face_adjacency_edges  # Edge index shared by each face pair
    face_normals = mesh.face_normals
    
    # Calculate angles between adjacent faces
    adjacent_face_normals = face_normals[face_adjacency]
    adjacent_norm_dots = np.sum(adjacent_face_normals[:, 0, :] * adjacent_face_normals[:, 1, :], axis=1)
    
    # Clip dot products to valid range for arccos
    adjacent_norm_dots = np.clip(adjacent_norm_dots, -1.0, 1.0)
    adjacent_angles = np.arccos(adjacent_norm_dots)
    
    # Get edges where angle exceeds a threshold (e.g., 30 degrees)
    diffraction_threshold = np.radians(30)
    diffracting_edges_mask = adjacent_angles > diffraction_threshold
    
    # Get the indices of the diffracting edges - these are indices into the face_adjacency_edges array
    # We need to map them back to indices into the edges array
    diffraction_indices = np.where(diffracting_edges_mask)[0]
    
    # If no diffracting edges found, return empty list
    if len(diffraction_indices) == 0:
        return []
        
    # Get the actual edge indices
    edge_indices = []
    for i in diffraction_indices:
        # Extract the edge index
        edge_idx = tuple(sorted(mesh.face_adjacency_edges[i]))
        
        # Try to find the matching edge in the mesh.edges
        try:
            # Convert mesh.edges to a list of sorted tuples for comparison
            edges_as_tuples = [tuple(sorted(edge)) for edge in mesh.edges]
            if edge_idx in edges_as_tuples:
                edge_indices.append(edges_as_tuples.index(edge_idx))
        except Exception:
            continue
    
    edge_indices = np.array(edge_indices)
    
    # If we don't have valid edge indices, return empty list
    if len(edge_indices) == 0:
        return []
    
    # 6. Apply weighting function to these edges
    # Only calculate weights for valid edges
    from aether.weight import compute_edge_weight
    try:
        edge_vectors_subset = edge_vectors_norm[edge_indices]
        edge_centers_subset = edge_centers[edge_indices]
        edge_lengths_subset = edge_lengths[edge_indices]
        
        edge_weights = compute_edge_weight(
            edge_vectors_subset,
            edge_centers_subset,
            edge_lengths_subset,
            config
        )
    except Exception as e:
        print(f"Warning: Edge weight calculation failed: {str(e)}")
        return []
    
    # 7. Create scatterer objects for significant edges
    # Sort edges by weight
    sorted_indices = np.argsort(-edge_weights)
    
    # Only include edges with non-zero weights
    positive_mask = edge_weights[sorted_indices] > 0
    sorted_indices = sorted_indices[positive_mask]
    
    # Collect results
    scatterers = []
    for i, idx in enumerate(sorted_indices):
        edge_idx = edge_indices[idx]
        
        # Only include top edges or those above threshold
        if i >= 100:
            break
            
        if edge_weights[idx] < 0.01 * edge_weights[sorted_indices[0]]:
            break
            
        scatterers.append({
            'position': edge_centers_subset[idx].tolist(),
            'direction': edge_vectors_subset[idx].tolist(),
            'length': float(edge_lengths_subset[idx]),
            'score': float(edge_weights[idx]),
            'type': 'edge',
            'edge_idx': int(edge_idx)
        })
    
    return scatterers


def detect_tips(mesh: trimesh.Trimesh, config: RadarConfig) -> List[Dict[str, Any]]:
    """
    Detect tip diffraction points on a 3D mesh.
    
    Args:
        mesh: A trimesh.Trimesh object representing the 3D model
        config: Radar configuration
        
    Returns:
        A list of dictionaries containing tip scatterer information
    """
    # 1. Get vertex positions
    vertex_positions = mesh.vertices
    
    # Skip if no vertices (should never happen in a valid mesh)
    if len(vertex_positions) == 0:
        return []
    
    # 2. Identify "sharp" vertices that might create tip diffraction
    # Get faces connected to each vertex
    vertex_faces = mesh.vertex_faces
    
    # Vertices with at least one face
    valid_vertices = np.any(vertex_faces >= 0, axis=1)
    
    # Get vertex normals
    vertex_normals = np.zeros_like(vertex_positions)
    face_normals = mesh.face_normals
    
    tip_candidates = []
    vertex_curvatures = []
    
    # Find vertices with high curvature (where face normals vary significantly)
    for i in range(len(mesh.vertices)):
        if not valid_vertices[i]:
            continue
        
        # Get faces containing this vertex
        faces = vertex_faces[i]
        valid_faces = faces[faces >= 0]
        
        if len(valid_faces) < 3:
            continue
            
        # Get normals of these faces
        normals = face_normals[valid_faces]
        
        # Calculate average normal
        avg_normal = np.mean(normals, axis=0)
        avg_normal_length = np.linalg.norm(avg_normal)
        
        # Skip if average normal is zero (perfectly symmetric)
        if avg_normal_length < 1e-6:
            continue
            
        avg_normal = avg_normal / avg_normal_length
        
        # Calculate how much the normals vary (as a measure of curvature)
        normal_dots = np.sum(normals * avg_normal, axis=1)
        normal_dots = np.clip(normal_dots, -1.0, 1.0)
        normal_angles = np.arccos(normal_dots)
        
        # Average angle deviation is our curvature measure
        curvature = np.mean(normal_angles)
        
        # If curvature exceeds threshold, it's a tip candidate
        if curvature > np.radians(30):
            tip_candidates.append(i)
            vertex_curvatures.append(curvature)
            
            # Store this average normal
            vertex_normals[i] = avg_normal
    
    # Convert to numpy arrays
    tip_candidates = np.array(tip_candidates)
    vertex_curvatures = np.array(vertex_curvatures)
    
    if len(tip_candidates) == 0:
        return []
    
    # 3. Apply weighting function to candidate vertices
    from aether.weight import compute_tip_weight
    tip_weights = compute_tip_weight(
        vertex_positions[tip_candidates],
        config
    )
    
    # Scale weights by curvature (higher curvature = stronger diffraction)
    tip_weights = tip_weights * vertex_curvatures
    
    # 4. Create scatterer objects for significant tips
    # Sort tips by weight
    sorted_indices = np.argsort(-tip_weights)
    
    # Only include tips with non-zero weights
    positive_mask = tip_weights[sorted_indices] > 0
    sorted_indices = sorted_indices[positive_mask]
    
    # Collect results
    scatterers = []
    for i, idx in enumerate(sorted_indices):
        vertex_idx = tip_candidates[idx]
        
        # Only include top tips or those above threshold
        if i >= 50:  # Fewer tips than edges typically
            break
            
        if tip_weights[idx] < 0.01 * tip_weights[sorted_indices[0]]:
            break
            
        scatterers.append({
            'position': vertex_positions[vertex_idx].tolist(),
            'normal': vertex_normals[vertex_idx].tolist(),
            'curvature': float(vertex_curvatures[idx]),
            'score': float(tip_weights[idx]),
            'type': 'tip',
            'vertex_idx': int(vertex_idx)
        })
    
    return scatterers


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
    
    # Initialize with specular scatterers
    scatterers = list(specular_scatterers)  # Create a copy
    
    # Add edge scatterers if enabled
    edge_scatterers = []
    if proc_config.edge_detection:
        try:
            edge_scatterers = detect_edges(mesh, config)
            if edge_scatterers:  # Only add if we got valid results
                scatterers.extend(edge_scatterers)
                print(f"Detected {len(edge_scatterers)} edge diffraction points")
            else:
                print("No edge diffraction points detected")
        except Exception as e:
            print(f"Warning: Edge detection failed: {str(e)}")
            # Continue with the rest of the analysis without edge detection
    
    # Add tip scatterers if enabled
    tip_scatterers = []
    if proc_config.tip_detection:
        try:
            tip_scatterers = detect_tips(mesh, config)
            scatterers.extend(tip_scatterers)
            print(f"Detected {len(tip_scatterers)} tip diffraction points")
        except Exception as e:
            print(f"Warning: Tip detection failed: {str(e)}")
    
    # Log detection results
    print(f"Total scatterers detected: {len(scatterers)} "
          f"(Specular: {len(specular_scatterers)}, "
          f"Edge: {len(edge_scatterers)}, "
          f"Tip: {len(tip_scatterers)})")
    
    # If no scatterers found, return empty list
    if not scatterers:
        return []
    
    # Normalize scores by scatterer type to avoid one mechanism dominating
    # Find max score for each type
    if specular_scatterers:
        specular_max = max(s['score'] for s in specular_scatterers)
        # Normalize specular scores
        for s in specular_scatterers:
            s['normalized_score'] = s['score'] / specular_max
    
    if edge_scatterers:
        edge_max = max(s['score'] for s in edge_scatterers)
        # Normalize edge scores
        for s in edge_scatterers:
            s['normalized_score'] = s['score'] / edge_max * 0.8  # Scale factor for relative importance
    
    if tip_scatterers:
        tip_max = max(s['score'] for s in tip_scatterers)
        # Normalize tip scores
        for s in tip_scatterers:
            s['normalized_score'] = s['score'] / tip_max * 0.6  # Scale factor for relative importance
    
    # Sort all scatterers by normalized score (descending)
    scatterers.sort(key=lambda s: s.get('normalized_score', 0), reverse=True)
    
    # Apply top-k filtering
    if len(scatterers) > proc_config.num_top_scatterers:
        scatterers = scatterers[:proc_config.num_top_scatterers]
    
    # Filter by minimum normalized score
    if scatterers:
        max_norm_score = max(s.get('normalized_score', 0) for s in scatterers)
        min_norm_score = max_norm_score * proc_config.min_score_threshold
        scatterers = [s for s in scatterers if s.get('normalized_score', 0) >= min_norm_score]
    
    # Remove temporary normalized_score field
    for s in scatterers:
        if 'normalized_score' in s:
            del s['normalized_score']
    
    return scatterers
