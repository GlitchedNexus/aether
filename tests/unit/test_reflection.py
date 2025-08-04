"""
Test reflection functionality for face reflection calculations.
"""

import pytest
import numpy as np
import trimesh
from aether.io import calculate_face_reflection_direction, find_mesh_intersections


def test_calculate_face_reflection_direction_single_ray():
    """Test reflection direction calculation for a single incident ray."""
    # Face normal pointing up
    face_normals = np.array([[0, 0, 1]])
    
    # Incident ray at 45 degrees from the left
    incident_direction = np.array([1, 0, -1]) / np.sqrt(2)
    
    with pytest.raises(NotImplementedError):
        reflection_direction = calculate_face_reflection_direction(face_normals, incident_direction)


def test_calculate_face_reflection_direction_multiple_rays():
    """Test reflection direction calculation for multiple incident rays."""
    # Two face normals
    face_normals = np.array([
        [0, 0, 1],  # Horizontal surface
        [1, 0, 0],  # Vertical surface
    ])
    
    # Two incident rays
    incident_directions = np.array([
        [1, 0, -1],  # Coming from the left and above
        [0, 1, 0],   # Coming from the front
    ]) / np.sqrt(2)
    
    with pytest.raises(NotImplementedError):
        reflection_directions = calculate_face_reflection_direction(face_normals, incident_directions)


def test_calculate_face_reflection_direction_grazing_angle():
    """Test reflection at grazing angles."""
    # Face normal pointing up
    face_normals = np.array([[0, 0, 1]])
    
    # Nearly grazing incident ray
    incident_direction = np.array([1, 0, -0.1])
    incident_direction = incident_direction / np.linalg.norm(incident_direction)
    
    with pytest.raises(NotImplementedError):
        reflection_direction = calculate_face_reflection_direction(face_normals, incident_direction)


def test_find_mesh_intersections_simple_plane():
    """Test ray-mesh intersection with a simple plane."""
    # Create a simple horizontal plane
    vertices = np.array([
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0],
    ])
    
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Ray from above pointing down
    ray_origins = np.array([[0, 0, 1]])
    ray_directions = np.array([[0, 0, -1]])
    
    with pytest.raises(NotImplementedError):
        intersections, face_indices, distances = find_mesh_intersections(
            mesh, ray_origins, ray_directions
        )


def test_find_mesh_intersections_multiple_rays():
    """Test ray-mesh intersection with multiple rays."""
    # Create a simple triangular mesh
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ])
    
    faces = np.array([[0, 1, 2]])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Multiple rays
    ray_origins = np.array([
        [0.2, 0.2, 1],  # Should hit
        [0.8, 0.8, 1],  # Should miss
        [0.1, 0.1, 1],  # Should hit
    ])
    ray_directions = np.array([
        [0, 0, -1],
        [0, 0, -1],
        [0, 0, -1],
    ])
    
    with pytest.raises(NotImplementedError):
        intersections, face_indices, distances = find_mesh_intersections(
            mesh, ray_origins, ray_directions
        )


def test_find_mesh_intersections_no_intersection():
    """Test ray-mesh intersection when rays miss the mesh."""
    # Create a simple plane
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ])
    
    faces = np.array([[0, 1, 2]])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Ray that misses the mesh
    ray_origins = np.array([[2, 2, 1]])
    ray_directions = np.array([[0, 0, -1]])
    
    with pytest.raises(NotImplementedError):
        intersections, face_indices, distances = find_mesh_intersections(
            mesh, ray_origins, ray_directions
        )


def test_find_mesh_intersections_parallel_rays():
    """Test ray-mesh intersection with parallel rays."""
    # Create a cube
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    
    # Parallel rays from above
    ray_origins = np.array([
        [-0.5, -0.5, 2],
        [0, 0, 2],
        [0.5, 0.5, 2],
    ])
    ray_directions = np.array([
        [0, 0, -1],
        [0, 0, -1],
        [0, 0, -1],
    ])
    
    with pytest.raises(NotImplementedError):
        intersections, face_indices, distances = find_mesh_intersections(
            mesh, ray_origins, ray_directions
        )
