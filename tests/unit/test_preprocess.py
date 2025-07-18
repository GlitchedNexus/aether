"""
Test the mesh preprocessing functionality.
"""

import pytest
import numpy as np
import trimesh
from aether.preprocess import normalize_mesh, check_mesh_quality, prepare_mesh


def test_normalize_mesh():
    """Test mesh normalization."""
    # Create an offset and scaled cube
    vertices = np.array([
        [10, 10, 10],
        [15, 10, 10],
        [15, 15, 10],
        [10, 15, 10],
        [10, 10, 15],
        [15, 10, 15],
        [15, 15, 15],
        [10, 15, 15],
    ])
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom face
        [4, 5, 6], [4, 6, 7],  # top face
        [0, 1, 5], [0, 5, 4],  # front face
        [3, 2, 6], [3, 6, 7],  # back face
        [0, 3, 7], [0, 7, 4],  # left face
        [1, 2, 6], [1, 6, 5],  # right face
    ])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Normalize the mesh
    normalized = normalize_mesh(mesh)
    
    # Check that it's centered at origin
    assert np.allclose(normalized.centroid, [0, 0, 0], atol=1e-5)
    
    # Check that it fits in a unit cube
    assert np.all(normalized.bounds[1] - normalized.bounds[0] <= 1.0)
    assert np.max(normalized.extents) <= 1.0


def test_check_mesh_quality():
    """Test mesh quality checking."""
    # Create a mesh with duplicate vertices and degenerate faces
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],  # Duplicate vertex
    ])
    
    faces = np.array([
        [0, 1, 2],  # Regular face
        [1, 1, 2],  # Degenerate face (two vertices are the same)
    ])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Check quality
    quality = check_mesh_quality(mesh)
    
    # Verify the report
    assert quality["vertex_count"] == 4
    assert quality["face_count"] == 2
    assert quality["has_duplicate_vertices"] == True
    assert quality["has_degenerate_faces"] == True


def test_prepare_mesh():
    """Test the mesh preparation pipeline."""
    # Create a mesh with issues
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],  # Duplicate vertex
    ])
    
    faces = np.array([
        [0, 1, 2],
        [3, 1, 2],  # Using duplicate vertex
    ])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Prepare the mesh
    prepared, quality_report = prepare_mesh(mesh, normalize=True)
    
    # Verify the results
    assert quality_report["has_duplicate_vertices"] == True
    assert len(prepared.vertices) <= 3  # Should have merged duplicates
    
    # Check normalization
    assert np.allclose(prepared.centroid, [0, 0, 0], atol=1e-5)
    assert np.max(prepared.extents) <= 1.0
