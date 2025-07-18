"""
Test the mesh loading and IO functionality.
"""

import pytest
import os
import numpy as np
import trimesh
from aether.io import load_mesh


def test_load_mesh_stl():
    """Test loading an STL file."""
    # Create a test cube
    cube = trimesh.creation.box(extents=[1, 1, 1])
    
    # Write to a temporary file
    temp_path = "tests/unit/temp_cube.stl"
    cube.export(temp_path)
    
    try:
        # Load it back
        loaded_mesh = load_mesh(temp_path)
        
        # Check basic properties
        assert isinstance(loaded_mesh, trimesh.Trimesh)
        assert loaded_mesh.is_watertight
        assert len(loaded_mesh.faces) == 12  # A cube has 12 triangular faces
        
        # Check that vertices are approximately equal (floating point comparison)
        assert np.allclose(sorted(loaded_mesh.vertices.sum(axis=1)), 
                          sorted(cube.vertices.sum(axis=1)),
                          rtol=1e-5)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_load_nonexistent_mesh():
    """Test that loading a nonexistent file raises an appropriate error."""
    with pytest.raises(Exception):
        load_mesh("nonexistent_file.stl")


def test_mesh_cleanup():
    """Test that mesh cleanup works as expected."""
    # Create a mesh with duplicate vertices
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],  # Duplicate of first vertex
    ])
    
    faces = np.array([
        [0, 1, 2],
        [3, 1, 2],  # Same face but using the duplicate vertex
    ])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    temp_path = "tests/unit/temp_mesh_cleanup.stl"
    mesh.export(temp_path)
    
    try:
        # Load it back, which should clean up duplicates
        loaded_mesh = load_mesh(temp_path)
        
        # Check that duplicate vertices were merged
        assert len(loaded_mesh.vertices) <= 3
        assert len(loaded_mesh.faces) <= 1
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
