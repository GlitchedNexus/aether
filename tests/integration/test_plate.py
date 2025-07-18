"""
Integration test for a simple plate scenario.
"""

import os
import pytest
import trimesh
import numpy as np
import tempfile
import shutil
import csv
from aether.io import load_mesh
from aether.extract import extract_all_scatterers
from aether.config import create_radar_config, create_processing_config
from aether.export import write_outputs


@pytest.fixture
def plate_mesh():
    """Create a simple square plate for testing."""
    # Create a 1x1 square plate on the XY plane
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ])
    
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_end_to_end_plate(plate_mesh, temp_output_dir):
    """Test the complete workflow on a simple plate."""
    # Save the plate to a temporary file
    temp_file = os.path.join(temp_output_dir, "plate.stl")
    plate_mesh.export(temp_file)
    
    # Create radar and processing configurations
    radar_config = create_radar_config(
        freq_ghz=10.0,
        tx_pos=[0.5, 0.5, 1.0],
        rx_pos=[0.5, 0.5, 1.0]
    )
    
    proc_config = create_processing_config(
        num_top_scatterers=10,
        min_score_threshold=0.01
    )
    
    try:
        # Load the mesh
        mesh = load_mesh(temp_file)
        
        # Extract scatterers
        scatterers = extract_all_scatterers(mesh, radar_config, proc_config)
        
        # There should be at least one scatterer for a plate with direct reflection
        assert len(scatterers) > 0
        
        # Export results
        output_dir = os.path.join(temp_output_dir, "output")
        write_outputs(mesh, scatterers, output_dir)
        
        # Verify output files exist
        assert os.path.exists(os.path.join(output_dir, "scatterers.csv"))
        assert os.path.exists(os.path.join(output_dir, "scatterers.json"))
        assert os.path.exists(os.path.join(output_dir, "visualization.ply"))
        
        # Check CSV content
        with open(os.path.join(output_dir, "scatterers.csv"), 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            first_row = next(reader)
            
            # Check header
            assert "x" in header
            assert "y" in header
            assert "z" in header
            assert "score" in header
            assert "type" in header
            
            # Check data row has expected number of fields
            assert len(first_row) == len(header)
            
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
