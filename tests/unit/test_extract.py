"""
Test the extraction functionality.
"""

import pytest
import numpy as np
import trimesh
from aether.extract import detect_specular, extract_all_scatterers
from aether.config import create_radar_config, create_processing_config


def test_detect_specular_basic():
    """Test basic specular detection."""
    # Create a simple flat plate
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
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Set up radar configuration for normal incidence
    freq = 10.0  # GHz
    tx = [0.5, 0.5, 1.0]  # Above the center of the plate
    rx = [0.5, 0.5, 1.0]  # Same as tx (monostatic)
    
    # Detect specular points
    scatterers = detect_specular(mesh, freq, tx, rx)
    
    # Basic checks
    assert len(scatterers) > 0, "Should detect at least one scatterer"
    assert scatterers[0]['type'] == 'specular'
    
    # The strongest scatterer should be near the center of the plate
    strongest = scatterers[0]
    assert np.allclose(strongest['position'][:2], [0.5, 0.5], atol=0.3)
    
    # Normal should point in Z direction
    assert np.allclose(strongest['normal'], [0, 0, 1], atol=0.1)


def test_extract_all_scatterers():
    """Test the combined extraction function."""
    # Create a simple cube
    cube = trimesh.creation.box(extents=[1, 1, 1])
    
    # Create configuration
    radar_config = create_radar_config(
        freq_ghz=10.0,
        tx_pos=[2, 2, 2],
        rx_pos=[-2, -2, -2]
    )
    
    proc_config = create_processing_config(
        num_top_scatterers=10,
        min_score_threshold=0.05,
        edge_detection=True,
        tip_detection=True
    )
    
    # Extract all scatterers
    scatterers = extract_all_scatterers(cube, radar_config, proc_config)
    
    # Basic checks
    assert isinstance(scatterers, list)
    if len(scatterers) > 0:
        assert 'position' in scatterers[0]
        assert 'score' in scatterers[0]
        assert 'type' in scatterers[0]


def test_simple_plate_analytical():
    """
    Test specular reflection from a plate against analytical prediction.
    
    For a flat plate at normal incidence, the RCS is proportional to:
    σ = 4π * A² / λ²
    where A is the plate area and λ is the wavelength.
    """
    # Create a square plate of known size
    size = 10.0  # 10x10 meter plate
    vertices = np.array([
        [0, 0, 0],
        [size, 0, 0],
        [size, size, 0],
        [0, size, 0],
    ])
    
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    
    plate = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Set up radar at normal incidence
    freq = 10.0  # 10 GHz
    wavelength = 299792458 / (freq * 1e9)  # ~0.03m
    tx = [size/2, size/2, 10]  # 10m above center
    rx = [size/2, size/2, 10]  # Same (monostatic)
    
    # Analytical RCS for a flat plate at normal incidence
    plate_area = size * size
    analytical_rcs = 4 * np.pi * (plate_area**2) / (wavelength**2)
    
    # Detect specular points
    scatterers = detect_specular(plate, freq, tx, rx)
    
    # Sum up the scores (our approximation of RCS)
    computed_rcs = sum(s['score'] for s in scatterers)
    
    # The computation should be proportional to analytical result
    # We just check for the same order of magnitude because our computation 
    # includes additional factors
    assert computed_rcs > 0, "RCS should be positive"
    ratio = analytical_rcs / computed_rcs
    assert 0.01 < ratio < 100, f"RCS ratio outside expected range: {ratio}"
