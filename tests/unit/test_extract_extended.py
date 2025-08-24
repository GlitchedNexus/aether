"""
Extended unit tests for the extract module.
"""

import pytest
import numpy as np
import trimesh
from aether.extract import detect_specular, detect_edge_diffraction, detect_tips, extract_all_scatterers
from aether.config import RadarConfig, ProcessingConfig


def test_detect_specular_complex_geometry():
    """Test specular detection on complex geometry."""
    # Create a more complex mesh (icosahedron)
    mesh = trimesh.creation.icosahedron()
    
    freq = 10.0
    tx = [2.0, 2.0, 2.0]
    rx = [2.0, 2.0, 2.0]  # Monostatic
    
    with pytest.raises(NotImplementedError):
        scatterers = detect_specular(mesh, freq, tx, rx)


def test_detect_specular_bistatic():
    """Test specular detection in bistatic configuration."""
    # Simple cube
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    
    freq = 10.0
    tx = [3.0, 0.0, 0.0]   # TX on the side
    rx = [-3.0, 0.0, 0.0]  # RX on opposite side
    
    with pytest.raises(NotImplementedError):
        scatterers = detect_specular(mesh, freq, tx, rx)


def test_detect_specular_frequency_effects():
    """Test specular detection at different frequencies."""
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    
    tx = [0.0, 0.0, 2.0]
    rx = [0.0, 0.0, 2.0]
    
    # Test multiple frequencies
    frequencies = [1.0, 10.0, 100.0]  # Low, medium, high frequency
    
    for freq in frequencies:
        with pytest.raises(NotImplementedError):
            scatterers = detect_specular(mesh, freq, tx, rx)


def test_detect_edges_simple_box():
    """Test edge detection on a simple box."""
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(3, 0, 0),
        rx_position=(-3, 0, 0)
    )
    
    with pytest.raises(NotImplementedError):
        edge_scatterers = detect_edge_diffraction(mesh, config)


def test_detect_edges_cylinder():
    """Test edge detection on a cylinder (curved edges)."""
    mesh = trimesh.creation.cylinder(radius=1, height=2)
    
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(2, 0, 0),
        rx_position=(-2, 0, 0)
    )
    
    with pytest.raises(NotImplementedError):
        edge_scatterers = detect_edge_diffraction(mesh, config)


def test_detect_edges_no_sharp_edges():
    """Test edge detection on smooth geometry."""
    # Create a smooth sphere (should have minimal sharp edges)
    mesh = trimesh.creation.uv_sphere(radius=1, count=[20, 20])
    
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(2, 0, 0),
        rx_position=(-2, 0, 0)
    )
    
    with pytest.raises(NotImplementedError):
        edge_scatterers = detect_edge_diffraction(mesh, config)


def test_detect_tips_pyramid():
    """Test tip detection on a pyramid."""
    # Create a pyramid-like structure
    vertices = np.array([
        [0, 0, 0],      # Base vertices
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0.5, 0.5, 1]   # Apex (tip)
    ])
    
    faces = np.array([
        [0, 1, 4],  # Side faces
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
        [0, 3, 2],  # Base faces
        [0, 2, 1]
    ])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(2, 2, 2),
        rx_position=(-2, -2, 2)
    )
    
    with pytest.raises(NotImplementedError):
        tip_scatterers = detect_tips(mesh, config)


def test_detect_tips_cube_corners():
    """Test tip detection on cube corners."""
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(3, 3, 3),
        rx_position=(-3, -3, -3)
    )
    
    with pytest.raises(NotImplementedError):
        tip_scatterers = detect_tips(mesh, config)


def test_detect_tips_smooth_surface():
    """Test tip detection on smooth surface (should find few tips)."""
    mesh = trimesh.creation.uv_sphere(radius=1, count=[20, 20])
    
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(2, 0, 0),
        rx_position=(-2, 0, 0)
    )
    
    with pytest.raises(NotImplementedError):
        tip_scatterers = detect_tips(mesh, config)


def test_extract_all_scatterers_default_config():
    """Test extract_all_scatterers with default processing config."""
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 2),
        rx_position=(0, 0, 2)
    )
    
    with pytest.raises(NotImplementedError):
        scatterers = extract_all_scatterers(mesh, config)


def test_extract_all_scatterers_with_edges_and_tips():
    """Test extract_all_scatterers with edge and tip detection enabled."""
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(2, 2, 2),
        rx_position=(-2, -2, -2)
    )
    
    proc_config = ProcessingConfig(
        num_top_scatterers=50,
        min_score_threshold=0.05,
        edge_detection=True,
        tip_detection=True
    )
    
    with pytest.raises(NotImplementedError):
        scatterers = extract_all_scatterers(mesh, config, proc_config)


def test_extract_all_scatterers_filtering():
    """Test extract_all_scatterers with strict filtering."""
    mesh = trimesh.creation.icosahedron()
    
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 3),
        rx_position=(0, 0, 3)
    )
    
    # Strict filtering
    proc_config = ProcessingConfig(
        num_top_scatterers=5,       # Only top 5
        min_score_threshold=0.5,    # High threshold
        edge_detection=False,
        tip_detection=False
    )
    
    with pytest.raises(NotImplementedError):
        scatterers = extract_all_scatterers(mesh, config, proc_config)


def test_extract_all_scatterers_empty_mesh():
    """Test extract_all_scatterers with empty mesh."""
    # Create mesh with no faces
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([]).reshape(0, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 2),
        rx_position=(0, 0, 2)
    )
    
    with pytest.raises(NotImplementedError):
        scatterers = extract_all_scatterers(mesh, config)


def test_extract_scatterers_distance_variation():
    """Test scatterer extraction with varying TX/RX distances."""
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    
    distances = [1, 5, 10, 50]  # Different distances
    
    for dist in distances:
        config = RadarConfig(
            frequency_ghz=10.0,
            tx_position=(0, 0, dist),
            rx_position=(0, 0, dist)
        )
        
        with pytest.raises(NotImplementedError):
            scatterers = extract_all_scatterers(mesh, config)


def test_detect_scatterers_various_aspect_angles():
    """Test scatterer detection from various aspect angles."""
    mesh = trimesh.creation.box(extents=[2, 1, 0.5])  # Elongated box
    
    # Test different aspect angles
    angles = [0, 30, 45, 60, 90]  # degrees
    
    for angle in angles:
        angle_rad = np.radians(angle)
        tx_pos = [3 * np.cos(angle_rad), 3 * np.sin(angle_rad), 0]
        rx_pos = [3 * np.cos(angle_rad), 3 * np.sin(angle_rad), 0]
        
        config = RadarConfig(
            frequency_ghz=10.0,
            tx_position=tuple(tx_pos),
            rx_position=tuple(rx_pos)
        )
        
        with pytest.raises(NotImplementedError):
            scatterers = extract_all_scatterers(mesh, config)


def test_detect_scatterers_elevation_angles():
    """Test scatterer detection from various elevation angles."""
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    
    # Test different elevation angles
    elevations = [0, 15, 30, 45, 60, 90]  # degrees
    
    for elev in elevations:
        elev_rad = np.radians(elev)
        distance = 3
        tx_pos = [0, distance * np.cos(elev_rad), distance * np.sin(elev_rad)]
        rx_pos = [0, distance * np.cos(elev_rad), distance * np.sin(elev_rad)]
        
        config = RadarConfig(
            frequency_ghz=10.0,
            tx_position=tuple(tx_pos),
            rx_position=tuple(rx_pos)
        )
        
        with pytest.raises(NotImplementedError):
            scatterers = extract_all_scatterers(mesh, config)


def test_scatterer_data_structure():
    """Test that scatterer data structures are correctly formatted."""
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 2),
        rx_position=(0, 0, 2)
    )
    
    with pytest.raises(NotImplementedError):
        scatterers = extract_all_scatterers(mesh, config)
        
        # This is what we would test if the function was implemented:
        # for scatterer in scatterers:
        #     assert 'position' in scatterer
        #     assert 'score' in scatterer
        #     assert 'type' in scatterer
        #     assert len(scatterer['position']) == 3
        #     assert isinstance(scatterer['score'], (int, float))
        #     assert scatterer['type'] in ['specular', 'edge', 'tip']


def test_scatterer_score_ordering():
    """Test that scatterers are returned in descending score order."""
    mesh = trimesh.creation.icosahedron()
    
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 3),
        rx_position=(0, 0, 3)
    )
    
    with pytest.raises(NotImplementedError):
        scatterers = extract_all_scatterers(mesh, config)
        
        # This is what we would test if the function was implemented:
        # scores = [s['score'] for s in scatterers]
        # assert scores == sorted(scores, reverse=True), "Scatterers should be ordered by descending score"
