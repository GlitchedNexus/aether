"""
Extended unit tests for the weight module.
"""

import pytest
import numpy as np
from aether.weight import (
    compute_specular_weight, compute_edge_weight, compute_tip_weight,
    apply_polarization_factor
)
from aether.config import RadarConfig


def test_compute_specular_weight_normal_incidence():
    """Test specular weight computation for normal incidence."""
    # Create radar config for normal incidence
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 5),
        rx_position=(0, 0, 5)  # Monostatic
    )
    
    # Face pointing up (perfect alignment)
    face_normals = np.array([[0, 0, 1]])
    face_centers = np.array([[0, 0, 0]])
    face_areas = np.array([1.0])
    
    with pytest.raises(NotImplementedError):
        weights = compute_specular_weight(face_normals, face_centers, face_areas, config)


def test_compute_specular_weight_grazing_incidence():
    """Test specular weight computation for grazing incidence."""
    # Create radar config for grazing incidence
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(10, 0, 0.1),
        rx_position=(-10, 0, 0.1)
    )
    
    # Face pointing up (poor alignment for grazing)
    face_normals = np.array([[0, 0, 1]])
    face_centers = np.array([[0, 0, 0]])
    face_areas = np.array([1.0])
    
    with pytest.raises(NotImplementedError):
        weights = compute_specular_weight(face_normals, face_centers, face_areas, config)


def test_compute_specular_weight_multiple_faces():
    """Test specular weight computation for multiple faces."""
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 5),
        rx_position=(0, 0, 5)
    )
    
    # Multiple faces with different orientations
    face_normals = np.array([
        [0, 0, 1],   # Pointing up (good alignment)
        [1, 0, 0],   # Pointing right (poor alignment)
        [0, 1, 0],   # Pointing forward (poor alignment)
        [0, 0, -1]   # Pointing down (worst alignment)
    ])
    face_centers = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    face_areas = np.array([1.0, 1.0, 1.0, 1.0])
    
    with pytest.raises(NotImplementedError):
        weights = compute_specular_weight(face_normals, face_centers, face_areas, config)


def test_compute_edge_weight_perpendicular_edges():
    """Test edge weight computation for edges perpendicular to line of sight."""
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 5),
        rx_position=(0, 0, 5)
    )
    
    # Edge perpendicular to line of sight (good for diffraction)
    edge_vectors = np.array([[1, 0, 0]])
    edge_centers = np.array([[0, 0, 0]])
    edge_lengths = np.array([2.0])
    
    with pytest.raises(NotImplementedError):
        weights = compute_edge_weight(edge_vectors, edge_centers, edge_lengths, config)


def test_compute_edge_weight_parallel_edges():
    """Test edge weight computation for edges parallel to line of sight."""
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 5),
        rx_position=(0, 0, 5)
    )
    
    # Edge parallel to line of sight (poor for diffraction)
    edge_vectors = np.array([[0, 0, 1]])
    edge_centers = np.array([[0, 0, 0]])
    edge_lengths = np.array([2.0])
    
    with pytest.raises(NotImplementedError):
        weights = compute_edge_weight(edge_vectors, edge_centers, edge_lengths, config)


def test_compute_edge_weight_frequency_dependence():
    """Test edge weight frequency dependence."""
    # High frequency config
    config_high = RadarConfig(
        frequency_ghz=30.0,
        tx_position=(2, 0, 0),
        rx_position=(-2, 0, 0)
    )
    
    # Low frequency config
    config_low = RadarConfig(
        frequency_ghz=1.0,
        tx_position=(2, 0, 0),
        rx_position=(-2, 0, 0)
    )
    
    edge_vectors = np.array([[0, 1, 0]])
    edge_centers = np.array([[0, 0, 0]])
    edge_lengths = np.array([1.0])
    
    with pytest.raises(NotImplementedError):
        weights_high = compute_edge_weight(edge_vectors, edge_centers, edge_lengths, config_high)
        weights_low = compute_edge_weight(edge_vectors, edge_centers, edge_lengths, config_low)


def test_compute_tip_weight_backscatter():
    """Test tip weight computation for backscatter geometry."""
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 5),
        rx_position=(0, 0, 5)  # Monostatic backscatter
    )
    
    vertex_positions = np.array([[0, 0, 0]])
    
    with pytest.raises(NotImplementedError):
        weights = compute_tip_weight(vertex_positions, config)


def test_compute_tip_weight_forward_scatter():
    """Test tip weight computation for forward scatter geometry."""
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 5),
        rx_position=(0, 0, -5)  # Forward scatter
    )
    
    vertex_positions = np.array([[0, 0, 0]])
    
    with pytest.raises(NotImplementedError):
        weights = compute_tip_weight(vertex_positions, config)


def test_compute_tip_weight_multiple_vertices():
    """Test tip weight computation for multiple vertices."""
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(3, 0, 0),
        rx_position=(-3, 0, 0)
    )
    
    # Multiple vertices at different distances
    vertex_positions = np.array([
        [0, 0, 0],    # Central vertex
        [1, 0, 0],    # Closer to TX
        [-1, 0, 0],   # Closer to RX
        [0, 2, 0]     # Offset vertex
    ])
    
    with pytest.raises(NotImplementedError):
        weights = compute_tip_weight(vertex_positions, config)


def test_apply_polarization_factor_vv():
    """Test VV polarization factor application."""
    weights = np.array([1.0, 0.5, 0.1])
    
    with pytest.raises(NotImplementedError):
        modified_weights = apply_polarization_factor(weights, 'VV')


def test_apply_polarization_factor_hh():
    """Test HH polarization factor application."""
    weights = np.array([1.0, 0.5, 0.1])
    
    with pytest.raises(NotImplementedError):
        modified_weights = apply_polarization_factor(weights, 'HH')


def test_apply_polarization_factor_cross_pol():
    """Test cross-polarization (HV/VH) factor application."""
    weights = np.array([1.0, 0.5, 0.1])
    
    with pytest.raises(NotImplementedError):
        modified_weights_hv = apply_polarization_factor(weights, 'HV')
        modified_weights_vh = apply_polarization_factor(weights, 'VH')


def test_apply_polarization_factor_unknown():
    """Test unknown polarization factor application."""
    weights = np.array([1.0, 0.5, 0.1])
    
    with pytest.raises(NotImplementedError):
        modified_weights = apply_polarization_factor(weights, 'XX')


def test_compute_weights_distance_scaling():
    """Test that weights scale properly with distance."""
    # Near configuration
    config_near = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 1),
        rx_position=(0, 0, 1)
    )
    
    # Far configuration
    config_far = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 10),
        rx_position=(0, 0, 10)
    )
    
    face_normals = np.array([[0, 0, 1]])
    face_centers = np.array([[0, 0, 0]])
    face_areas = np.array([1.0])
    
    with pytest.raises(NotImplementedError):
        weights_near = compute_specular_weight(face_normals, face_centers, face_areas, config_near)
        weights_far = compute_specular_weight(face_normals, face_centers, face_areas, config_far)


def test_compute_weights_area_scaling():
    """Test that specular weights scale properly with face area."""
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 5),
        rx_position=(0, 0, 5)
    )
    
    face_normals = np.array([
        [0, 0, 1],
        [0, 0, 1]
    ])
    face_centers = np.array([
        [0, 0, 0],
        [1, 0, 0]
    ])
    # Different areas
    face_areas = np.array([1.0, 4.0])  # Second face is 4x larger
    
    with pytest.raises(NotImplementedError):
        weights = compute_specular_weight(face_normals, face_centers, face_areas, config)


def test_compute_weights_edge_cases():
    """Test weight computation edge cases."""
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 1),
        rx_position=(0, 0, 1)
    )
    
    # Empty arrays
    with pytest.raises(NotImplementedError):
        weights = compute_specular_weight(
            np.array([]).reshape(0, 3),
            np.array([]).reshape(0, 3),
            np.array([]),
            config
        )
    
    # Zero area faces
    face_normals = np.array([[0, 0, 1]])
    face_centers = np.array([[0, 0, 0]])
    face_areas = np.array([0.0])
    
    with pytest.raises(NotImplementedError):
        weights = compute_specular_weight(face_normals, face_centers, face_areas, config)
