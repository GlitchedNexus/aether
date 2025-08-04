import pytest
import numpy as np
import trimesh
from aether.extract import detect_edges
from aether.config import RadarConfig

def test_edge_detection():
    """Test that edge detection works correctly."""
    # Create a simple cube mesh
    cube = trimesh.creation.box()
    
    # Create a radar configuration
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=np.array([100, 0, 50]),
        rx_position=np.array([100, 0, 50])
    )
    
    # Detect edges
    edges = detect_edges(cube, config)
    
    # Check that edges were detected
    assert len(edges) > 0
    
    # Check that each edge has the required fields
    for edge in edges:
        assert 'position' in edge
        assert 'direction' in edge
        assert 'length' in edge
        assert 'score' in edge
        assert 'type' in edge
        assert edge['type'] == 'edge'
        assert 'edge_idx' in edge
        
        # Check position and direction are 3D vectors
        assert len(edge['position']) == 3
        assert len(edge['direction']) == 3

def test_edge_detection_with_complex_mesh():
    """Test edge detection with a more complex mesh."""
    # Create a more complex mesh (an icosahedron)
    mesh = trimesh.creation.icosahedron()
    
    # Create a radar configuration
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=np.array([10, 0, 5]),
        rx_position=np.array([10, 0, 5])
    )
    
    # Detect edges
    edges = detect_edges(mesh, config)
    
    # Check that edges were detected
    assert len(edges) > 0
    
    # Test with different radar positions
    config.tx_position = np.array([-10, 5, 0])
    config.rx_position = np.array([-10, 5, 0])
    
    edges2 = detect_edges(mesh, config)
    
    # Scores should be different based on radar position
    if len(edges) > 0 and len(edges2) > 0:
        assert edges[0]['score'] != edges2[0]['score']
