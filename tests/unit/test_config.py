"""
Test the configuration functionality.
"""

import pytest
import numpy as np
from aether.config import RadarConfig, ProcessingConfig, create_radar_config, create_processing_config


def test_radar_config():
    """Test RadarConfig creation and properties."""
    config = RadarConfig(
        frequency_ghz=10.0,
        tx_position=(0, 0, 10),
        rx_position=(0, 0, -10)
    )
    
    # Check properties
    assert config.frequency_ghz == 10.0
    assert config.tx_position == (0, 0, 10)
    assert config.rx_position == (0, 0, -10)
    
    # Check derived properties
    assert np.isclose(config.wavelength, 0.0299792458)  # c / (10 * 10^9)
    assert np.array_equal(config.tx_pos_array, [0, 0, 10])
    assert np.array_equal(config.rx_pos_array, [0, 0, -10])


def test_processing_config():
    """Test ProcessingConfig creation with defaults and custom values."""
    # Default config
    default_config = ProcessingConfig()
    assert default_config.num_top_scatterers == 100
    assert default_config.min_score_threshold == 0.01
    assert default_config.edge_detection == False
    assert default_config.tip_detection == False
    
    # Custom config
    custom_config = ProcessingConfig(
        num_top_scatterers=50,
        min_score_threshold=0.05,
        edge_detection=True,
        tip_detection=True
    )
    assert custom_config.num_top_scatterers == 50
    assert custom_config.min_score_threshold == 0.05
    assert custom_config.edge_detection == True
    assert custom_config.tip_detection == True


def test_create_radar_config():
    """Test the helper function for creating radar configs."""
    config = create_radar_config(15.0, [1, 2, 3], [4, 5, 6])
    
    assert config.frequency_ghz == 15.0
    assert config.tx_position == (1, 2, 3)
    assert config.rx_position == (4, 5, 6)


def test_create_processing_config():
    """Test the helper function for creating processing configs."""
    # Default values
    default_config = create_processing_config()
    assert default_config.num_top_scatterers == 100
    assert default_config.min_score_threshold == 0.01
    assert default_config.edge_detection == False
    assert default_config.tip_detection == False
    
    # Custom values
    custom_config = create_processing_config(
        num_top_scatterers=10,
        min_score_threshold=0.1,
        edge_detection=True,
        tip_detection=True
    )
    assert custom_config.num_top_scatterers == 10
    assert custom_config.min_score_threshold == 0.1
    assert custom_config.edge_detection == True
    assert custom_config.tip_detection == True
