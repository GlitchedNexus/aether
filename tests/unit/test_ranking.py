"""
Test the ranking and weighting functionality.
"""

import pytest
import numpy as np
from aether.ranking import top_k_scatterers, threshold_scatterers, relative_threshold_scatterers


def test_top_k_scatterers():
    """Test the top-k selection function."""
    # Create test data
    weights = np.array([5.0, 10.0, 3.0, 7.0, 2.0])
    indices = np.array([0, 1, 2, 3, 4])
    
    # Get top 3
    top_weights, top_indices = top_k_scatterers(weights, 3, indices)
    
    # Check results
    assert len(top_weights) == 3
    assert len(top_indices) == 3
    assert np.array_equal(top_weights, [10.0, 7.0, 5.0])
    assert np.array_equal(top_indices, [1, 3, 0])


def test_threshold_scatterers():
    """Test the threshold filtering function."""
    # Create test data
    weights = np.array([5.0, 10.0, 3.0, 7.0, 2.0])
    indices = np.array([0, 1, 2, 3, 4])
    
    # Filter with threshold 5.0
    filtered_weights, filtered_indices = threshold_scatterers(weights, 5.0, indices)
    
    # Check results
    assert len(filtered_weights) == 3
    assert len(filtered_indices) == 3
    assert np.array_equal(filtered_weights, [5.0, 10.0, 7.0])
    assert np.array_equal(filtered_indices, [0, 1, 3])


def test_relative_threshold_scatterers():
    """Test the relative threshold filtering function."""
    # Create test data
    weights = np.array([5.0, 10.0, 3.0, 7.0, 2.0])
    indices = np.array([0, 1, 2, 3, 4])
    
    # Filter with relative threshold 0.5 (half of max)
    filtered_weights, filtered_indices = relative_threshold_scatterers(weights, 0.5, indices)
    
    # Max is 10.0, so threshold is 5.0
    assert len(filtered_weights) == 3
    assert len(filtered_indices) == 3
    assert np.array_equal(filtered_weights, [5.0, 10.0, 7.0])
    assert np.array_equal(filtered_indices, [0, 1, 3])


def test_empty_input():
    """Test behavior with empty input arrays."""
    weights = np.array([])
    indices = np.array([])
    
    # All functions should handle empty arrays gracefully
    top_w, top_i = top_k_scatterers(weights, 3, indices)
    assert len(top_w) == 0
    assert len(top_i) == 0
    
    thresh_w, thresh_i = threshold_scatterers(weights, 5.0, indices)
    assert len(thresh_w) == 0
    assert len(thresh_i) == 0
    
    rel_w, rel_i = relative_threshold_scatterers(weights, 0.5, indices)
    assert len(rel_w) == 0
    assert len(rel_i) == 0
