"""
Aether Configuration Module

This module handles radar frequency/Tx-Rx definitions and any global settings.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class RadarConfig:
    """Configuration class for radar parameters."""
    frequency_ghz: float                     # Radar frequency in GHz
    tx_position: Tuple[float, float, float]  # Transmitter position in XYZ (meters)
    rx_position: Tuple[float, float, float]  # Receiver position in XYZ (meters)

    @property
    def wavelength(self) -> float:
        """Calculate wavelength in meters from frequency in GHz."""
        return 299792458 / (self.frequency_ghz * 1e9)  # Speed of light / frequency in Hz

    @property
    def tx_pos_array(self) -> np.ndarray:
        """Get transmitter position as numpy array."""
        return np.array(self.tx_position)
    
    @property
    def rx_pos_array(self) -> np.ndarray:
        """Get receiver position as numpy array."""
        return np.array(self.rx_position)


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    num_top_scatterers: int = 100        # Number of top scatterers to retain
    min_score_threshold: float = 0.01    # Minimum normalized score (relative to max) to keep
    edge_detection: bool = False         # Whether to detect edge diffraction
    tip_detection: bool = False          # Whether to detect tip diffraction
    face_detection: bool = False         # Whether to detect face scattering
    min_edge_length: float = 1e-3       # Minimum edge length to consider (meters)
    min_tip_curvature: float = 0.1      # Minimum tip curvature to consider


def create_radar_config(freq_ghz: float, tx_pos: List[float], rx_pos: List[float], ox_pos: List[float]) -> RadarConfig:
    """
    Create a radar configuration object from parameters.
    
    Args:
        freq_ghz: Radar frequency in GHz
        tx_pos: Transmitter position as [x, y, z] in meters
        rx_pos: Receiver position as [x, y, z] in meters

    Returns:
        RadarConfig object
    """
    return RadarConfig(
        frequency_ghz=freq_ghz,
        tx_position=(tx_pos[0], tx_pos[1], tx_pos[2]),
        rx_position=(rx_pos[0], rx_pos[1], rx_pos[2]),
    )


def create_processing_config(
    num_top_scatterers: int = 100,
    min_score_threshold: float = 0.01,
    edge_detection: bool = False,
    tip_detection: bool = False,
    face_detection: bool = False
) -> ProcessingConfig:
    """
    Create a processing configuration object.
    
    Args:
        num_top_scatterers: Number of top scatterers to retain
        min_score_threshold: Minimum normalized score to keep
        edge_detection: Whether to detect edge diffraction
        tip_detection: Whether to detect tip diffraction
        face_detection: Whether to detect face scattering

    Returns:
        ProcessingConfig object
    """
    return ProcessingConfig(
        num_top_scatterers=num_top_scatterers,
        min_score_threshold=min_score_threshold,
        edge_detection=edge_detection,
        tip_detection=tip_detection,
        face_detection=face_detection
    )


@dataclass
class ObjectConfig:
    """Configuration class for object parameters."""
    position: Tuple[float, float, float]  # Object position in XYZ (meters)
    size: Tuple[float, float, float]      # Object size (width, height, depth) in meters
    material: str                         # Object material type

    @property
    def object_pos_array(self) -> np.ndarray:
        """Get object position as numpy array."""
        return np.array(self.position)
    

def create_object_config(
    position: List[float],
    size: List[float],
    material: str
) -> ObjectConfig:
    """
    Create an object configuration object.

    Args:
        position: Object position as [x, y, z] in meters
        size: Object size as [width, height, depth] in meters
        material: Object material type

    Returns:
        ObjectConfig object
    """
    return ObjectConfig(
        position=(position[0], position[1], position[2]),
        size=(size[0], size[1], size[2]),
        material=material
    )