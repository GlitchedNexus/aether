# aether/export.py

import os
import numpy as np
import pandas as pd
import trimesh
from typing import List, Dict, Any
import json

def write_outputs(mesh: trimesh.Trimesh, scatterers: List[Dict[str, Any]], outdir: str) -> None:
    """
    Export scatterers data and visualization outputs to the specified directory.
    
    Args:
        mesh: The original mesh
        scatterers: List of scatterer dictionaries from detect_specular
        outdir: Directory to write outputs to
    """
    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    
    # Export scatterers to CSV
    export_to_csv(scatterers, os.path.join(outdir, "scatterers.csv"))
    
    # Export JSON for advanced analysis
    export_to_json(scatterers, os.path.join(outdir, "scatterers.json"))
    
    # Create visualization mesh with heatmap
    vis_mesh = create_visualization_mesh(mesh, scatterers)
    vis_mesh.export(os.path.join(outdir, "visualization.ply"))
    
    print(f"Outputs written to {outdir}:")
    print(f"  - {os.path.join(outdir, 'scatterers.csv')}")
    print(f"  - {os.path.join(outdir, 'scatterers.json')}")
    print(f"  - {os.path.join(outdir, 'visualization.ply')}")

def export_to_csv(scatterers: List[Dict[str, Any]], filepath: str) -> None:
    """Export scatterers data to CSV format."""
    if not scatterers:
        print("Warning: No scatterers found to export")
        with open(filepath, 'w') as f:
            f.write("x,y,z,score,type\n")
        return
    
    # Create DataFrame
    data = {
        'x': [s['position'][0] for s in scatterers],
        'y': [s['position'][1] for s in scatterers],
        'z': [s['position'][2] for s in scatterers],
        'score': [s['score'] for s in scatterers],
        'type': [s['type'] for s in scatterers],
        'face_idx': [s['face_idx'] for s in scatterers],
    }
    
    df = pd.DataFrame(data)
    
    # Export to CSV
    df.to_csv(filepath, index=False)

def export_to_json(scatterers: List[Dict[str, Any]], filepath: str) -> None:
    """Export scatterers data to JSON format for advanced analysis."""
    with open(filepath, 'w') as f:
        json.dump(scatterers, f, indent=2)

def create_visualization_mesh(mesh: trimesh.Trimesh, scatterers: List[Dict[str, Any]]) -> trimesh.Trimesh:
    """
    Create a visualization mesh with colored faces based on scatterer scores.
    
    Args:
        mesh: Original mesh
        scatterers: List of scatterer dictionaries
    
    Returns:
        A new mesh with vertex colors representing scatterer intensity
    """
    # Create a copy of the mesh for visualization
    vis_mesh = mesh.copy()
    
    # Initialize colors to a light gray (default for non-scatterers)
    default_color = [200, 200, 200, 255]  # RGBA
    face_colors = np.tile(default_color, (len(mesh.faces), 1))
    
    if scatterers:
        # Get min and max scores for normalization
        scores = np.array([s['score'] for s in scatterers])
        max_score = scores.max()
        min_score = scores.min()
        
        # Check if we have a valid range for normalization
        score_range = max_score - min_score
        
        # Set colors based on normalized scores
        for scatterer in scatterers:
            face_idx = scatterer['face_idx']
            
            # Handle the case where all scores are the same
            if score_range <= 1e-10:
                normalized_score = 1.0  # All scores are equal, use max color
            else:
                normalized_score = (scatterer['score'] - min_score) / score_range
            
            # Safety check for numerical issues
            normalized_score = np.clip(normalized_score, 0.0, 1.0)
            
            # Create a color from blue (cold) to red (hot) based on score
            # Low scores: blue [0, 0, 255]
            # High scores: red [255, 0, 0]
            r = int(255 * normalized_score)
            b = int(255 * (1 - normalized_score))
            g = 0
            
            face_colors[face_idx] = [r, g, b, 255]
    
    # Apply colors to the mesh
    vis_mesh.visual.face_colors = face_colors
    
    return vis_mesh
