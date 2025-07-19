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
    
    # Create visualization mesh with color-coded scatterers
    vis_mesh = create_visualization_mesh(mesh, scatterers)
    vis_mesh.export(os.path.join(outdir, "visualization.ply"))
    
    # Create a README file explaining the outputs
    with open(os.path.join(outdir, "README.txt"), 'w') as f:
        f.write("Aether Radar Scattering Analysis Results\n")
        f.write("======================================\n\n")
        f.write("Files:\n")
        f.write("- scatterers.csv: CSV file with scatterer positions and scores\n")
        f.write("- scatterers.json: JSON file with detailed scatterer information\n")
        f.write("- visualization.ply: 3D model with color-coded scatterers\n\n")
        f.write("Color Legend for Visualization:\n")
        f.write("- Red to Yellow: Specular reflection points (intensity increases with yellowness)\n")
        f.write("- Blue to Cyan: Edge diffraction points (intensity increases with cyan-ness)\n")
        f.write("- Purple to Pink: Tip diffraction points (intensity increases with pink-ness)\n")
        f.write("- Light Gray: Non-scattering areas\n\n")
        f.write("To view the visualization.ply file, use a 3D viewer like MeshLab, Blender, or CloudCompare that supports PLY files with vertex colors.\n")
    
    print(f"Outputs written to {outdir}:")
    print(f"  - {os.path.join(outdir, 'scatterers.csv')}")
    print(f"  - {os.path.join(outdir, 'scatterers.json')}")
    print(f"  - {os.path.join(outdir, 'visualization.ply')}")
    print(f"  - {os.path.join(outdir, 'README.txt')}")

def export_to_csv(scatterers: List[Dict[str, Any]], filepath: str) -> None:
    """Export scatterers data to CSV format."""
    if not scatterers:
        print("Warning: No scatterers found to export")
        with open(filepath, 'w') as f:
            f.write("x,y,z,score,type\n")
        return
    
    # Create DataFrame with basic fields that all scatterers have
    data = {
        'x': [s['position'][0] for s in scatterers],
        'y': [s['position'][1] for s in scatterers],
        'z': [s['position'][2] for s in scatterers],
        'score': [s['score'] for s in scatterers],
        'type': [s['type'] for s in scatterers],
    }
    
    # Add face_idx if present in specular scatterers
    if all('face_idx' in s for s in scatterers if s['type'] == 'specular'):
        data['face_idx'] = [s.get('face_idx', -1) for s in scatterers]
    
    # Add edge_idx if present in edge scatterers
    if all('edge_idx' in s for s in scatterers if s['type'] == 'edge'):
        data['edge_idx'] = [s.get('edge_idx', -1) for s in scatterers]
    
    # Add vertex_idx if present in tip scatterers
    if all('vertex_idx' in s for s in scatterers if s['type'] == 'tip'):
        data['vertex_idx'] = [s.get('vertex_idx', -1) for s in scatterers]
    
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
        A new mesh with face colors representing scatterer intensity
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
            # Get normalized score
            if score_range <= 1e-10:
                normalized_score = 1.0  # All scores are equal, use max color
            else:
                normalized_score = (scatterer['score'] - min_score) / score_range
                
            # Create a color based on scatterer type and score
            if scatterer['type'] == 'specular':
                # Specular scatterers: Red to Yellow (higher intensity = more yellow)
                r = 255
                g = int(255 * normalized_score)
                b = 0
            elif scatterer['type'] == 'edge':
                # Edge scatterers: Blue to Cyan (higher intensity = more cyan)
                r = 0
                g = int(255 * normalized_score) 
                b = 255
            elif scatterer['type'] == 'tip':
                # Tip scatterers: Purple to Pink (higher intensity = more pink)
                r = int(128 + 127 * normalized_score)
                g = 0
                b = int(128 + 127 * normalized_score)
            else:
                # Unknown type: Grayscale
                r = g = b = int(128 + 127 * normalized_score)
            
            color = [r, g, b, 255]
            
            # Apply color based on scatterer type
            if 'face_idx' in scatterer:
                # For specular scatterers, color the face
                face_idx = scatterer['face_idx']
                face_colors[face_idx] = color
            elif 'edge_idx' in scatterer:
                # For edge scatterers, find the faces that share this edge
                edge_idx = scatterer['edge_idx']
                
                # Find the edge in mesh.edges
                edge = mesh.edges[edge_idx]
                # Find all faces that use this edge (contain both vertices)
                mask = np.zeros(len(mesh.faces), dtype=bool)
                for i in range(len(mesh.faces)):
                    # Check if both vertices of the edge are in this face
                    if edge[0] in mesh.faces[i] and edge[1] in mesh.faces[i]:
                        mask[i] = True
                
                connected_faces = np.where(mask)[0]
                for connected_face in connected_faces:
                    face_colors[connected_face] = color
            elif 'vertex_idx' in scatterer:
                # For tip scatterers, color all faces connected to the vertex
                vertex_idx = scatterer['vertex_idx']
                # Find all faces that use this vertex
                connected_faces = np.where(np.any(mesh.faces == vertex_idx, axis=1))[0]
                for connected_face in connected_faces:
                    face_colors[connected_face] = color
            
            # Safety check for numerical issues
            normalized_score = np.clip(normalized_score, 0.0, 1.0)
    
    # Apply colors to the mesh
    vis_mesh.visual.face_colors = face_colors
    
    # Print color stats and legend
    color_count = np.sum(np.any(face_colors != [200, 200, 200, 255], axis=1))
    print(f"Applied colors to {color_count} out of {len(face_colors)} faces")
    
    # Print color legend to explain what the colors mean
    print("\nVisualization Color Legend:")
    print("  • Red to Yellow: Specular reflection points (intensity increases with yellowness)")
    print("  • Blue to Cyan: Edge diffraction points (intensity increases with cyan-ness)")
    print("  • Purple to Pink: Tip diffraction points (intensity increases with pink-ness)")
    print("  • Light Gray: Non-scattering areas")
    
    return vis_mesh
