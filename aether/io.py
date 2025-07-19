"""
Aether IO Module

This module provides mesh loading and cleaning routines.
"""

import trimesh
import numpy as np
import os
from typing import Tuple, Dict, Any

def calculate_face_normals(mesh):
    """Calculate face normals if they're not provided by trimesh"""
    # For each face, get the vertices
    triangles = mesh.triangles
    
    # Calculate two edges for each face
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    
    # Cross product gives the normal vector
    normals = np.cross(edge1, edge2)
    
    # Normalize the normals
    lengths = np.sqrt(np.sum(normals**2, axis=1))
    # Prevent division by zero
    valid = lengths > 1e-10
    if not all(valid):
        print(f"Warning: {np.sum(~valid)} degenerate faces found with zero-length normal")
    
    # Only normalize valid normals, set others to [0,0,1]
    normals_out = np.zeros_like(normals)
    normals_out[valid] = normals[valid] / lengths[valid].reshape(-1, 1)
    normals_out[~valid] = [0, 0, 1]  # Default normal for degenerate faces
    
    return normals_out

def load_mesh(path: str) -> trimesh.Trimesh:
    """
    Load a mesh from `path` and return a trimesh.Trimesh object.
    
    Args:
        path: Path to the mesh file (STL, OBJ, PLY, etc.)
    
    Returns:
        A trimesh.Trimesh object representing the loaded mesh
    """
    try:
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mesh file not found: {path}")
        
        # Try to import scipy to see if it's available (needed by trimesh for some operations)
        try:
            import scipy
            has_scipy = True
        except ImportError:
            has_scipy = False
            print("Warning: scipy not found. Some mesh operations may be limited.")
        
        # Basic load operation
        mesh = trimesh.load_mesh(path)
        
        # Basic validation
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Loaded file is not a triangle mesh")
        
        print(f"Loaded mesh with {len(mesh.faces)} faces and {len(mesh.vertices)} vertices")
        
        # Always ensure normals are calculated first
        if mesh.face_normals is None or len(mesh.face_normals) == 0:
            print("Calculating face normals...")
            mesh.face_normals = calculate_face_normals(mesh)
        
        # Process the mesh according to available dependencies
        if has_scipy:
            # Full processing with scipy
            try:
                # Merge duplicate vertices
                # Don't assign the result directly as it might be None in some versions
                result = mesh.merge_vertices()
                if result is not None:
                    mesh = result
                
                # Ensure consistent face winding only if the mesh has normals
                if hasattr(mesh, 'fix_normals'):
                    mesh.fix_normals()
            except Exception as e:
                print(f"Warning: Advanced mesh processing failed: {str(e)}")
                print("Falling back to basic processing.")
        else:
            # Simple processing without scipy
            # Try to merge vertices manually if possible
            try:
                # Merge duplicate vertices using numpy operations
                unique_vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
                if len(unique_vertices) < len(mesh.vertices):
                    print("Merging duplicate vertices without scipy...")
                    # Create new faces with updated indices
                    new_faces = inverse[mesh.faces]
                    mesh = trimesh.Trimesh(vertices=unique_vertices, faces=new_faces)
                    # Ensure we copy over the face normals
                    if hasattr(mesh, 'face_normals') and mesh.face_normals is None:
                        mesh.face_normals = calculate_face_normals(mesh)
            except Exception as e:
                print(f"Warning: Failed to merge vertices: {str(e)}")
        
        # After merging vertices, also remove duplicate faces
        
        # Remove duplicate faces by first converting them to a structured array
        if len(mesh.faces) > 0:
            # First, sort each face's vertex indices to normalize the order
            sorted_faces = np.sort(mesh.faces, axis=1)
            # Use structured array to identify unique faces
            sorted_faces_view = sorted_faces.view(np.dtype((np.void, sorted_faces.dtype.itemsize * sorted_faces.shape[1])))
            unique_faces_idx = np.unique(sorted_faces_view, return_index=True)[1]
            
            # If we found duplicate faces, recreate the mesh with only unique faces
            if len(unique_faces_idx) < len(mesh.faces):
                print(f"Removing {len(mesh.faces) - len(unique_faces_idx)} duplicate faces")
                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[unique_faces_idx])
        
        return mesh
    except Exception as e:
        raise RuntimeError(f"Failed to load mesh from {path}: {str(e)}")