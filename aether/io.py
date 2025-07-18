# aether/io.py

import trimesh
import numpy as np

def load_mesh(path: str) -> trimesh.Trimesh:
    """
    Load a mesh from `path` and return a trimesh.Trimesh object.
    
    Args:
        path: Path to the mesh file (STL, OBJ, PLY, etc.)
    
    Returns:
        A trimesh.Trimesh object representing the loaded mesh
    """
    try:
        # Basic load operation
        mesh = trimesh.load_mesh(path)
        
        # Basic validation
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Loaded file is not a triangle mesh")
        
        print(f"Loaded mesh with {len(mesh.faces)} faces and {len(mesh.vertices)} vertices")
        
        # Skip complicated cleaning operations for now
        # Just make sure we have normals
        if mesh.face_normals is None or len(mesh.face_normals) == 0:
            print("Calculating face normals...")
            # Simple normal calculation if needed
            mesh.face_normals = calculate_face_normals(mesh)
            
        return mesh
    except Exception as e:
        raise RuntimeError(f"Failed to load mesh from {path}: {str(e)}")

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
    normals = normals / lengths.reshape(-1, 1)
    
    return normals
