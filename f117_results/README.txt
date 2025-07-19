Aether Radar Scattering Analysis Results
======================================

Files:
- scatterers.csv: CSV file with scatterer positions and scores
- scatterers.json: JSON file with detailed scatterer information
- visualization.ply: 3D model with color-coded scatterers

Color Legend for Visualization:
- Red to Yellow: Specular reflection points (intensity increases with yellowness)
  * Pure red: Low-intensity specular reflection
  * Yellow (red+green): High-intensity specular reflection
- Blue to Cyan: Edge diffraction points (intensity increases with cyan-ness)
  * Pure blue: Low-intensity edge diffraction
  * Cyan (blue+green): High-intensity edge diffraction
- Purple to Pink: Tip diffraction points (intensity increases with pink-ness)
  * Purple: Low-intensity tip diffraction
  * Pink: High-intensity tip diffraction
- Light Gray: Non-scattering areas

To view the visualization.ply file, use a 3D viewer like MeshLab, Blender, or CloudCompare that supports PLY files with vertex colors.
