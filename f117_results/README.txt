Aether Radar Scattering Analysis Results
======================================

Files:
- scatterers.csv: CSV file with scatterer positions and scores
- scatterers.json: JSON file with detailed scatterer information
- visualization.ply: 3D model with color-coded scatterers

Enhanced Color Legend for Visualization:
========================================

SPECULAR REFLECTION (Red to Yellow gradient):
- Pure Red (255,0,0): Minimal specular reflection
- Orange-Red (255,64,0): Low specular reflection
- Orange (255,128,0): Moderate specular reflection
- Yellow-Orange (255,192,0): High specular reflection
- Pure Yellow (255,255,0): Maximum specular reflection
→ Represents mirror-like reflections from flat surfaces

EDGE DIFFRACTION (Blue to Cyan gradient):
- Pure Blue (0,0,255): Minimal edge diffraction
- Blue-Cyan (0,64,255): Low edge diffraction
- Medium Cyan (0,128,255): Moderate edge diffraction
- Bright Cyan (0,192,255): High edge diffraction
- Pure Cyan (0,255,255): Maximum edge diffraction
→ Represents scattering from sharp edges and ridges

TIP DIFFRACTION (Purple to Pink gradient):
- Deep Purple (128,0,128): Minimal tip diffraction
- Purple (160,0,160): Low tip diffraction
- Magenta (192,0,192): Moderate tip diffraction
- Light Pink (224,0,224): High tip diffraction
- Bright Pink (255,0,255): Maximum tip diffraction
→ Represents scattering from sharp points and corners

TRANSMITTER/RECEIVER MARKERS:
- Bright Green Sphere: Transmitter position
- Bright Red Sphere: Receiver position
- Semi-transparent lines: Line-of-sight connections

NON-SCATTERING AREAS:
- Light Gray (200,200,200): Areas with minimal radar response

Intensity Scale: Color intensity directly correlates with Radar Cross Section (RCS) contribution.
Higher intensity colors indicate stronger radar returns and greater detectability.

Technical Notes:
- Specular scatterers dominate at high frequencies and smooth surfaces
- Edge diffraction becomes prominent at surface discontinuities
- Tip diffraction is strongest at sharp geometric features
- Multiple scattering mechanisms may contribute at the same location

Viewing Instructions:
To view the visualization.ply file, use a 3D viewer that supports PLY files with vertex colors:
- MeshLab (recommended): Free, cross-platform mesh processing tool
- Blender: Advanced 3D modeling software with import capabilities
- CloudCompare: Point cloud and mesh analysis software
- ParaView: Scientific data visualization tool
