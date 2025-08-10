# Radar Scattering Theory & Implementation

This document outlines the core theoretical concepts behind radar scattering analysis and their implementation in the Aether project.

## 1. Electromagnetic Fundamentals

### 1.1 Radar Cross-Section (RCS)

**Theory**: The RCS (σ) measures how detectable an object is by radar, defined as:

σ = 4π × (scattered power per unit solid angle / incident power density)

**Implementation**:

- `weight.py` calculates normalized scattered power based on angle and material properties
- `ranking.py` sorts detected scatterers by their effective RCS contribution

### 1.2 Scattering Mechanisms

**Theory**: Three primary scattering mechanisms:

1. **Specular Reflection**: Mirror-like reflection from flat surfaces
2. **Edge Diffraction**: Scattering from sharp edges
3. **Tip Diffraction**: Scattering from sharp points/corners

**Implementation**:

- `extract.py` contains separate detectors for each mechanism:
  - `detect_specular()` finds surfaces with normals aligned to the bisector
  - `detect_edges()` identifies sharp edges using adjacency information
  - `detect_tips()` finds vertices with high curvature (multiple incident edges)

## 2. Geometric Optics & Physical Optics

### 2.1 Reflection Geometry

**Theory**: For specular reflection, the angle of incidence equals the angle of reflection. Maximum reflection occurs when surface normal bisects incident and reflected rays.

**Implementation**:

- `extract.py` computes the bisector vector between transmitter and receiver
- Performs dot product between face normals and bisector to identify specular points

### 2.2 Physical Theory of Diffraction (PTD)

**Theory**: Edge diffraction intensity follows:

- Angular dependence on incident and observation angles
- Proportional to wavelength
- Inversely proportional to distance from edge

**Implementation**:

- `extract.py` identifies edges using dihedral angles between adjacent faces
- `weight.py` applies frequency-dependent weighting to edge contributions

## 3. Computational Electromagnetics

### 3.1 Discretization & Mesh Representation

**Theory**: Continuous surfaces are approximated with triangular facets. Accuracy depends on mesh resolution relative to wavelength.

**Implementation**:

- `io.py` loads and validates mesh quality
- `preprocess.py` ensures mesh is suitable for EM calculations:
  - Removes duplicate vertices
  - Repairs non-manifold edges
  - Normalizes scale relative to wavelength

### 3.2 Frequency Considerations

**Theory**: Valid approximations depend on feature size relative to wavelength:

- High frequency → physical optics valid (λ << object size)
- Low frequency → full-wave methods needed (not implemented)

**Implementation**:

- `config.py` defines radar frequency and calculates wavelength
- Warnings issued when mesh resolution inadequate for chosen frequency

## 4. Amplitude Weighting

### 4.1 Scattering Coefficients

**Theory**: Amplitude of scattered field depends on:

- Material properties (permittivity, conductivity)
- Polarization of incident wave
- Local geometry (curvature, orientation)

**Implementation**:

- `weight.py` applies amplitude factors based on:

  - Specular: Fresnel reflection coefficients (material-dependent)
  - Edge: Diffraction coefficients from asymptotic theory
  - Tip: Corner diffraction coefficients

### 4.2 Polarization Effects

**Theory**: Scattering behavior depends on incident wave polarization and surface properties.

**Implementation**:

- `config.py` defines polarization state
- `weight.py` calculates polarization-dependent responses

## 5. Heatmap Generation

### 5.1 Spatial Distribution Visualization

**Theory**: Scatter points can be visualized by projecting onto 2D views with intensity encoding.

**Implementation**:

- `export.py` generates heatmaps showing scatter intensity distribution
- Uses intensity weighting from `ranking.py`
- Projects 3D points onto standard views (top, side, front)

## 6. Algorithm Performance Considerations

### 6.1 Computational Complexity

**Theory**: Naive scatterer detection is O(n) with number of mesh elements, but spatial partitioning can improve this.

**Implementation**:

- `extract.py` uses spatial indexing for transmitter/receiver ray casting
- JIT compilation with Numba accelerates hotspot functions
- Future optimization path includes Rust implementations for core compute kernels

### 6.2 Accuracy vs. Performance Tradeoffs

**Theory**: Higher mesh resolution increases accuracy but computational cost.

**Implementation**:

- `ranking.py` implements top-k selection to focus on dominant scatterers
- CLI offers resolution control parameters
