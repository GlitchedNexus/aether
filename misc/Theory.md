# Radar Scattering Theory & Implementation

This document outlines the core theoretical concepts, equations, and implementation details behind radar scattering analysis in the Aether project.

## Module Overview

The Aether package consists of the following core modules:

1. **io.py**: Mesh loading and validation

- Load STL/PLY/OBJ files
- Calculate face normals and properties
- Validate mesh quality metrics

2. **preprocess.py**: Mesh preparation

- Scale normalization
- Quality checks
- Manifold validation

3. **config.py**: Configuration

- Radar parameters (frequency, positions)
- Processing settings
- Material properties

4. **extract.py**: Scatterer detection

- Specular reflection points
- Edge diffraction sources
- Tip diffraction points

5. **weight.py**: Amplitude calculations

- Material-based coefficients
- Frequency scaling
- Polarization effects

6. **ranking.py**: Result processing

- Scatterer scoring
- Threshold filtering
- Top-K selection

7. **export.py**: Output generation

- CSV/JSON data export
- Visualization meshes
- Heatmap generation

---

## 1. Electromagnetic Fundamentals

### 1.1 Radar Cross-Section (RCS)

**Theory:** The RCS (σ) measures how detectable an object is by radar, defined as:

$$
\sigma = 4\pi \times \frac{\text{scattered power per unit solid angle}}{\text{incident power density}}
$$

For a flat plate at normal incidence:

$$
\sigma = \frac{4\pi A^2}{\lambda^2}
$$

Where:

- $A$ = physical area of the plate
- $\lambda$ = radar wavelength

**Implementation:**

- `weight.py` calculates normalized scattered power using:

$$
\sigma_\text{normalized} = \sigma_\text{physical} \times F(\theta, \phi) \times P(\text{pol})
$$

Where:

- $\sigma_\text{physical}$ = physical cross-section
- $F(\theta, \phi)$ = angular dependency function
- $P(\text{pol})$ = polarization factor

- `ranking.py` sorts scatterers using weighted RCS:

```python
# Normalize RCS values to [0,1] range
relative_rcs = weights / max(weights)
threshold_mask = relative_rcs > threshold
```

---

### 1.2 Scattering Mechanisms

**Theory:** Three primary scattering mechanisms with characteristic equations:

1. **Specular Reflection**

- Follows law of reflection: $\theta_i = \theta_r$
- Reflection direction: $\vec{r} = \vec{d} - 2(\vec{d} \cdot \hat{n})\hat{n}$

Where:

- $\vec{d}$ = incident direction vector
- $\hat{n}$ = surface normal unit vector
- $\vec{r}$ = reflected direction vector

2. **Edge Diffraction**

- Field amplitude $\propto 1/\sqrt{k}$ where $k = 2\pi/\lambda$
- Angular distribution follows UTD diffraction coefficient:

$$
D(\theta, \phi) = -\frac{e^{-j\pi/4}}{2n\sqrt{2\pi k}} \times \frac{1}{\sin(\beta)}
$$

Where:

- $n$ = wedge parameter ($n=2$ for straight edge)
- $\beta$ = angle between edge and observation direction

3. **Tip Diffraction**

- Field amplitude $\propto 1/k$
- Pattern follows spherical wave: $e^{-jkr}/r$

**Implementation:**

- `extract.py` contains detectors using geometric criteria:

```python
# Specular reflection detection
reflected = incident - 2 * np.dot(incident, normal) * normal
alignment = np.dot(reflected, rx_direction)
is_specular = alignment > cos(threshold_angle)

# Edge detection
edge_angle = arccos(np.dot(n1, n2))  # Between adjacent face normals
is_edge = edge_angle > edge_threshold

# Tip detection
vertex_curvature = 2 * np.pi - sum(face_angles)
is_tip = vertex_curvature > tip_threshold
```

---

## 2. Geometric Optics & Physical Optics

### 2.1 Reflection Geometry

**Theory:** For specular reflection, the angle of incidence equals the angle of reflection. Maximum reflection occurs when surface normal bisects incident and reflected rays.

The bisector vector $\vec{b}$ is computed as:

$$
\vec{b} = \frac{\vec{d}_i + \vec{d}_r}{|\vec{d}_i + \vec{d}_r|}
$$

Where:

- $\vec{d}_i$ = normalized incident direction vector
- $\vec{d}_r$ = normalized reflection direction vector

The specular reflection condition is met when:

$$
|\hat{n} \cdot \vec{b}| > \cos(\theta_t)
$$

Where:

- $\hat{n}$ = surface normal unit vector
- $\theta_t$ = threshold angle for specular detection

**Implementation:**

```python
# Calculate bisector vector
incident = (tx_pos - face_center) / norm(tx_pos - face_center)
reflected = (rx_pos - face_center) / norm(rx_pos - face_center)
bisector = (incident + reflected) / norm(incident + reflected)

# Check specular condition
alignment = abs(np.dot(face_normal, bisector))
is_specular = alignment > cos(threshold_angle)
```

---

### 2.2 Physical Theory of Diffraction (PTD)

**Theory:** Edge diffraction follows the Uniform Theory of Diffraction (UTD) with three key components:

1. **Field Amplitude:**

$$
E_d = E_i \times D(\theta, \phi) \times A(s) \times e^{-jks}
$$

Where:

- $E_d$ = diffracted field
- $E_i$ = incident field
- $D(\theta, \phi)$ = diffraction coefficient
- $A(s)$ = spatial attenuation factor
- $k = 2\pi/\lambda$ = wavenumber
- $s$ = distance from edge

2. **Diffraction Coefficient:**

$$
D(\theta, \phi) = -\frac{e^{-j\pi/4}}{2n\sqrt{2\pi k}} \times \frac{1}{\sin(\beta)}
$$

3. **Edge Detection Criteria:**

$$
\cos\theta = \hat{n}_1 \cdot \hat{n}_2 < \cos(\theta_e)
$$

Where:

- $\hat{n}_1, \hat{n}_2$ = normals of adjacent faces
- $\theta_e$ = edge detection threshold angle

**Implementation:**

```python
# Edge detection
def detect_edges(mesh, edge_angle_threshold):
  # Get face adjacency
  edges_face = mesh.edges_face
  face_normals = mesh.face_normals

  # Calculate dihedral angles
  edge_angles = []
  for edge_idx, (f1, f2) in enumerate(edges_face):
    if f2 == -1:  # Boundary edge
      continue
    cos_theta = np.dot(face_normals[f1], face_normals[f2])
    edge_angles.append(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

  # Identify sharp edges
  sharp_edges = edge_angles > edge_angle_threshold
  return sharp_edges

# Weight calculation
def compute_edge_weight(edge_vectors, wavelength, observation_angles):
  k = 2 * np.pi / wavelength
  spatial_factor = 1.0 / np.sqrt(k * edge_length)
  angular_factor = compute_utd_coefficient(k, observation_angles)
  return spatial_factor * angular_factor
```

---

## 3. Computational Electromagnetics

### 3.1 Discretization & Mesh Representation

**Theory:** Continuous surfaces are approximated with triangular facets. Accuracy depends on mesh resolution relative to wavelength.

The minimum mesh resolution requirement is:

$$
\Delta x < \lambda / 10
$$

Where:

- $\Delta x$ = maximum edge length of triangles
- $\lambda$ = radar wavelength

Face area normalization:

$$
A_\text{norm} = \frac{A}{\lambda^2}
$$

Where:

- $A$ = physical face area
- $A_\text{norm}$ = normalized area for frequency scaling

**Implementation:**

```python
def normalize_mesh(mesh, wavelength):
  # Scale check
  max_edge = max(mesh.edges_length)
  if max_edge > wavelength/10:
    warnings.warn("Mesh resolution may be insufficient")

  # Area normalization
  face_areas = mesh.area_faces
  norm_areas = face_areas / (wavelength * wavelength)
  return norm_areas
```

---

### 3.2 Frequency Considerations

**Theory:** Valid approximations depend on feature size relative to wavelength:

1. **High Frequency Regime** ($\lambda \ll$ object size):

$$
kL \gg 1 \quad (k = 2\pi/\lambda,\, L = \text{object size})
$$

- Physical Optics (PO) approximation valid
- Ray-optical methods applicable
- Edge diffraction follows UTD

2. **Low Frequency Regime** ($\lambda \sim$ object size):

$$
kL \sim 1
$$

- Full-wave methods needed
- Method of Moments (MoM) or FDTD required
- Not implemented in current version

**Implementation:**

```python
def check_frequency_regime(mesh, freq_ghz):
  wavelength = SPEED_OF_LIGHT / (freq_ghz * 1e9)
  k = 2 * np.pi / wavelength

  # Characteristic size (max dimension)
  L = max(mesh.extents)
  kL = k * L

  if kL < 10:
    warnings.warn("Object too small for high-freq. approximation")
    return "low_frequency"
  return "high_frequency"
```

---

### 3.3 Mesh Quality Metrics

Key quality parameters monitored by `preprocess.py`:

1. **Triangle Aspect Ratio:**

$$
AR = \frac{\text{max\_edge}}{\text{min\_height}}
$$

Requirement: $AR < 5$ for reliable computations

2. **Normal Consistency:**

- All face normals point outward
- Computed using right-hand rule on vertex ordering

3. **Manifold Check:**

- Each edge shared by exactly 2 faces
- No isolated or duplicate vertices

---

## 4. Amplitude Weighting

### 4.1 Scattering Coefficients

**Theory:** The total scattered field amplitude is computed as:

$$
E_s = E_i \times (R_s + D_e + D_t)
$$

Where:

- $E_s$ = scattered field
- $E_i$ = incident field
- $R_s$ = specular reflection coefficient
- $D_e$ = edge diffraction coefficient
- $D_t$ = tip diffraction coefficient

Each component is weighted based on:

1. **Specular Reflection:**

$$
R_s = \Gamma \times \sqrt{\frac{A}{\lambda}} \times F(\theta, \phi)
$$

Where:

- $\Gamma$ = Fresnel reflection coefficient
- $A$ = physical area of facet
- $F(\theta, \phi)$ = angular pattern function

2. **Edge Diffraction:**

$$
D_e = -\frac{e^{-j\pi/4}}{2n\sqrt{2\pi k}} \times \frac{1}{\sin(\beta)} \times \frac{L}{\lambda}
$$

Where:

- $n$ = wedge parameter
- $L$ = edge length
- $\beta$ = diffraction angle

3. **Tip Diffraction:**

$$
D_t = -\frac{e^{-jk_0 r}}{r} \times F(\theta, \phi)
$$

Where:

- $r$ = distance to observation point
- $k_0$ = free space wavenumber

---

### 4.2 Polarization Effects

**Theory:** Polarization modifies scattering through matrix operations:

$$
\begin{bmatrix}
E_\parallel \\
E_\perp
\end{bmatrix}
=
\begin{bmatrix}
S_{11} & S_{12} \\
S_{21} & S_{22}
\end{bmatrix}
\begin{bmatrix}
E_{i\parallel} \\
E_{i\perp}
\end{bmatrix}
$$

Where:

- $E_\parallel, E_\perp$ = parallel/perpendicular field components
- $S_{ij}$ = scattering matrix elements

Implementation in `weight.py`:

```python
def apply_polarization_factor(weights, polarization='VV'):
  # Polarization factors from scattering matrix
  pol_factors = {
    'VV': 1.0,      # Vertical-Vertical
    'HH': 0.8,      # Horizontal-Horizontal
    'HV': 0.2,      # Horizontal-Vertical (cross-pol)
    'VH': 0.2       # Vertical-Horizontal (cross-pol)
  }
  return weights * pol_factors[polarization]
```

---

## 5. Ranking and Selection

### 5.1 Scatterer Ranking

**Theory:** Multiple ranking methods implemented:

1. **Absolute Threshold:**

$$
S = \{ s \mid \sigma(s) > T_\text{abs} \}
$$

2. **Relative Threshold:**

$$
S = \{ s \mid \sigma(s) > \alpha \times \max(\sigma) \}
$$

3. **Top-K Selection:**

$$
S = \text{sort}(\sigma)[0:k]
$$

Implementation in `ranking.py`:

```python
def rank_scatterers(weights, method='top_k', **params):
  if method == 'abs_threshold':
    mask = weights > params['threshold']
  elif method == 'rel_threshold':
    mask = weights > params['alpha'] * np.max(weights)
  else:  # top_k
    indices = np.argsort(weights)[-params['k']:]
    mask = np.zeros_like(weights, dtype=bool)
    mask[indices] = True
  return mask
```

---

### 5.2 Computational Efficiency

**Theory:** Optimizations implemented:

1. **Spatial Indexing:**

- kd-tree for nearest neighbor searches
- $O(\log n)$ complexity vs $O(n)$ for naive search

2. **Vectorized Operations:**

- Numpy arrays for bulk computations
- Minimizes Python loop overhead

3. **Early Pruning:**

- Reject faces based on normal orientation
- Skip detailed computations for non-contributing elements

Implementation example:

```python
def optimize_computation(mesh, tx_pos, rx_pos):
  # Build spatial index
  spatial_index = scipy.spatial.cKDTree(mesh.vertices)

  # Vectorized normal check
  v_tx = mesh.face_centroids - tx_pos
  v_rx = mesh.face_centroids - rx_pos

  # Dot product with normals (vectorized)
  dp_tx = np.einsum('ij,ij->i', v_tx, mesh.face_normals)
  dp_rx = np.einsum('ij,ij->i', v_rx, mesh.face_normals)

  # Early pruning mask
  valid_faces = (dp_tx < 0) & (dp_rx < 0)

  return valid_faces
```
