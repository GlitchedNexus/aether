# 🧰 1. Project Setup & Tooling

1. **Version Control**

   - Initialize a Git repo:

     ```bash
     git init
     ```

   - Add a `.gitignore` for Python artifacts, IDE files, venvs, etc.

2. **Language & Environment**

   - **Python 3.10+** for modern typing support
   - Use **pyenv** (optional) and create a virtualenv:

     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

3. **Formatting & Linting**

   - **Black**: `black .`
   - **isort**: `isort .`
   - **flake8**: `flake8`

4. **Dependency Management**

   - Use **pyproject.toml** with [Poetry](https://python-poetry.org/) or **pip‑tools**
   - Pin exact versions for reproducibility

---

## 📦 2. Dependency Matrix

| Package                | Purpose                                           | Mandatory? | Alternatives / Trade‑Offs              |
| ---------------------- | ------------------------------------------------- | :--------: | -------------------------------------- |
| **trimesh**            | Mesh I/O, cleaning, adjacency queries             |    Yes     | Combine `meshio` + manual ops          |
| **numpy**              | Vectorized math (arrays, dot, cross, transforms)  |    Yes     | Pure Python loops (→ much slower)      |
| **numba**              | JIT‑accelerate tight loops (point classification) |    Opt.    | Omit if meshes <100K tris; slower code |
| **click**              | User‑friendly CLI parsing                         |    Yes     | Built‑in `argparse` (more boilerplate) |
| **pytest**             | Unit & integration testing                        |    Yes     | `unittest` (more verbose)              |
| **pandas**             | Tabular export & analysis of scatterer data       |    Opt.    | Write CSV manually                     |
| **rich**               | Pretty CLI output, progress bars                  |    Opt.    | Plain `print` / progressbar            |
| **black/isort/flake8** | Code style & linting                              |    Yes     | —                                      |

> **Why each?**
>
> - **trimesh** + **numpy**: end‑to‑end mesh pipeline with minimal code.
> - **numba**: accelerate hotspots when needed.
> - **click**: clean, composable CLI.
> - **pytest**: industry standard for Python testing.
> - **rich**: enhances UX for progress and errors.

---

## 🏗 3. High‑Level Module Layout

```
aether_py/
├── aether/
│   ├── io.py            # mesh loading & cleaning
│   ├── preprocess.py    # scale normalization & QC
│   ├── config.py        # radar scenario definitions
│   ├── extract.py       # specular, edge, tip detection
│   ├── weight.py        # amplitude weighting
│   ├── ranking.py       # threshold / top‑k selection
│   └── export.py        # heatmap & CSV output
├── cli.py               # click‑based entry point
├── benchmarks/          # simple scripts & meshes for perf tests
├── tests/
│   ├── unit/            # pytest unit tests
│   └── integration/     # end‑to‑end tests on sample meshes
└── pyproject.toml
```

---

## 🚀 4. Development Workflow

1. **Scaffold**

   - Create the directory structure above.
   - Stub out each module with clear function signatures and docstrings.

2. **Mesh I/O & Preprocess**

   - Implement `io.load_mesh(path) → trimesh.Trimesh` with cleaning steps.
   - Unit‑test for inverted normals, duplicate vertices, etc.

3. **Simple Extractor**

   - In `extract.py`, write a pure‑Python specular detector that picks triangles whose normals face the receiver.
   - Test on a single‑triangle mesh with known analytic result.

4. **CLI & Integration**

   - Wire up `cli.py` to call: load → preprocess → extract → export.
   - Write an end‑to‑end pytest:

     ```bash
     pytest tests/integration/test_plate.py
     ```

     which runs `python -m aether analyse sample/plate.stl`.

5. **Weighting & Ranking**

   - Implement `weight.compute_polarization(...)` and `ranking.topk(...)`.
   - Unit‑test expected ordering on synthetic inputs.

6. **Benchmark & Profile**

   - Add scripts in `benchmarks/` using `timeit` or `pytest-benchmark`.
   - Identify hotspots: triangle loops, large sorts, I/O.

7. **Optimize**

   - Decorate slow functions with `@numba.njit` and re‑benchmark.
   - Mark remaining slow routines for eventual Rust port.

---

## 🧪 5. Testing & CI

- **pytest** with coverage:

  - `tests/unit` for pure functions
  - `tests/integration` for full‑pipeline runs

- **CI Pipeline** (GitHub Actions example):

  1. Checkout & setup Python
  2. `pip install .[dev]`
  3. `black --check . && isort --check . && flake8`
  4. `pytest --cov=aether`
  5. Upload coverage badge

- **Golden Data**:

  - Store a cube, sphere, and simple wing in `benchmarks/gold/`.
  - Keep expected CSVs for regression.

---

## 📈 6. Benchmarking & Profiling

1. **Micro‑benchmarks**

   ```python
   import timeit
   setup = "from aether.extract import specular; mesh = load_mesh('...')"
   print(timeit.timeit("specular(mesh, ...)", setup=setup, number=10))
   ```

2. **Line Profiling**

   - Install: `pip install line_profiler`
   - Add `@profile` to suspect functions
   - Run: `kernprof -l -v benchmarks/profile_plate.py`

3. **Memory Profiling**

   - Use `memory_profiler` on large meshes to find leaks or bloat.

---

## ⚙️ 7. Transition Plan: Python → Rust

| Step | Action                                                                  |
| ---- | ----------------------------------------------------------------------- |
| 1    | **Identify Hotspots** from profiling (e.g. specular, edge diffraction). |
| 2    | **Create Rust crate** (`aether-core`) mirroring Python API signatures.  |
| 3    | **Port & validate** a small function (e.g. normal‑vector dot product).  |
| 4    | **Expose via PyO3**:                                                    |

```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version="0.18", features=["extension-module"] }
```

```rust
#[pyfunction]
fn extract_specular(...) -> PyResult<Vec<(f64,f64,f64,f64)>> { … }
```

\| 5 | **Publish wheels** with **maturin**, update Python build config accordingly. |
\| 6 | **Iterate**: port next hotspot, keeping Python orchestration intact. |

---

## 🎓 8. Further Reading & Resources

- **PyO3 Guide**: [https://pyo3.rs/](https://pyo3.rs/)
- **Maturin**: [https://github.com/PyO3/maturin](https://github.com/PyO3/maturin)
- **Numba User Guide**: [https://numba.pydata.org/numba-doc/latest/](https://numba.pydata.org/numba-doc/latest/)
- **pytest‑benchmark**: [https://pytest-benchmark.readthedocs.io/](https://pytest-benchmark.readthedocs.io/)
- **Rust FFI Patterns**: [https://doc.rust-lang.org/nomicon/ffi.html](https://doc.rust-lang.org/nomicon/ffi.html)

---

This roadmap will let you **prototype rapidly**, **validate thoroughly**, and **incrementally swap in Rust** where it matters—while keeping the developer experience smooth and consistent.
