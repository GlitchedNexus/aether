"""
Simple benchmark for specular detection performance.
"""

import time
import trimesh
import numpy as np
from aether.extract import detect_specular


def benchmark_specular_detection(mesh_path, freq, tx, rx, num_runs=5):
    """Benchmark the specular detection on a given mesh."""
    mesh = trimesh.load(mesh_path)
    
    # Run multiple times and average the results
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        scatterers = detect_specular(mesh, freq, tx, rx)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Print results
    print(f"\nBenchmark results for {mesh_path}:")
    print(f"  Face count: {len(mesh.faces)}")
    print(f"  Scatterers found: {len(scatterers)}")
    print(f"  Average time: {avg_time:.4f} seconds")
    print(f"  Min time: {min_time:.4f} seconds")
    print(f"  Max time: {max_time:.4f} seconds")
    
    return {
        "mesh_path": mesh_path,
        "face_count": len(mesh.faces),
        "scatterer_count": len(scatterers),
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time
    }


if __name__ == "__main__":
    # Create a simple cube for benchmarking
    cube = trimesh.creation.box(extents=[1, 1, 1])
    cube_path = "benchmarks/gold/cube.stl"
    cube.export(cube_path)
    
    # Create a sphere with increasing resolution for scaling tests
    for resolution in [10, 50, 100]:
        sphere = trimesh.creation.icosphere(subdivisions=resolution)
        sphere_path = f"benchmarks/gold/sphere_{len(sphere.faces)}_faces.stl"
        sphere.export(sphere_path)
    
    # Configuration
    freq = 10.0  # GHz
    tx = [5, 5, 5]
    rx = [-5, -5, -5]
    
    # Run benchmarks
    results = []
    results.append(benchmark_specular_detection(cube_path, freq, tx, rx))
    
    for resolution in [10, 50, 100]:
        sphere_path = f"benchmarks/gold/sphere_{resolution * resolution * 8}_faces.stl"
        results.append(benchmark_specular_detection(sphere_path, freq, tx, rx))
    
    # Compare scaling
    print("\nScaling comparison:")
    for i in range(1, len(results)):
        face_ratio = results[i]["face_count"] / results[0]["face_count"]
        time_ratio = results[i]["avg_time"] / results[0]["avg_time"]
        print(f"  {results[i]['face_count']} faces: {face_ratio:.1f}x faces → {time_ratio:.1f}x time")
