"""
Integration test for the CLI interface.
"""

import os
import pytest
import subprocess
import tempfile
import shutil
import trimesh
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def cube_mesh_path(temp_dir):
    """Create a simple cube and return its path."""
    cube = trimesh.creation.box(extents=[1, 1, 1])
    
    # Save to the temp directory
    path = os.path.join(temp_dir, "cube.stl")
    cube.export(path)
    
    return path


def test_cli_help():
    """Test that the CLI help command works."""
    # Run the CLI with --help
    result = subprocess.run(
        ["python", "cli.py", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )
    
    # Check that it returns a success code
    assert result.returncode == 0
    
    # Check that the output contains expected text
    assert "Aether" in result.stdout
    assert "Usage:" in result.stdout


def test_cli_analyse(cube_mesh_path, temp_dir):
    """Test the analyse command."""
    output_dir = os.path.join(temp_dir, "output")
    
    # Run the analyse command
    result = subprocess.run(
        [
            "python", "cli.py", "analyse",
            cube_mesh_path,
            "--freq", "10",
            "--tx", "2", "2", "2",
            "--rx", "-2", "-2", "-2",
            "--outdir", output_dir
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )
    
    # Check for successful execution
    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
    
    # Check for output files
    assert os.path.exists(os.path.join(output_dir, "scatterers.csv"))
    assert os.path.exists(os.path.join(output_dir, "scatterers.json"))
    assert os.path.exists(os.path.join(output_dir, "visualization.ply"))


def test_cli_legacy(cube_mesh_path, temp_dir):
    """Test the legacy command for backward compatibility."""
    output_dir = os.path.join(temp_dir, "legacy_output")
    
    # Run the legacy command
    result = subprocess.run(
        [
            "python", "cli.py", "legacy",
            cube_mesh_path,
            "--freq", "10",
            "--tx", "2", "2", "2",
            "--rx", "-2", "-2", "-2",
            "--outdir", output_dir
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False
    )
    
    # Check for successful execution
    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
    
    # Check for output files
    assert os.path.exists(os.path.join(output_dir, "scatterers.csv"))
    assert os.path.exists(os.path.join(output_dir, "scatterers.json"))
    assert os.path.exists(os.path.join(output_dir, "visualization.ply"))
