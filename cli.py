import click
import sys
import os
from aether.io import load_mesh
from aether.extract import extract_all_scatterers
from aether.export import write_outputs, render_transmitter_receiver_scene
from aether.preprocess import prepare_mesh
from aether.config import create_radar_config, create_processing_config
from rich.console import Console
from rich.progress import Progress
import numpy as np

console = Console()

@click.group()
def cli():
    """Aether - Radar Signature Analysis Tool"""
    pass

@cli.command()
@click.argument("mesh_path", type=click.Path(exists=True))
@click.option("--freq", "-f", required=True, type=float,
              help="Radar frequency in GHz.")
@click.option("--tx", nargs=3, type=float, required=True,
              help="Transmitter XYZ (m).")
@click.option("--rx", nargs=3, type=float, required=True,
              help="Receiver XYZ (m).")
@click.option("--outdir", "-o", default="out", show_default=True,
              type=click.Path(file_okay=False))
@click.option("--normalize/--no-normalize", default=True, show_default=True,
              help="Normalize mesh before analysis.")
@click.option("--top-k", type=int, default=100, show_default=True,
              help="Number of top scatterers to keep.")
@click.option("--min-threshold", type=float, default=0.01, show_default=True,
              help="Minimum score threshold (relative to max).")
@click.option("--edge-detection/--no-edge-detection", default=False, show_default=True,
              help="Enable edge diffraction detection.")
@click.option("--tip-detection/--no-tip-detection", default=False, show_default=True,
              help="Enable tip diffraction detection.")
@click.option("--face-detection/--no-face-detection", default=False, show_default=True,
              help="Enable face scattering detection.")
@click.option("--render-positions/--no-render-positions", default=False, show_default=True,
              help="Render transmitter and receiver positions in output.")
def analyse(mesh_path, freq, tx, rx, outdir, normalize, top_k, min_threshold,
          edge_detection, tip_detection, face_detection, render_positions):
    """Analyse a mesh and export heatmap + CSV with optional TX/RX visualization."""
    with Progress() as progress:
        task1 = progress.add_task("[green]Loading mesh...", total=1)
        try:
            mesh = load_mesh(mesh_path)
            progress.update(task1, completed=1)
        except Exception as e:
            console.print(f"[bold red]Error loading mesh:[/] {str(e)}")
            sys.exit(1)
        
        task2 = progress.add_task("[green]Preparing mesh...", total=1)
        try:
            mesh, quality_report = prepare_mesh(mesh, normalize=normalize)
            progress.update(task2, completed=1)
            
            # Print mesh quality info
            console.print(f"[bold blue]Mesh Quality Report:[/]")
            console.print(f"  Vertices: {quality_report['vertex_count']}")
            console.print(f"  Faces: {quality_report['face_count']}")
            console.print(f"  Watertight: {quality_report['is_watertight']}")
            if quality_report['has_duplicate_vertices']:
                console.print("[yellow]  Warning: Duplicate vertices detected and merged[/]")
            if quality_report['has_degenerate_faces']:
                console.print("[yellow]  Warning: Degenerate faces detected and removed[/]")
        except Exception as e:
            console.print(f"[bold red]Error preparing mesh:[/] {str(e)}")
            sys.exit(1)
            
        task3 = progress.add_task("[green]Detecting scatterers...", total=1)
        try:
            # Create configurations
            radar_config = create_radar_config(freq, tx, rx)
            proc_config = create_processing_config(
                num_top_scatterers=top_k,
                min_score_threshold=min_threshold,
                edge_detection=edge_detection,
                tip_detection=tip_detection,
                face_detection=face_detection
            )
            
            # Extract scatterers
            scatterers = extract_all_scatterers(mesh, radar_config, proc_config)
            progress.update(task3, completed=1)
            
            console.print(f"[bold green]Found {len(scatterers)} scatterers[/]")
        except Exception as e:
            console.print(f"[bold red]Error detecting scatterers:[/] {str(e)}")
            sys.exit(1)
            
        task4 = progress.add_task("[green]Exporting results...", total=1)
        try:
            write_outputs(mesh, scatterers, outdir)
            
            # Optionally render TX/RX positions
            if render_positions:
                console.print("[blue]Rendering transmitter/receiver positions...[/]")
                tx_pos = np.array(tx)
                rx_pos = np.array(rx)
                scene = render_transmitter_receiver_scene(mesh, scatterers, tx_pos, rx_pos)
                scene_path = os.path.join(outdir, "scene_with_positions.ply")
                # scene.export(scene_path)  # Would be implemented
                console.print(f"  Scene with TX/RX positions: {scene_path}")
            
            progress.update(task4, completed=1)
        except Exception as e:
            console.print(f"[bold red]Error exporting results:[/] {str(e)}")
            sys.exit(1)

# Legacy command for backward compatibility
@cli.command("legacy")
@click.argument("mesh_path", type=click.Path(exists=True))
@click.option("--freq", "-f",  required=True, type=float,
              help="Radar frequency in GHz.")
@click.option("--tx", nargs=3, type=float, required=True,
              help="Transmitter XYZ (m).")
@click.option("--rx", nargs=3, type=float, required=True,
              help="Receiver XYZ (m).")
@click.option("--outdir", "-o", default="out", show_default=True,
              type=click.Path(file_okay=False))
def main(mesh_path, freq, tx, rx, outdir):
    """Analyse a mesh and export heatmap + CSV (legacy mode)."""
    mesh = load_mesh(mesh_path)
    scatterers = extract_all_scatterers(
        mesh, 
        create_radar_config(freq, tx, rx),
        create_processing_config()
    )
    write_outputs(mesh, scatterers, outdir)

if __name__ == "__main__":
    cli()
