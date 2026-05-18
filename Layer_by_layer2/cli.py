from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .extractor import (
        LayerByLayer2Config,
        extract_layers_from_mesh,
        plot_layers,
        save_layers_csv,
        save_layers_json,
    )
except ImportError:
    import sys

    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from Layer_by_layer2.extractor import (
        LayerByLayer2Config,
        extract_layers_from_mesh,
        plot_layers,
        save_layers_csv,
        save_layers_json,
    )


def find_default_mesh_path() -> Path | None:
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "STLfiles" / "CustomEdgecaseThesis.stl",
        repo_root / "SlicerProgram" / "mesh" / "CustomEdgecaseThesis.stl",
        repo_root / "STLfiles" / "EdgeCaseSplit.stl",
        repo_root / "SlicerProgram" / "mesh" / "EdgeCaseSplit.stl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_mesh_path(mesh_arg: str) -> Path:
    if mesh_arg:
        return Path(mesh_arg)

    default_mesh = find_default_mesh_path()
    if default_mesh is not None:
        return default_mesh

    raise FileNotFoundError(
        "No mesh path was provided and no default demo mesh could be found. "
        "Pass --mesh <path-to-stl>."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the improved layer-by-layer extraction on an STL mesh."
    )
    parser.add_argument(
        "--mesh",
        default="",
        help=(
            "Path to the STL mesh. If omitted, the script tries to use "
            "CustomEdgecaseThesis.stl from the repository."
        ),
    )
    parser.add_argument("--json", default="", help="Optional JSON output path.")
    parser.add_argument("--csv", default="", help="Optional CSV output path.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting.")
    parser.add_argument("--hide-centroids", action="store_true", help="Do not draw the centroid polyline.")
    parser.add_argument("--hide-rejected", action="store_true", help="Do not draw rejected points.")
    parser.add_argument("--show-legend", action="store_true", help="Draw the plot legend.")
    parser.add_argument("--legend-stride", type=int, default=10, help="Show legend entries for every Nth layer.")
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable preprocessing-based outlier removal before layer extraction.",
    )
    parser.add_argument("--preprocess-support-radius", type=float, default=1.5)
    parser.add_argument("--preprocess-support-min-other", type=int, default=1)
    parser.add_argument("--preprocess-component-radius", type=float, default=2.5)
    parser.add_argument("--preprocess-min-component-points", type=int, default=12)

    parser.add_argument("--first-layer-min-points", type=int, default=1000)
    parser.add_argument("--first-layer-dz-init", type=float, default=0.5)
    parser.add_argument("--first-layer-dz-step", type=float, default=0.5)
    parser.add_argument("--first-layer-dz-max", type=float, default=2.0)

    parser.add_argument("--adjacency-radius", type=float, default=4.0)
    parser.add_argument("--adjacency-radius-max", type=float, default=10.0)
    parser.add_argument("--adjacency-radius-step", type=float, default=0.2)
    parser.add_argument("--isolation-radius", type=float, default=1.5)
    parser.add_argument("--isolation-min-other", type=int, default=1)
    parser.add_argument("--component-radius", type=float, default=2.5)
    parser.add_argument(
        "--component-mode",
        choices=("largest", "closest", "largest_then_closest"),
        default="largest_then_closest",
    )
    parser.add_argument("--plane-inlier-tolerance", type=float, default=0.75)
    parser.add_argument("--plane-refine-steps", type=int, default=2)
    parser.add_argument("--min-layer-points", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=1000)
    parser.add_argument("--round-decimals", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mesh_path = resolve_mesh_path(args.mesh)

    config = LayerByLayer2Config(
        preprocess_enabled=not args.no_preprocess,
        preprocess_support_radius=args.preprocess_support_radius,
        preprocess_support_min_other=args.preprocess_support_min_other,
        preprocess_component_radius=args.preprocess_component_radius,
        preprocess_min_component_points=args.preprocess_min_component_points,
        first_layer_min_points=args.first_layer_min_points,
        first_layer_dz_init=args.first_layer_dz_init,
        first_layer_dz_step=args.first_layer_dz_step,
        first_layer_dz_max=args.first_layer_dz_max,
        adjacency_radius=args.adjacency_radius,
        adjacency_radius_max=args.adjacency_radius_max,
        adjacency_radius_step=args.adjacency_radius_step,
        isolation_radius=args.isolation_radius,
        isolation_min_other=args.isolation_min_other,
        component_radius=args.component_radius,
        component_mode=args.component_mode,
        plane_inlier_tolerance=args.plane_inlier_tolerance,
        plane_refine_steps=args.plane_refine_steps,
        min_layer_points=args.min_layer_points,
        max_layers=args.max_layers,
        round_decimals=args.round_decimals,
    )

    result = extract_layers_from_mesh(mesh_path, config=config)

    print(f"Loaded mesh: {mesh_path}")
    print(f"Original mesh vertices: {result.original_point_count}")
    print(f"Vertices after preprocessing: {result.preprocessed_point_count}")
    print(f"Preprocessing removed points: {result.preprocessing_removed_point_count}")
    print(f"Extracted layers: {len(result.layers)}")
    print(f"Rejected candidate points: {len(result.rejected_points)}")
    print(f"Remaining unassigned points: {result.remaining_point_count}")
    print(f"First layer threshold z: {result.first_layer_threshold_z:.4f}")
    print(f"Runtime: {result.runtime_seconds:.4f} s")

    if args.json:
        save_layers_json(args.json, result)
        print(f"Saved JSON: {Path(args.json)}")

    if args.csv:
        save_layers_csv(args.csv, result)
        print(f"Saved CSV: {Path(args.csv)}")

    if not args.no_plot:
        plot_layers(
            result,
            show_centroids=not args.hide_centroids,
            show_rejected=not args.hide_rejected,
            show_legend=args.show_legend,
            legend_stride=args.legend_stride,
        )


if __name__ == "__main__":
    main()
