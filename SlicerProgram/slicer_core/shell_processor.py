import csv
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from numpy.typing import NDArray

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

from .config import SlicerConfig
from .settings_manager import SettingsManager

logger = logging.getLogger(__name__)

_OUTPUT_SIGNATURE_KEYS = (
    "knn_k",
    "sphere_min_diameter",
    "sphere_max_diameter",
    "overlap_factor",
    "centreline_samples",
    "plane_spacing",
    "shell_point_spacing",
    "start_z",
    "start_z_tolerance",
    "spline_s",
    "centreline_extension_length",
    "tree_sphere_graph_k",
    "graph_strategy",
    "enable_nurbs",
    "adaptive_plane_spacing",
    "adaptive_spacing_min_factor",
    "outer_corner_spacing_safety",
    "max_point_distance_from_centreline",
    "limit_curve_bbox",
    "curve_bbox_padding_ratio",
    "nurbs_surface_count",
    "nurbs_angle_tolerance_deg",
    "nurbs_min_points",
)


class SlicerRunCancelled(RuntimeError):
    """Raised when the user interrupts a slicer run."""


def _setting_value(settings_source: Any, key: str, default: Any = None) -> Any:
    getter = getattr(settings_source, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def _sanitize_folder_token(value: Any, fallback: str = "unknown", max_length: int = 48) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or fallback)).strip("._-")
    if not token:
        token = fallback
    if len(token) > max_length:
        token = token[:max_length].rstrip("._-")
    return token or fallback


def _format_number_token(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _sanitize_folder_token(value)

    if abs(number) < 1e-9:
        number = 0.0

    if abs(number - round(number)) < 1e-9:
        token = str(int(round(number)))
    else:
        token = f"{number:.3f}".rstrip("0").rstrip(".")
    return token.replace("-", "m").replace(".", "p")


def _format_sample_token(value: Any) -> str:
    try:
        samples = int(round(float(value)))
    except (TypeError, ValueError):
        return _sanitize_folder_token(value)

    if samples >= 1000 and samples % 1000 == 0:
        return f"{samples // 1000}k"
    return str(samples)


def _build_readable_settings_slug(settings_source: Any) -> str:
    parts = [
        f"K{_format_number_token(_setting_value(settings_source, 'knn_k', 0))}",
        f"Dmin{_format_number_token(_setting_value(settings_source, 'sphere_min_diameter', 0))}",
        f"Dmax{_format_number_token(_setting_value(settings_source, 'sphere_max_diameter', 0))}",
        f"OF{_format_number_token(_setting_value(settings_source, 'overlap_factor', 0))}",
        f"s{_format_sample_token(_setting_value(settings_source, 'centreline_samples', 0))}",
        f"PS{_format_number_token(_setting_value(settings_source, 'plane_spacing', 0))}",
        f"SP{_format_number_token(_setting_value(settings_source, 'shell_point_spacing', 0))}",
    ]

    start_z = _setting_value(settings_source, "start_z", 0.0)
    try:
        if abs(float(start_z)) > 1e-9:
            parts.append(f"SZ{_format_number_token(start_z)}")
    except (TypeError, ValueError):
        parts.append(f"SZ{_sanitize_folder_token(start_z)}")

    return "_".join(parts)


def _build_settings_signature(
    settings_source: Any,
    line_method: str,
    sphere_generation_method: str,
) -> str:
    payload = {
        key: _setting_value(settings_source, key)
        for key in _OUTPUT_SIGNATURE_KEYS
    }
    payload["line_method"] = line_method
    payload["sphere_generation_method"] = sphere_generation_method
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:8]


def build_output_dir(
    base_output_dir: Path,
    stl_file: Path,
    settings_source: Any,
    line_method: Optional[str] = None,
    sphere_generation_method: Optional[str] = None,
) -> Path:
    base_output_dir = Path(base_output_dir)
    stl_file = Path(stl_file)
    line_token = _sanitize_folder_token(
        line_method or _setting_value(settings_source, "line_method", "auto")
    )
    sphere_token = _sanitize_folder_token(
        sphere_generation_method
        or _setting_value(settings_source, "sphere_generation_method", "auto")
    )
    mesh_token = _sanitize_folder_token(stl_file.stem, fallback="mesh", max_length=32)
    readable_slug = _build_readable_settings_slug(settings_source)
    signature = _build_settings_signature(
        settings_source,
        line_method=line_token,
        sphere_generation_method=sphere_token,
    )
    run_folder = f"{mesh_token}_{readable_slug}_{signature}"
    return base_output_dir / line_token / sphere_token / run_folder


def _write_run_parameters_csv(
    output_path: Path,
    settings: SettingsManager,
    summary: Dict[str, Any],
):
    rows = [
        ("run_timestamp", summary.get("run_timestamp", "")),
        ("stl_file", summary.get("stl_file", "")),
        ("runtime_seconds", f"{summary.get('runtime_seconds', 0.0):.6f}"),
    ]

    for key in sorted(summary):
        if key in {"branch_metadata", "run_timestamp", "stl_file", "runtime_seconds"}:
            continue
        rows.append((f"summary.{key}", summary[key]))

    for key in sorted(settings.settings):
        rows.append((f"setting.{key}", settings.settings[key]))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        writer.writerows(rows)


@dataclass
class ShellPassResult:
    pass_index: int
    slices: List[trimesh.path.path.Path]
    origins: NDArray
    normals: NDArray
    guide_points: NDArray
    pos_dense_centerline: Optional[NDArray]
    shell_points: NDArray
    plane_index: NDArray
    plane_fragment_index: Optional[NDArray]
    plane_distances: NDArray
    spheres: List[Tuple[NDArray, float]] = field(default_factory=list)
    branch_index: Optional[NDArray] = None
    plane_branch_index: Optional[NDArray] = None
    centrelines: List[Tuple[int, NDArray]] = field(default_factory=list)
    surface_curves: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def _iteratively_smooth_centreline(
    csp,
    centreline: NDArray,
    base_smoothing: float,
    centreline_samples: int,
    extra_passes: int,
) -> NDArray:
    if extra_passes <= 0:
        return np.asarray(centreline, dtype=float)

    smoothed = np.asarray(centreline, dtype=float)
    for pass_idx in range(extra_passes):
        smoothing = max(base_smoothing, 1e-6) * (1.0 + 0.5 * (pass_idx + 1))
        candidate = csp.fit_branch_centreline(
            smoothed,
            spline_s=smoothing,
            centreline_samples=max(centreline_samples, len(smoothed)),
            preserve_endpoints=True,
        )
        if candidate is None or len(candidate) < 2:
            break
        smoothed = np.asarray(candidate, dtype=float)
    return smoothed


def _trim_polyline_to_expanded_bbox(
    polyline: NDArray,
    mesh: trimesh.Trimesh,
    padding_ratio: float = 0.1,
) -> NDArray:
    polyline = np.asarray(polyline, dtype=float)
    if len(polyline) == 0:
        return polyline

    bounds = np.asarray(mesh.bounds, dtype=float)
    mins = bounds[0]
    maxs = bounds[1]
    extents = np.maximum(maxs - mins, 1e-9)
    padding = extents * float(max(padding_ratio, 0.0))
    lower = mins - padding
    upper = maxs + padding

    clipped = np.clip(polyline, lower, upper)

    if len(clipped) < 2:
        return clipped

    # Remove only exact consecutive duplicates introduced by clipping,
    # but do not trim away whole leading/trailing sections.
    keep_mask = np.ones(len(clipped), dtype=bool)
    keep_mask[1:] = np.any(np.abs(np.diff(clipped, axis=0)) > 1e-9, axis=1)
    filtered = clipped[keep_mask]
    if len(filtered) < 2:
        return clipped
    return filtered


def _import_centreline_module():
    try:
        import numpy as _np
        import scipy.interpolate as scipy_interpolate

        if (
            not hasattr(scipy_interpolate, "make_splprep")
            and hasattr(scipy_interpolate, "splprep")
            and hasattr(scipy_interpolate, "splev")
        ):
            def _compat_make_splprep(*args, **kwargs):
                tck, u = scipy_interpolate.splprep(*args, **kwargs)

                class _SplineWrapper:
                    def __call__(self, values):
                        return _np.asarray(scipy_interpolate.splev(values, tck))

                return _SplineWrapper(), u

            scipy_interpolate.make_splprep = _compat_make_splprep
    except Exception as exc:
        logger.debug("SciPy compatibility alias could not be installed: %s", exc)

    from slicer_algorithms import centreline_shell_points as csp

    return csp


def normalize_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (PACKAGE_ROOT / path).resolve()
    return path


def _frame_axes(normal: NDArray) -> Tuple[NDArray, NDArray]:
    normal = np.asarray(normal, dtype=float)
    normal = normal / max(np.linalg.norm(normal), 1e-12)

    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)

    axis_u = np.cross(normal, reference)
    axis_u = axis_u / max(np.linalg.norm(axis_u), 1e-12)
    axis_v = np.cross(normal, axis_u)
    axis_v = axis_v / max(np.linalg.norm(axis_v), 1e-12)
    return axis_u, axis_v


def build_plane_paths(
    shell_points: NDArray,
    plane_index: NDArray,
    plane_origins: NDArray,
    plane_normals: NDArray,
    plane_fragment_index: Optional[NDArray] = None,
    cancel_callback: Optional[Callable[[], None]] = None,
) -> List[trimesh.path.path.Path]:
    paths: List[trimesh.path.path.Path] = []

    for idx, origin in enumerate(plane_origins):
        if cancel_callback is not None and idx % 25 == 0:
            cancel_callback()
        points = np.asarray(shell_points[plane_index == idx], dtype=float)
        if len(points) < 2:
            continue

        axis_u, axis_v = _frame_axes(plane_normals[idx])
        rel = points - origin
        angles = np.arctan2(rel @ axis_v, rel @ axis_u)
        ordered = points[np.argsort(angles)]

        if len(ordered) > 2:
            ordered = np.vstack([ordered, ordered[0]])

        try:
            paths.append(trimesh.load_path(ordered))
        except Exception as exc:
            logger.debug("Skipping plane %s path reconstruction: %s", idx, exc)

    return paths


def _compact_planes(
    shell_points: NDArray,
    plane_index: NDArray,
    plane_origins: NDArray,
    plane_normals: NDArray,
    plane_distances: NDArray,
    plane_branch_index: Optional[NDArray] = None,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, Optional[NDArray]]:
    if len(plane_origins) == 0:
        return (
            shell_points,
            plane_index,
            plane_origins,
            plane_normals,
            plane_distances,
            plane_branch_index,
        )

    used_planes = np.unique(np.asarray(plane_index, dtype=int))
    if len(used_planes) == len(plane_origins):
        return (
            shell_points,
            plane_index,
            plane_origins,
            plane_normals,
            plane_distances,
            plane_branch_index,
        )

    remap = -np.ones(len(plane_origins), dtype=int)
    remap[used_planes] = np.arange(len(used_planes))
    compact_plane_index = remap[np.asarray(plane_index, dtype=int)]
    compact_branch_index = (
        np.asarray(plane_branch_index, dtype=int)[used_planes]
        if plane_branch_index is not None
        else None
    )
    return (
        shell_points,
        compact_plane_index,
        np.asarray(plane_origins)[used_planes],
        np.asarray(plane_normals)[used_planes],
        np.asarray(plane_distances)[used_planes],
        compact_branch_index,
    )


def _plane_spacing_stats(
    plane_distances: NDArray,
    plane_branch_index: Optional[NDArray] = None,
) -> Dict[str, float]:
    plane_distances = np.asarray(plane_distances, dtype=float)
    if len(plane_distances) < 2:
        return {}

    diffs: List[float] = []
    if plane_branch_index is None:
        diffs = np.diff(plane_distances).tolist()
    else:
        plane_branch_index = np.asarray(plane_branch_index, dtype=int)
        for branch_id in np.unique(plane_branch_index):
            branch_distances = plane_distances[plane_branch_index == branch_id]
            if len(branch_distances) >= 2:
                diffs.extend(np.diff(branch_distances).tolist())

    diffs_array = np.asarray([diff for diff in diffs if diff > 1e-9], dtype=float)
    if len(diffs_array) == 0:
        return {}

    return {
        "plane_spacing_min": float(np.min(diffs_array)),
        "plane_spacing_max": float(np.max(diffs_array)),
        "plane_spacing_mean": float(np.mean(diffs_array)),
    }


class CentrelineShellSlicer:
    def __init__(self, settings: SettingsManager, run_time_config: SlicerConfig):
        self.settings = settings
        self.run_config = run_time_config
        self.output_root = Path(run_time_config.output_dir)
        self.mesh: Optional[trimesh.Trimesh] = None
        self.pass_results: List[ShellPassResult] = []
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.run_config.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _log_config(self):
        logger.info("--- SlicerProgram Config ---")
        logger.info("STL File: %s", self.run_config.stl_file)
        logger.info("Line Method: %s", self.settings.get("line_method"))
        logger.info(
            "Sphere Generation Method: %s",
            self.settings.get("sphere_generation_method", "auto"),
        )
        logger.info("Plane Spacing: %.3f", self.settings.get("plane_spacing"))
        logger.info(
            "Shell Point Spacing: %.3f", self.settings.get("shell_point_spacing")
        )
        logger.info("Start Z: %.3f", self.settings.get("start_z"))
        logger.info(
            "Centreline Extension Length: %.3f",
            self.settings.get("centreline_extension_length", 0.0),
        )
        logger.info("Repeat Run Smoothing Passes: %s", self.settings.get("rerun_smoothing_passes", 0))
        logger.info("--------------------------------")

    def run(
        self,
        progress_callback: Optional[Callable[[int], None]] = None,
        cancel_callback: Optional[Callable[[], None]] = None,
    ):
        run_started = time.perf_counter()
        if progress_callback:
            progress_callback(0)

        if cancel_callback:
            cancel_callback()
        self._log_config()
        csp = _import_centreline_module()
        mesh_path = normalize_path(str(self.run_config.stl_file))
        logger.info("Loading mesh from %s", mesh_path)
        self.mesh = csp.load_mesh(str(mesh_path))

        if progress_callback:
            progress_callback(10)

        if cancel_callback:
            cancel_callback()
        requested_sphere_method = self.settings.get("sphere_generation_method", "auto")
        spheres, effective_sphere_method = csp.compute_hitbox_spheres(
            self.mesh,
            knn_k=self.settings.get("knn_k"),
            overlap_factor=self.settings.get("overlap_factor"),
            min_diameter=self.settings.get("sphere_min_diameter", 0.0),
            max_diameter=self.settings.get("sphere_max_diameter", 0.0),
            sphere_generation_method=requested_sphere_method,
            return_method=True,
            cancel_callback=cancel_callback,
        )
        logger.info(
            "Requested sphere generation: %s -> using %s",
            requested_sphere_method,
            effective_sphere_method,
        )
        requested_method = self.settings.get("line_method", "auto")
        effective_method = requested_method
        if requested_method == "auto":
            effective_method = (
                "tree"
                if csp.detect_branching_from_spheres(
                    spheres,
                    sphere_graph_k=self.settings.get("tree_sphere_graph_k"),
                    graph_strategy=self.settings.get("graph_strategy", "mst"),
                )
                else "single"
            )
        logger.info(
            "Requested line method: %s -> using %s",
            requested_method,
            effective_method,
        )
        resolved_output_dir = build_output_dir(
            base_output_dir=self.output_root,
            stl_file=mesh_path,
            settings_source=self.settings,
            line_method=effective_method,
            sphere_generation_method=effective_sphere_method,
        )
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        self.run_config = SlicerConfig(
            stl_file=self.run_config.stl_file,
            log_dir=self.run_config.log_dir,
            output_dir=resolved_output_dir,
            plot_filename_prefix=self.run_config.plot_filename_prefix,
        )
        logger.info("Run outputs will be saved to %s", self.run_config.output_dir)

        if progress_callback:
            progress_callback(35)

        if cancel_callback:
            cancel_callback()
        max_point_distance = self.settings.get(
            "max_point_distance_from_centreline",
            0.0,
        )
        centreline_extension_length = self.settings.get(
            "centreline_extension_length",
            0.0,
        )

        if effective_method == "single":
            logger.info(
                "Building a single centreline from %s spheres.",
                len(spheres),
            )
            centreline = csp.fit_single_sphere_centreline(
                spheres,
                sphere_graph_k=self.settings.get("tree_sphere_graph_k"),
                graph_strategy=self.settings.get("graph_strategy", "mst"),
                spline_s=self.settings.get("spline_s"),
                centreline_samples=self.settings.get("centreline_samples"),
            )
            centreline = csp.extend_centreline_to_mesh_ends(
                centreline,
                self.mesh,
            )
            centreline = _iteratively_smooth_centreline(
                csp,
                centreline,
                base_smoothing=self.settings.get("spline_s"),
                centreline_samples=self.settings.get("centreline_samples"),
                extra_passes=self.settings.get("rerun_smoothing_passes", 0),
            )
            centreline = csp.extend_centreline_by_length(
                centreline,
                extension_length=centreline_extension_length,
            )
            if self.settings.get("limit_curve_bbox", True):
                centreline = _trim_polyline_to_expanded_bbox(
                    centreline,
                    self.mesh,
                    padding_ratio=self.settings.get("curve_bbox_padding_ratio", 0.1),
                )
            centreline, centreline_start_arc = csp.trim_centreline_from_z(
                centreline,
                start_z=self.settings.get("start_z"),
                start_z_tolerance=self.settings.get("start_z_tolerance"),
            )
            if progress_callback:
                progress_callback(45)
            logger.info("Sampling shell points along the single centreline.")
            (
                shell_points,
                plane_index,
                plane_fragment_index,
                plane_origins,
                plane_normals,
                plane_distances,
                start_arc,
            ) = csp.slice_shell_points_from_faces(
                self.mesh,
                centreline,
                plane_spacing=self.settings.get("plane_spacing"),
                shell_point_spacing=self.settings.get("shell_point_spacing"),
                start_z=self.settings.get("start_z"),
                adaptive_plane_spacing=self.settings.get("adaptive_plane_spacing", True),
                adaptive_spacing_min_factor=self.settings.get("adaptive_spacing_min_factor", 0.6),
                outer_corner_spacing_safety=self.settings.get("outer_corner_spacing_safety", 0.35),
                max_point_distance_from_centreline=max_point_distance,
                cancel_callback=cancel_callback,
                progress_label="Single-centreline slicing",
            )
            (
                shell_points,
                plane_index,
                plane_origins,
                plane_normals,
                plane_distances,
                plane_branch_index,
            ) = _compact_planes(
                shell_points,
                plane_index,
                plane_origins,
                plane_normals,
                plane_distances,
                plane_branch_index=np.zeros(len(plane_origins), dtype=int),
            )
            centrelines = [(0, centreline)]
            branch_index = np.zeros(len(shell_points), dtype=int)
            branch_metadata = [
                {
                    "branch_id": 0,
                    "plane_count": len(plane_origins),
                    "trimmed_by": float(centreline_start_arc),
                }
            ]
            logger.info(
                "Single-centreline slicing produced %s planes and %s shell points.",
                len(plane_origins),
                len(shell_points),
            )
        else:
            logger.info(
                "Building tree branch centrelines from %s spheres.",
                len(spheres),
            )
            branch_centrelines = csp.build_tree_branch_centrelines(
                spheres,
                sphere_graph_k=self.settings.get("tree_sphere_graph_k"),
                graph_strategy=self.settings.get("graph_strategy", "mst"),
                spline_s=self.settings.get("spline_s"),
                centreline_samples=self.settings.get("centreline_samples"),
            )
            logger.info(
                "Built %s tree branch centreline(s).",
                len(branch_centrelines),
            )
            if progress_callback:
                progress_callback(45)

            def _report_branch_progress(stage, branch_pos, branch_count, branch_id, details):
                branch_number = int(branch_pos) + 1
                total_branches = max(int(branch_count), 1)

                if stage == "start":
                    logger.info(
                        "Processing branch %s/%s (id=%s, length %.2f mm).",
                        branch_number,
                        total_branches,
                        branch_id,
                        float(details.get("centreline_length", 0.0)),
                    )
                    return

                if stage == "skipped":
                    logger.warning(
                        "Skipped branch %s/%s (id=%s): %s",
                        branch_number,
                        total_branches,
                        branch_id,
                        details.get("error", "no shell points found"),
                    )
                elif stage == "complete":
                    logger.info(
                        "Finished branch %s/%s (id=%s): %s planes, %s shell points.",
                        branch_number,
                        total_branches,
                        branch_id,
                        int(details.get("plane_count", 0)),
                        int(details.get("shell_point_count", 0)),
                    )

                if progress_callback:
                    progress_callback(
                        min(
                            69,
                            45 + int(round(25.0 * branch_number / total_branches)),
                        )
                    )

            if len(branch_centrelines) == 1:
                branch_id, branch_centreline = branch_centrelines[0]
                branch_centrelines = [
                    (
                        branch_id,
                        csp.extend_centreline_to_mesh_ends(
                            branch_centreline,
                            self.mesh,
                        ),
                    )
                ]
            extra_smoothing = self.settings.get("rerun_smoothing_passes", 0)
            if extra_smoothing > 0:
                branch_centrelines = [
                    (
                        branch_id,
                        _iteratively_smooth_centreline(
                            csp,
                            centreline,
                            base_smoothing=self.settings.get("spline_s"),
                            centreline_samples=self.settings.get("centreline_samples"),
                            extra_passes=extra_smoothing,
                        ),
                    )
                    for branch_id, centreline in branch_centrelines
                ]
            branch_centrelines = csp.extend_branch_centrelines_by_length(
                branch_centrelines,
                extension_length=centreline_extension_length,
            )
            if self.settings.get("limit_curve_bbox", True):
                branch_centrelines = [
                    (
                        branch_id,
                        _trim_polyline_to_expanded_bbox(
                            centreline,
                            self.mesh,
                            padding_ratio=self.settings.get("curve_bbox_padding_ratio", 0.1),
                        ),
                    )
                    for branch_id, centreline in branch_centrelines
                ]
            (
                shell_points,
                plane_index,
                plane_fragment_index,
                branch_index,
                plane_origins,
                plane_normals,
                plane_distances,
                plane_branch_index,
                branch_metadata,
            ) = csp.slice_shell_points_from_branch_centrelines(
                self.mesh,
                branch_centrelines,
                plane_spacing=self.settings.get("plane_spacing"),
                shell_point_spacing=self.settings.get("shell_point_spacing"),
                start_z=self.settings.get("start_z"),
                start_z_tolerance=self.settings.get("start_z_tolerance"),
                adaptive_plane_spacing=self.settings.get("adaptive_plane_spacing", True),
                adaptive_spacing_min_factor=self.settings.get("adaptive_spacing_min_factor", 0.6),
                outer_corner_spacing_safety=self.settings.get("outer_corner_spacing_safety", 0.35),
                max_point_distance_from_centreline=max_point_distance,
                cancel_callback=cancel_callback,
                branch_progress_callback=_report_branch_progress,
            )
            (
                shell_points,
                plane_index,
                plane_origins,
                plane_normals,
                plane_distances,
                plane_branch_index,
            ) = _compact_planes(
                shell_points,
                plane_index,
                plane_origins,
                plane_normals,
                plane_distances,
                plane_branch_index=plane_branch_index,
            )
            branch_plane_counts = {
                int(branch_id): int(np.count_nonzero(plane_branch_index == branch_id))
                for branch_id in np.unique(plane_branch_index)
            }
            for meta in branch_metadata:
                meta["plane_count"] = branch_plane_counts.get(meta["branch_id"], 0)
            centrelines = [
                (meta["branch_id"], meta["centreline"]) for meta in branch_metadata
            ]
            start_arc = 0.0
            centreline_start_arc = 0.0
            logger.info(
                "Tree slicing produced %s planes and %s shell points across %s branches.",
                len(plane_origins),
                len(shell_points),
                len(centrelines),
            )

        if progress_callback:
            progress_callback(70)

        if cancel_callback:
            cancel_callback()
        if self.settings.get("enable_nurbs", True):
            logger.info("Generating NURBS-like surface curves.")
            surface_curves = csp.generate_nurbs_like_surface_intersections(
                centrelines,
                plane_origins,
                plane_normals,
                shell_points,
                plane_index,
                num_surfaces=self.settings.get("nurbs_surface_count"),
                angle_tolerance_deg=self.settings.get("nurbs_angle_tolerance_deg"),
                min_points_per_surface=self.settings.get("nurbs_min_points"),
                branch_index=branch_index,
                plane_branch_index=plane_branch_index,
                cancel_callback=cancel_callback,
            )
        else:
            logger.info("NURBS surface generation disabled.")
            surface_curves = []
        if progress_callback:
            progress_callback(85)

        logger.info("Reconstructing plane paths for visualization.")
        slices = build_plane_paths(
            shell_points,
            plane_index,
                plane_origins,
                plane_normals,
                cancel_callback=cancel_callback,
                plane_fragment_index=plane_fragment_index,
            )
        logger.info("Reconstructed %s plane paths.", len(slices))
        if progress_callback:
            progress_callback(92)

        runtime_seconds = time.perf_counter() - run_started
        spacing_stats = _plane_spacing_stats(plane_distances, plane_branch_index)
        summary = {
            "run_timestamp": self.run_timestamp,
            "stl_file": str(mesh_path),
            "runtime_seconds": float(runtime_seconds),
            "line_method": effective_method,
            "requested_line_method": requested_method,
            "sphere_generation_method": effective_sphere_method,
            "requested_sphere_generation_method": requested_sphere_method,
            "output_directory": str(self.run_config.output_dir),
            "graph_strategy": self.settings.get("graph_strategy", "mst"),
            "sphere_min_diameter": self.settings.get("sphere_min_diameter", 0.0),
            "sphere_max_diameter": self.settings.get("sphere_max_diameter", 0.0),
            "sphere_count": len(spheres),
            "plane_count": len(plane_origins),
            "shell_point_count": len(shell_points),
            "surface_curve_count": len(surface_curves),
            "max_point_distance_from_centreline": max_point_distance,
            "centreline_extension_length": centreline_extension_length,
            "branch_count": len(centrelines),
            "start_arc": float(start_arc),
            "centreline_start_arc": float(centreline_start_arc),
            "rerun_smoothing_passes": self.settings.get("rerun_smoothing_passes", 0),
            "limit_curve_bbox": self.settings.get("limit_curve_bbox", True),
            "curve_bbox_padding_ratio": self.settings.get("curve_bbox_padding_ratio", 0.1),
            "branch_metadata": branch_metadata,
        }
        summary.update(spacing_stats)

        layers_json_path = (
            self.run_config.output_dir
            / f"shell_layers_{self.run_timestamp}.json"
            if self.settings.get("export_layers_json", True)
            else None
        )
        if layers_json_path is not None:
            summary["shell_layers_json"] = str(layers_json_path)

        if cancel_callback:
            cancel_callback()
        logger.info("Exporting run artifacts.")
        if self.settings.get("export_csv", True):
            csv_path = (
                self.run_config.output_dir
                / f"shell_points_{self.run_timestamp}.csv"
            )
            parameter_csv_path = (
                self.run_config.output_dir
                / f"run_parameters_{self.run_timestamp}.csv"
            )
            summary["shell_points_csv"] = str(csv_path)
            summary["run_parameters_csv"] = str(parameter_csv_path)

            csp.export_shell_points_csv(
                csv_path,
                shell_points,
                plane_index,
                plane_distances,
                branch_index=branch_index,
            )
            logger.info("Saved shell-point CSV to %s", csv_path)

            _write_run_parameters_csv(
                parameter_csv_path,
                self.settings,
                summary,
            )
            logger.info("Saved run-parameter CSV to %s", parameter_csv_path)

        if cancel_callback:
            cancel_callback()
        if layers_json_path is not None:
            csp.export_shell_layers_json(
                layers_json_path,
                shell_points,
                plane_index,
                plane_origins,
                plane_normals,
                plane_distances,
                plane_fragment_index=plane_fragment_index,
                branch_index=branch_index,
                plane_branch_index=plane_branch_index,
                metadata={
                    "run_timestamp": self.run_timestamp,
                    "stl_file": str(mesh_path),
                    "line_method": effective_method,
                    "sphere_generation_method": effective_sphere_method,
                    "plane_spacing": self.settings.get("plane_spacing"),
                    "shell_point_spacing": self.settings.get("shell_point_spacing"),
                    "max_point_distance_from_centreline": max_point_distance,
                    "centreline_extension_length": centreline_extension_length,
                },
            )
            logger.info("Saved layer JSON to %s", layers_json_path)
        if progress_callback:
            progress_callback(97)

        result = ShellPassResult(
            pass_index=1,
            slices=slices,
            origins=plane_origins,
            normals=plane_normals,
            guide_points=plane_origins,
            pos_dense_centerline=centrelines[0][1] if centrelines else None,
            shell_points=shell_points,
            plane_index=plane_index,
            plane_fragment_index=plane_fragment_index,
            plane_distances=plane_distances,
            spheres=spheres,
            branch_index=branch_index,
            plane_branch_index=plane_branch_index,
            centrelines=centrelines,
            surface_curves=surface_curves,
            summary=summary,
        )
        self.pass_results = [result]

        logger.info("Centreline shell run finished successfully.")
        logger.info("Branches: %s", result.summary["branch_count"])
        logger.info("Planes: %s", result.summary["plane_count"])
        logger.info("Shell points: %s", result.summary["shell_point_count"])
        logger.info("Runtime: %.3f seconds", result.summary["runtime_seconds"])

        if progress_callback:
            progress_callback(100)
