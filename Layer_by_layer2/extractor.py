from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import trimesh
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LayerByLayer2Config:
    preprocess_enabled: bool = True
    preprocess_support_radius: float = 1.5
    preprocess_support_min_other: int = 1
    preprocess_component_radius: float = 2.5
    preprocess_min_component_points: int = 12
    first_layer_min_points: int = 1000
    first_layer_dz_init: float = 0.5
    first_layer_dz_step: float = 0.5
    first_layer_dz_max: float = 2.0
    adjacency_radius: float = 4.0
    adjacency_radius_max: float = 10.0
    adjacency_radius_step: float = 0.2
    isolation_radius: float = 1.5
    isolation_min_other: int = 1
    component_radius: float = 2.5
    component_mode: str = "largest_then_closest"
    keep_all_coplanar_components: bool = True
    plane_inlier_tolerance: float = 0.75
    plane_refine_steps: int = 2
    min_layer_points: int = 8
    max_layers: int = 1000
    round_decimals: int | None = None


@dataclass
class LayerByLayer2Result:
    layers: list[np.ndarray]
    centroids: list[np.ndarray]
    normals: list[np.ndarray]
    search_radii: list[float | None]
    component_sizes: list[int]
    preprocessed_point_count: int
    preprocessing_removed_point_count: int
    preprocessing_rejected_points: np.ndarray
    rejected_points: np.ndarray
    first_layer_threshold_z: float
    original_point_count: int
    remaining_point_count: int
    runtime_seconds: float
    config: LayerByLayer2Config


@dataclass
class CandidateFilterResult:
    accepted_points: np.ndarray
    accepted_indices: np.ndarray
    normal: np.ndarray
    origin: np.ndarray
    component_size: int
    rejected_points: np.ndarray


def _validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected points with shape (N, 3).")
    if len(points) == 0:
        raise ValueError("At least one point is required.")
    return points


def load_mesh_vertices(mesh_path: str | Path, round_decimals: int | None = None) -> np.ndarray:
    mesh = trimesh.load(mesh_path, force="mesh")
    points = np.asarray(mesh.vertices, dtype=float)
    if round_decimals is not None:
        points = np.round(points, int(round_decimals))
    return points


def first_layer_by_target_count(
    points: np.ndarray,
    min_points: int = 3000,
    dz_init: float = 0.5,
    dz_step: float = 0.5,
    dz_max: float = 15.0,
) -> tuple[np.ndarray, float, int]:
    points = _validate_points(points)
    z0 = float(np.min(points[:, 2]))
    dz = float(max(dz_init, 1e-9))

    best_mask = points[:, 2] <= z0 + dz
    best_threshold = z0 + dz
    best_count = int(np.count_nonzero(best_mask))

    while dz <= dz_max + 1e-12:
        threshold = z0 + dz
        mask = points[:, 2] <= threshold
        count = int(np.count_nonzero(mask))
        if count >= int(min_points):
            return mask, threshold, count
        if count > best_count:
            best_mask = mask
            best_threshold = threshold
            best_count = count
        dz += dz_step

    return best_mask, best_threshold, best_count


def drop_isolated_indices(points: np.ndarray, radius: float = 1.5, min_other: int = 1) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty((0,), dtype=int)
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=float(radius), return_length=True)
    keep_mask = np.asarray(counts) >= (int(min_other) + 1)
    return np.flatnonzero(keep_mask)


def split_connected_components(points: np.ndarray, radius: float) -> list[np.ndarray]:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return []

    tree = cKDTree(points)
    neighbours = tree.query_ball_point(points, r=float(radius))
    visited = np.zeros(len(points), dtype=bool)
    components: list[np.ndarray] = []

    for start_idx in range(len(points)):
        if visited[start_idx]:
            continue

        stack = [start_idx]
        component = []
        visited[start_idx] = True

        while stack:
            idx = stack.pop()
            component.append(idx)
            for nbr in neighbours[idx]:
                nbr = int(nbr)
                if not visited[nbr]:
                    visited[nbr] = True
                    stack.append(nbr)

        components.append(np.asarray(component, dtype=int))

    return components


def preprocess_points(points: np.ndarray, config: LayerByLayer2Config) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    if len(points) == 0 or not config.preprocess_enabled:
        return points.copy(), np.empty((0, 3), dtype=float)

    working_points = points.copy()
    rejected_parts: list[np.ndarray] = []

    if config.preprocess_support_radius > 0.0:
        keep_local = drop_isolated_indices(
            working_points,
            radius=config.preprocess_support_radius,
            min_other=config.preprocess_support_min_other,
        )
        keep_mask = np.zeros(len(working_points), dtype=bool)
        keep_mask[keep_local] = True
        if np.any(~keep_mask):
            rejected_parts.append(working_points[~keep_mask])
        working_points = working_points[keep_mask]

    if (
        len(working_points) > 0
        and config.preprocess_component_radius > 0.0
        and config.preprocess_min_component_points > 1
    ):
        components = split_connected_components(
            working_points,
            radius=config.preprocess_component_radius,
        )
        keep_components = [
            np.asarray(component, dtype=int)
            for component in components
            if len(component) >= int(config.preprocess_min_component_points)
        ]
        if keep_components:
            keep_local = np.unique(np.concatenate(keep_components))
        elif components:
            keep_local = max(components, key=len)
        else:
            keep_local = np.empty((0,), dtype=int)

        keep_mask = np.zeros(len(working_points), dtype=bool)
        keep_mask[np.asarray(keep_local, dtype=int)] = True
        if np.any(~keep_mask):
            rejected_parts.append(working_points[~keep_mask])
        working_points = working_points[keep_mask]

    rejected_points = (
        np.vstack(rejected_parts)
        if rejected_parts
        else np.empty((0, 3), dtype=float)
    )
    return working_points, rejected_points


def best_fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError("Need at least three points to fit a plane.")
    origin = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - origin, full_matrices=False)
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("Could not determine a stable plane normal.")
    return origin, normal / norm


def plane_inlier_mask(
    points: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    signed = (np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)) @ np.asarray(
        normal,
        dtype=float,
    )
    return np.abs(signed) <= float(max(tolerance, 1e-9))


def refine_plane_inliers(
    points: np.ndarray,
    tolerance: float,
    refine_steps: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return np.arange(len(points), dtype=int), np.array([0.0, 0.0, 1.0]), np.mean(points, axis=0)

    indices = np.arange(len(points), dtype=int)
    working_indices = indices.copy()
    origin, normal = best_fit_plane(points)

    for _ in range(max(int(refine_steps), 1)):
        working_points = points[working_indices]
        if len(working_points) < 3:
            break
        origin, normal = best_fit_plane(working_points)
        local_mask = plane_inlier_mask(working_points, origin, normal, tolerance)
        next_indices = working_indices[local_mask]
        if len(next_indices) == len(working_indices):
            working_indices = next_indices
            break
        if len(next_indices) < 3:
            break
        working_indices = next_indices

    if len(working_indices) >= 3:
        origin, normal = best_fit_plane(points[working_indices])
        final_mask = plane_inlier_mask(points[working_indices], origin, normal, tolerance)
        working_indices = working_indices[final_mask]

    return working_indices.astype(int), origin, normal


def layer_centroid(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty((0,), dtype=float)
    return np.mean(points, axis=0)


def _component_sort_key(
    component_points: np.ndarray,
    reference_point: np.ndarray | None,
    mode: str,
) -> tuple:
    size = int(len(component_points))
    centroid = layer_centroid(component_points)
    distance = (
        float(np.linalg.norm(centroid - reference_point))
        if reference_point is not None and centroid.size == 3
        else np.inf
    )

    normalized_mode = str(mode or "largest_then_closest").strip().lower()
    if normalized_mode == "closest":
        return (distance, -size)
    if normalized_mode == "largest":
        return (-size, distance)
    return (-size, distance)


def select_primary_component(
    points: np.ndarray,
    components: list[np.ndarray],
    reference_point: np.ndarray | None,
    mode: str = "largest_then_closest",
) -> np.ndarray:
    if not components:
        return np.empty((0,), dtype=int)

    best_component = None
    best_key = None
    for component in components:
        component_points = points[component]
        key = _component_sort_key(component_points, reference_point, mode)
        if best_key is None or key < best_key:
            best_key = key
            best_component = component

    return np.asarray(best_component, dtype=int) if best_component is not None else np.empty((0,), dtype=int)


def select_component_by_seed_overlap(
    component_indices: list[np.ndarray],
    global_indices: np.ndarray,
    seed_indices: np.ndarray,
    points: np.ndarray,
    reference_point: np.ndarray | None,
) -> np.ndarray:
    if not component_indices:
        return np.empty((0,), dtype=int)

    seed_index_set = {int(idx) for idx in np.asarray(seed_indices, dtype=int)}
    best_component = None
    best_key = None

    for component in component_indices:
        component_global = global_indices[np.asarray(component, dtype=int)]
        overlap = sum(int(idx) in seed_index_set for idx in component_global)
        component_points = points[component_global]
        centroid = layer_centroid(component_points)
        distance = (
            float(np.linalg.norm(centroid - reference_point))
            if reference_point is not None and centroid.size == 3
            else np.inf
        )
        key = (-overlap, -len(component_global), distance)
        if best_key is None or key < best_key:
            best_key = key
            best_component = component_global

    return np.asarray(best_component, dtype=int) if best_component is not None else np.empty((0,), dtype=int)


def filter_candidate_layer(
    expansion_points: np.ndarray,
    seed_indices: np.ndarray,
    reference_centroid: np.ndarray | None,
    reference_normal: np.ndarray | None,
    config: LayerByLayer2Config,
) -> CandidateFilterResult:
    expansion_points = np.asarray(expansion_points, dtype=float)
    seed_indices = np.asarray(seed_indices, dtype=int)

    if len(expansion_points) == 0 or len(seed_indices) == 0:
        return CandidateFilterResult(
            accepted_points=np.empty((0, 3), dtype=float),
            accepted_indices=np.empty((0,), dtype=int),
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            origin=np.zeros(3, dtype=float),
            component_size=0,
            rejected_points=np.empty((0, 3), dtype=float),
        )

    seed_indices = np.unique(seed_indices[(seed_indices >= 0) & (seed_indices < len(expansion_points))])
    if len(seed_indices) == 0:
        return CandidateFilterResult(
            accepted_points=np.empty((0, 3), dtype=float),
            accepted_indices=np.empty((0,), dtype=int),
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            origin=np.zeros(3, dtype=float),
            component_size=0,
            rejected_points=np.empty((0, 3), dtype=float),
        )

    seed_points = expansion_points[seed_indices]
    kept_indices = drop_isolated_indices(
        seed_points,
        radius=config.isolation_radius,
        min_other=config.isolation_min_other,
    )
    if len(kept_indices) == 0:
        kept_indices = np.arange(len(seed_points), dtype=int)

    filtered_seed_points = seed_points[kept_indices]
    components = split_connected_components(filtered_seed_points, radius=config.component_radius)
    selected_component_local = select_primary_component(
        filtered_seed_points,
        components,
        reference_point=reference_centroid,
        mode=config.component_mode,
    )
    if len(selected_component_local) == 0:
        return CandidateFilterResult(
            accepted_points=np.empty((0, 3), dtype=float),
            accepted_indices=np.empty((0,), dtype=int),
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            origin=layer_centroid(filtered_seed_points),
            component_size=0,
            rejected_points=seed_points.copy(),
        )

    seed_component_points = filtered_seed_points[selected_component_local]
    seed_inlier_local, origin, normal = refine_plane_inliers(
        seed_component_points,
        tolerance=config.plane_inlier_tolerance,
        refine_steps=config.plane_refine_steps,
    )
    if reference_normal is not None and len(reference_normal) == 3:
        reference_normal = np.asarray(reference_normal, dtype=float)
        if float(np.dot(normal, reference_normal)) < 0.0:
            normal = -normal

    selected_seed_global = seed_indices[kept_indices[selected_component_local]]
    selected_seed_inlier_global = (
        selected_seed_global[seed_inlier_local]
        if len(seed_inlier_local) > 0
        else selected_seed_global
    )

    slab_global = np.flatnonzero(
        plane_inlier_mask(
            expansion_points,
            origin,
            normal,
            config.plane_inlier_tolerance,
        )
    )
    if len(slab_global) == 0:
        return CandidateFilterResult(
            accepted_points=np.empty((0, 3), dtype=float),
            accepted_indices=np.empty((0,), dtype=int),
            normal=np.asarray(normal, dtype=float),
            origin=np.asarray(origin, dtype=float),
            component_size=int(len(seed_component_points)),
            rejected_points=seed_points.copy(),
        )

    slab_points = expansion_points[slab_global]
    slab_kept_local = drop_isolated_indices(
        slab_points,
        radius=config.isolation_radius,
        min_other=config.isolation_min_other,
    )
    if len(slab_kept_local) == 0:
        slab_kept_local = np.arange(len(slab_global), dtype=int)
    slab_global = slab_global[slab_kept_local]
    slab_points = expansion_points[slab_global]

    if config.keep_all_coplanar_components:
        accepted_global = slab_global
    else:
        slab_components = split_connected_components(slab_points, radius=config.component_radius)
        accepted_global = select_component_by_seed_overlap(
            slab_components,
            slab_global,
            selected_seed_inlier_global,
            expansion_points,
            reference_centroid,
        )
        if len(accepted_global) == 0:
            accepted_global = slab_global

    accepted_global = np.unique(np.asarray(accepted_global, dtype=int))
    accepted_points = expansion_points[accepted_global]
    seed_accepted_mask = np.isin(seed_indices, accepted_global, assume_unique=False)
    rejected_points = seed_points[~seed_accepted_mask]

    return CandidateFilterResult(
        accepted_points=np.asarray(accepted_points, dtype=float),
        accepted_indices=np.asarray(accepted_global, dtype=int),
        normal=np.asarray(normal, dtype=float),
        origin=np.asarray(origin, dtype=float),
        component_size=int(len(seed_component_points)),
        rejected_points=np.asarray(rejected_points, dtype=float),
    )


def _candidate_mask_from_previous_layer(previous_layer: np.ndarray, remaining: np.ndarray, radius: float) -> np.ndarray:
    if len(previous_layer) == 0 or len(remaining) == 0:
        return np.zeros(len(remaining), dtype=bool)
    tree = cKDTree(previous_layer)
    hits = tree.query_ball_point(remaining, r=float(radius))
    return np.fromiter((len(item) > 0 for item in hits), dtype=bool, count=len(remaining))


def extract_layers_from_points(
    points: np.ndarray,
    config: LayerByLayer2Config | None = None,
) -> LayerByLayer2Result:
    start_time = perf_counter()
    config = LayerByLayer2Config() if config is None else config
    points = _validate_points(points)
    original_point_count = int(len(points))

    if config.round_decimals is not None:
        points = np.round(points, int(config.round_decimals))

    preprocessed_points, preprocessing_rejected_points = preprocess_points(points, config)
    remaining = preprocessed_points.copy()
    rejected_buckets: list[np.ndarray] = []

    first_mask, threshold_z, _ = first_layer_by_target_count(
        remaining,
        min_points=config.first_layer_min_points,
        dz_init=config.first_layer_dz_init,
        dz_step=config.first_layer_dz_step,
        dz_max=config.first_layer_dz_max,
    )

    first_seed_indices = np.flatnonzero(first_mask)
    first_filtered = filter_candidate_layer(
        remaining,
        seed_indices=first_seed_indices,
        reference_centroid=None,
        reference_normal=None,
        config=config,
    )
    if len(first_filtered.accepted_indices) > 0:
        keep_mask = np.ones(len(remaining), dtype=bool)
        keep_mask[first_filtered.accepted_indices] = False
        remaining = remaining[keep_mask]
    else:
        remaining = remaining[~first_mask]
        if np.any(first_mask):
            rejected_buckets.append(points[first_mask])

    layers: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    search_radii: list[float | None] = []
    component_sizes: list[int] = []

    if len(first_filtered.accepted_points) >= int(config.min_layer_points):
        layers.append(first_filtered.accepted_points)
        centroids.append(layer_centroid(first_filtered.accepted_points))
        normals.append(first_filtered.normal)
        search_radii.append(None)
        component_sizes.append(first_filtered.component_size)

    layer_idx = len(layers)
    while layer_idx < int(config.max_layers) and len(remaining) > 0 and layers:
        previous_layer = layers[-1]
        previous_centroid = centroids[-1]
        previous_normal = normals[-1]

        chosen_mask = None
        chosen_radius = None
        radius = float(config.adjacency_radius)
        while radius <= float(config.adjacency_radius_max) + 1e-12:
            candidate_mask = _candidate_mask_from_previous_layer(previous_layer, remaining, radius)
            if np.any(candidate_mask):
                chosen_mask = candidate_mask
                chosen_radius = radius
                break
            radius += float(config.adjacency_radius_step)

        if chosen_mask is None:
            break

        filtered = filter_candidate_layer(
            remaining,
            seed_indices=np.flatnonzero(chosen_mask),
            reference_centroid=previous_centroid,
            reference_normal=previous_normal,
            config=config,
        )

        if len(filtered.accepted_points) < int(config.min_layer_points):
            rejected_buckets.append(remaining[chosen_mask])
            remaining = remaining[~chosen_mask]
            continue

        keep_mask = np.ones(len(remaining), dtype=bool)
        keep_mask[filtered.accepted_indices] = False
        remaining = remaining[keep_mask]
        layers.append(filtered.accepted_points)
        centroids.append(layer_centroid(filtered.accepted_points))
        normals.append(filtered.normal)
        search_radii.append(chosen_radius)
        component_sizes.append(filtered.component_size)
        layer_idx += 1

    rejected_points = (
        np.vstack(rejected_buckets)
        if rejected_buckets
        else np.empty((0, 3), dtype=float)
    )

    return LayerByLayer2Result(
        layers=layers,
        centroids=centroids,
        normals=normals,
        search_radii=search_radii,
        component_sizes=component_sizes,
        preprocessed_point_count=int(len(preprocessed_points)),
        preprocessing_removed_point_count=int(len(preprocessing_rejected_points)),
        preprocessing_rejected_points=preprocessing_rejected_points,
        rejected_points=rejected_points,
        first_layer_threshold_z=float(threshold_z),
        original_point_count=original_point_count,
        remaining_point_count=int(len(remaining)),
        runtime_seconds=float(perf_counter() - start_time),
        config=config,
    )


def extract_layers_from_mesh(
    mesh_path: str | Path,
    config: LayerByLayer2Config | None = None,
) -> LayerByLayer2Result:
    config = LayerByLayer2Config() if config is None else config
    points = load_mesh_vertices(mesh_path, round_decimals=config.round_decimals)
    return extract_layers_from_points(points, config=config)


def _point_record(point: Iterable[float]) -> dict[str, float]:
    x, y, z = point
    return {"x": float(x), "y": float(y), "z": float(z)}


def layers_to_json_payload(result: LayerByLayer2Result) -> dict:
    layers_payload = []
    for layer_idx, layer_points in enumerate(result.layers):
        centroid = result.centroids[layer_idx]
        normal = result.normals[layer_idx]
        layers_payload.append(
            {
                "layer": int(layer_idx),
                "point_count": int(len(layer_points)),
                "search_radius": (
                    None
                    if result.search_radii[layer_idx] is None
                    else float(result.search_radii[layer_idx])
                ),
                "component_size_before_plane_filter": int(result.component_sizes[layer_idx]),
                "centroid": _point_record(centroid),
                "normal": _point_record(normal),
                "points": [_point_record(point) for point in layer_points],
            }
        )

    return {
        "method": "layer_by_layer_component_plane_filter",
        "original_point_count": int(result.original_point_count),
        "preprocessed_point_count": int(result.preprocessed_point_count),
        "preprocessing_removed_point_count": int(result.preprocessing_removed_point_count),
        "remaining_point_count": int(result.remaining_point_count),
        "runtime_seconds": float(result.runtime_seconds),
        "rejected_point_count": int(len(result.rejected_points)),
        "first_layer_threshold_z": float(result.first_layer_threshold_z),
        "layer_count": int(len(result.layers)),
        "config": asdict(result.config),
        "layers": layers_payload,
        "preprocessing_rejected_points": [
            _point_record(point) for point in result.preprocessing_rejected_points
        ],
        "rejected_points": [_point_record(point) for point in result.rejected_points],
    }


def save_layers_json(path: str | Path, result: LayerByLayer2Result) -> None:
    payload = layers_to_json_payload(result)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_layers_csv(path: str | Path, result: LayerByLayer2Result) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "x", "y", "z"])
        for layer_idx, layer_points in enumerate(result.layers):
            for point in layer_points:
                writer.writerow([layer_idx, float(point[0]), float(point[1]), float(point[2])])


def set_axes_equal(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_layers(
    result: LayerByLayer2Result,
    elev: float = 0.0,
    azim: float = 270.0,
    point_size: float = 2.0,
    show_centroids: bool = True,
    show_rejected: bool = True,
    show_legend: bool = False,
    legend_stride: int = 10,
    legend_layer_color: str = "red",
    base_layer_color: str = "#6f8fa6",
    title: str = "Layer-by-layer extraction 2",
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)

    if show_rejected and len(result.rejected_points) > 0:
        ax.scatter(
            result.rejected_points[:, 0],
            result.rejected_points[:, 1],
            result.rejected_points[:, 2],
            s=max(point_size * 0.8, 1.0),
            color="lightgray",
            alpha=0.5,
            label="Rejected",
        )

    centroid_points = []

    for layer_idx, layer_points in enumerate(result.layers):
        if len(layer_points) == 0:
            continue
        highlight_layer = int(max(1, legend_stride)) > 0 and (layer_idx % int(max(1, legend_stride)) == 0)
        layer_color = legend_layer_color if highlight_layer else base_layer_color
        layer_label = f"Layer {layer_idx}" if highlight_layer else "_nolegend_"
        ax.scatter(
            layer_points[:, 0],
            layer_points[:, 1],
            layer_points[:, 2],
            s=point_size,
            color=layer_color,
            label=layer_label,
        )
        if show_centroids:
            centroid_points.append(result.centroids[layer_idx])

    if centroid_points:
        centroid_array = np.asarray(centroid_points, dtype=float)
        ax.plot(
            centroid_array[:, 0],
            centroid_array[:, 1],
            centroid_array[:, 2],
            color="black",
            linewidth=1.5,
            marker="o",
            markersize=3,
            label="Centroids",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    set_axes_equal(ax)
    if show_legend:
        ax.legend(loc="best", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.show()
