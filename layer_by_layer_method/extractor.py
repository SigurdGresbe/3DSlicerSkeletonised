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
class LayerByLayerConfig:
    first_layer_min_points: int = 1000
    first_layer_dz_init: float = 0.5
    first_layer_dz_step: float = 0.5
    first_layer_dz_max: float = 2.0
    adjacency_radius: float = 4.0
    adjacency_radius_max: float = 10.0
    adjacency_radius_step: float = 0.2
    isolation_radius: float = 1.5
    isolation_min_other: int = 1
    min_layer_points: int = 8
    max_layers: int = 1000
    round_decimals: int | None = None


@dataclass
class LayerExtractionResult:
    layers: list[np.ndarray]
    centroids: list[np.ndarray]
    search_radii: list[float | None]
    first_layer_threshold_z: float
    original_point_count: int
    remaining_point_count: int
    runtime_seconds: float
    config: LayerByLayerConfig


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


def drop_isolated(points: np.ndarray, radius: float = 1.5, min_other: int = 1) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=float(radius), return_length=True)
    keep_mask = np.asarray(counts) >= (int(min_other) + 1)
    return points[keep_mask]


def layer_centroid(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty((0,), dtype=float)
    return np.mean(points, axis=0)


def _candidate_mask_from_previous_layer(
    previous_layer: np.ndarray,
    remaining: np.ndarray,
    radius: float,
) -> np.ndarray:
    if len(previous_layer) == 0 or len(remaining) == 0:
        return np.zeros(len(remaining), dtype=bool)
    tree = cKDTree(previous_layer)
    hits = tree.query_ball_point(remaining, r=float(radius))
    return np.fromiter((len(item) > 0 for item in hits), dtype=bool, count=len(remaining))


def extract_layers_from_points(
    points: np.ndarray,
    config: LayerByLayerConfig | None = None,
) -> LayerExtractionResult:
    start_time = perf_counter()
    config = LayerByLayerConfig() if config is None else config
    points = _validate_points(points)

    if config.round_decimals is not None:
        points = np.round(points, int(config.round_decimals))

    remaining = points.copy()
    first_mask, threshold_z, _ = first_layer_by_target_count(
        remaining,
        min_points=config.first_layer_min_points,
        dz_init=config.first_layer_dz_init,
        dz_step=config.first_layer_dz_step,
        dz_max=config.first_layer_dz_max,
    )

    first_layer_raw = remaining[first_mask]
    first_layer = drop_isolated(
        first_layer_raw,
        radius=config.isolation_radius,
        min_other=config.isolation_min_other,
    )
    if len(first_layer) == 0:
        first_layer = first_layer_raw

    layers: list[np.ndarray] = [np.asarray(first_layer, dtype=float)]
    centroids: list[np.ndarray] = [layer_centroid(first_layer)]
    search_radii: list[float | None] = [None]

    remaining = remaining[~first_mask]

    layer_idx = 1
    while layer_idx < int(config.max_layers) and len(remaining) > 0:
        previous_layer = layers[-1]
        if len(previous_layer) == 0:
            break

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

        candidate_layer = remaining[chosen_mask]
        filtered_layer = drop_isolated(
            candidate_layer,
            radius=config.isolation_radius,
            min_other=config.isolation_min_other,
        )

        # Remove consumed candidate points even when they are rejected, mirroring
        # the legacy scripts that delete isolated/noisy candidates from the pool.
        remaining = remaining[~chosen_mask]

        if len(filtered_layer) < int(config.min_layer_points):
            continue

        layers.append(np.asarray(filtered_layer, dtype=float))
        centroids.append(layer_centroid(filtered_layer))
        search_radii.append(chosen_radius)
        layer_idx += 1

    return LayerExtractionResult(
        layers=layers,
        centroids=centroids,
        search_radii=search_radii,
        first_layer_threshold_z=float(threshold_z),
        original_point_count=int(len(points)),
        remaining_point_count=int(len(remaining)),
        runtime_seconds=float(perf_counter() - start_time),
        config=config,
    )


def extract_layers_from_mesh(
    mesh_path: str | Path,
    config: LayerByLayerConfig | None = None,
) -> LayerExtractionResult:
    config = LayerByLayerConfig() if config is None else config
    points = load_mesh_vertices(mesh_path, round_decimals=config.round_decimals)
    return extract_layers_from_points(points, config=config)


def _point_record(point: Iterable[float]) -> dict[str, float]:
    x, y, z = point
    return {"x": float(x), "y": float(y), "z": float(z)}


def layers_to_json_payload(result: LayerExtractionResult) -> dict:
    layers_payload = []
    for layer_idx, layer_points in enumerate(result.layers):
        centroid = result.centroids[layer_idx]
        search_radius = result.search_radii[layer_idx]
        centroid_payload = (
            _point_record(centroid)
            if centroid.size == 3
            else None
        )
        layers_payload.append(
            {
                "layer": int(layer_idx),
                "point_count": int(len(layer_points)),
                "search_radius": None if search_radius is None else float(search_radius),
                "centroid": centroid_payload,
                "points": [_point_record(point) for point in layer_points],
            }
        )

    return {
        "method": "layer_by_layer_proximity",
        "original_point_count": int(result.original_point_count),
        "remaining_point_count": int(result.remaining_point_count),
        "runtime_seconds": float(result.runtime_seconds),
        "first_layer_threshold_z": float(result.first_layer_threshold_z),
        "layer_count": int(len(result.layers)),
        "config": asdict(result.config),
        "layers": layers_payload,
    }


def save_layers_json(path: str | Path, result: LayerExtractionResult) -> None:
    payload = layers_to_json_payload(result)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_layers_csv(path: str | Path, result: LayerExtractionResult) -> None:
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
    result: LayerExtractionResult,
    elev: float = 0.0,
    azim: float = 270.0,
    point_size: float = 2.0,
    show_centroids: bool = True,
    title: str = "Layer-by-layer extraction",
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)

    colors = plt.cm.viridis(np.linspace(0.0, 1.0, max(1, len(result.layers))))
    centroid_points = []

    for layer_idx, layer_points in enumerate(result.layers):
        if len(layer_points) == 0:
            continue
        ax.scatter(
            layer_points[:, 0],
            layer_points[:, 1],
            layer_points[:, 2],
            s=point_size,
            color=colors[layer_idx],
            label=f"Layer {layer_idx}",
        )
        centroid = result.centroids[layer_idx]
        if show_centroids and centroid.size == 3:
            centroid_points.append(centroid)

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
    ax.legend(loc="best", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.show()
