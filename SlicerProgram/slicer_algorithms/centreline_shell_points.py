import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
try:
    from scipy.interpolate import make_splprep
except ImportError:
    from scipy.interpolate import splprep, splev

    def make_splprep(*args, **kwargs):
        tck, u = splprep(*args, **kwargs)

        class _SplineWrapper:
            def __call__(self, values):
                return np.asarray(splev(values, tck))

        return _SplineWrapper(), u
from scipy.spatial import cKDTree

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH_DIR = PACKAGE_ROOT / "mesh"

from .HitBoxMethod import (
    build_knn_graph,
    build_skeleton_tree,
    build_sphere_graph,
    build_sphere_skeleton_graph,
    extract_tree_paths,
    generate_component_centroid_spheres,
    generate_spheres,
    load_mesh,
)

logger = logging.getLogger(__name__)

"""
Optional models are: 
- STLfiles/CustomEdgeCaseThesis.stl: A simple bent column.
- STLfiles/EdgeCaseSplit.stl: A simple split case with a single branch point and two end branches.
- STLfiles/EdgeCaseSplitJoin.stl: A case with a split followed by a join.
"""

CONFIG = {
    "mesh": str(DEFAULT_MESH_DIR / "EdgeCaseSplit.stl"),
    "line_method": "auto",
    "sphere_generation_method": "auto",
    "plane_spacing": 2.0,
    "adaptive_plane_spacing": True,
    "adaptive_spacing_min_factor": 0.6,
    "outer_corner_spacing_safety": 0.35,
    "shell_point_spacing": 1.0,
    "start_z": 0.0,
    "start_z_tolerance": 5.0,
    "knn_k": 4,
    "sphere_min_diameter": 0.0,
    "sphere_max_diameter": 0.0,
    "overlap_factor": 1,
    "spline_s": 2.0,
    "centreline_samples": 200,
    "centreline_extension_length": 0.0,
    "tree_sphere_graph_k": 4,
    "graph_strategy": "mst",
    "nurbs_surface_count": 16,
    "nurbs_angle_tolerance_deg": 20.0,
    "nurbs_min_points": 3,
    "csv": "",
    "json": "",
    "no_plot": False,
}


def compute_hitbox_centreline(
    mesh,
    knn_k=4,
    overlap_factor=0.7,
    sphere_min_diameter=0.0,
    sphere_max_diameter=0.0,
    sphere_generation_method="auto",
    spline_s=2.0,
    centreline_samples=200,
    sphere_graph_k=2,
    graph_strategy="mst",
):
    """
    Recreate the single-path centreline used in HitBoxMethod.plot_centrelines.
    """
    spheres = compute_hitbox_spheres(
        mesh,
        knn_k=knn_k,
        overlap_factor=overlap_factor,
        min_diameter=sphere_min_diameter,
        max_diameter=sphere_max_diameter,
        sphere_generation_method=sphere_generation_method,
    )

    centreline = fit_single_sphere_centreline(
        spheres,
        sphere_graph_k=sphere_graph_k,
        graph_strategy=graph_strategy,
        spline_s=spline_s,
        centreline_samples=centreline_samples,
    )
    return extend_centreline_to_mesh_ends(centreline, mesh)


def normalize_sphere_generation_method(method):
    method = str(method or "auto").strip().lower().replace("-", "_")
    aliases = {
        "path": "skeleton_paths",
        "paths": "skeleton_paths",
        "skeleton": "skeleton_paths",
        "skeleton_path": "skeleton_paths",
        "skeleton_paths": "skeleton_paths",
        "component": "component_centroid",
        "components": "component_centroid",
        "component_centroid": "component_centroid",
        "component_centroids": "component_centroid",
        "connected_components": "component_centroid",
        "fallback": "component_centroid",
        "auto": "auto",
    }
    return aliases.get(method, "auto")


def compute_hitbox_spheres(
    mesh,
    knn_k=4,
    overlap_factor=1,
    min_diameter=0.0,
    max_diameter=0.0,
    sphere_generation_method="auto",
    return_method=False,
    cancel_callback=None,
):
    points = mesh.vertices
    graph = build_knn_graph(points, k=knn_k, cancel_callback=cancel_callback)

    requested_method = normalize_sphere_generation_method(sphere_generation_method)
    effective_method = requested_method

    if requested_method == "component_centroid":
        spheres = generate_component_centroid_spheres(
            points,
            graph,
            overlap_factor=overlap_factor,
            min_diameter=min_diameter,
            max_diameter=max_diameter,
            cancel_callback=cancel_callback,
        )
    else:
        spheres = generate_spheres(
            points,
            graph,
            mesh.bounds,
            overlap_factor=overlap_factor,
            min_diameter=min_diameter,
            max_diameter=max_diameter,
            cancel_callback=cancel_callback,
        )
        effective_method = "skeleton_paths"
        if not spheres and requested_method == "auto":
            print("No spheres from skeleton paths, using component-centroid spheres")
            spheres = generate_component_centroid_spheres(
                points,
                graph,
                overlap_factor=overlap_factor,
                min_diameter=min_diameter,
                max_diameter=max_diameter,
                cancel_callback=cancel_callback,
            )
            effective_method = "component_centroid"

    if len(spheres) < 2:
        raise RuntimeError("Need at least 2 spheres to build a centreline.")
    if return_method:
        return spheres, effective_method
    return spheres


def fit_branch_centreline(
    points,
    spline_s=2.0,
    centreline_samples=200,
    preserve_endpoints=False,
):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return None
    if len(points) < 4:
        return points.copy()

    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    if u[-1] <= 0:
        return points.copy()
    u = u / u[-1]

    effective_s = 0.0 if preserve_endpoints else spline_s
    spline, _ = make_splprep(points.T, u=u, s=effective_s)
    u_new = np.linspace(0.0, 1.0, centreline_samples)
    centreline = spline(u_new).T

    if preserve_endpoints:
        centreline[0] = points[0]
        centreline[-1] = points[-1]

    return centreline


def ordered_single_sphere_centres(
    spheres,
    sphere_graph_k=2,
    graph_strategy="mst",
):
    """
    Return sphere centres ordered along the longest path of the sphere graph.

    The raw sphere list is generation order, not centreline order. Using the
    graph path keeps the fitted single centreline attached to the sphere chain.
    """
    tree, centres = prepare_sphere_tree(
        spheres,
        sphere_graph_k=sphere_graph_k,
        graph_strategy=graph_strategy,
    )

    if tree.number_of_nodes() == 0:
        raise RuntimeError("Sphere graph is empty.")

    if tree.number_of_nodes() == 1:
        return centres.copy()

    tree = prune_short_terminal_branches(tree, centres)

    endpoints = [node for node in tree.nodes if tree.degree[node] <= 1]
    if len(endpoints) < 2:
        endpoints = list(tree.nodes)

    best_path = None
    best_length = -np.inf

    for start_pos, start in enumerate(endpoints):
        lengths = nx.single_source_dijkstra_path_length(tree, start, weight="weight")
        for end in endpoints[start_pos + 1:]:
            length = lengths.get(end)
            if length is None or length <= best_length:
                continue
            best_length = length
            best_path = nx.shortest_path(tree, start, end, weight="weight")

    if best_path is None:
        best_path = list(nx.dfs_preorder_nodes(tree, source=endpoints[0]))

    ordered = centres[np.asarray(best_path, dtype=int)]
    if ordered[0, 2] > ordered[-1, 2]:
        ordered = ordered[::-1]
    return ordered


def fit_single_sphere_centreline(
    spheres,
    sphere_graph_k=2,
    graph_strategy="mst",
    spline_s=2.0,
    centreline_samples=200,
):
    centres = ordered_single_sphere_centres(
        spheres,
        sphere_graph_k=sphere_graph_k,
        graph_strategy=graph_strategy,
    )
    centreline = fit_branch_centreline(
        centres,
        spline_s=spline_s,
        centreline_samples=centreline_samples,
        preserve_endpoints=True,
    )
    if centreline is None or len(centreline) < 2:
        raise RuntimeError("No valid single centreline could be fitted.")
    return centreline


def _extend_endpoint_to_mesh(
    endpoint,
    inward_neighbor,
    mesh_vertices,
    radius_factor=2.5,
):
    endpoint = np.asarray(endpoint, dtype=float)
    inward_neighbor = np.asarray(inward_neighbor, dtype=float)
    direction = endpoint - inward_neighbor
    direction_norm = np.linalg.norm(direction)
    if direction_norm <= 1e-12:
        return endpoint
    direction = direction / direction_norm

    rel = mesh_vertices - endpoint
    axial = rel @ direction
    lateral_vectors = rel - axial[:, None] * direction
    lateral = np.linalg.norm(lateral_vectors, axis=1)

    local_radius = float(np.percentile(lateral[np.argsort(np.linalg.norm(rel, axis=1))[:50]], 50))
    if not np.isfinite(local_radius) or local_radius <= 1e-9:
        local_radius = float(np.percentile(lateral, 10))
    if not np.isfinite(local_radius) or local_radius <= 1e-9:
        return endpoint

    for scale in (radius_factor, radius_factor * 1.5, radius_factor * 2.0):
        candidate_mask = (axial > 0.0) & (lateral <= local_radius * scale)
        if not np.any(candidate_mask):
            continue
        extension = float(np.percentile(axial[candidate_mask], 98))
        if extension > direction_norm * 0.1:
            return endpoint + direction * extension

    return endpoint


def extend_centreline_to_mesh_ends(centreline, mesh):
    """
    Extend a single centreline from its sphere endpoints to nearby mesh ends.

    Sphere centres often stop before the physical cap/opening because they are
    generated from local neighborhoods. This projects each endpoint along its
    local tangent and uses nearby mesh vertices to estimate the missing end.
    """
    centreline = np.asarray(centreline, dtype=float)
    if len(centreline) < 2:
        return centreline

    mesh_vertices = np.asarray(mesh.vertices, dtype=float)
    if len(mesh_vertices) == 0:
        return centreline

    start = _extend_endpoint_to_mesh(
        centreline[0],
        centreline[1],
        mesh_vertices,
    )
    end = _extend_endpoint_to_mesh(
        centreline[-1],
        centreline[-2],
        mesh_vertices,
    )

    extended = centreline.copy()
    if np.linalg.norm(start - extended[0]) > 1e-9:
        extended = np.vstack([start, extended])
    if np.linalg.norm(end - extended[-1]) > 1e-9:
        extended = np.vstack([extended, end])
    return extended


def extend_centreline_by_length(
    centreline,
    extension_length=0.0,
    extend_start=True,
    extend_end=True,
):
    """
    Extend a polyline along its local endpoint tangents by a fixed length.
    """
    centreline = np.asarray(centreline, dtype=float)
    extension_length = float(max(extension_length, 0.0))
    if len(centreline) < 2 or extension_length <= 0.0:
        return centreline.copy()

    extended = centreline.copy()
    if extend_start:
        start_direction = extended[0] - extended[1]
        start_norm = np.linalg.norm(start_direction)
        if start_norm > 1e-12:
            start = extended[0] + (start_direction / start_norm) * extension_length
            extended = np.vstack([start, extended])

    if extend_end:
        end_direction = extended[-1] - extended[-2]
        end_norm = np.linalg.norm(end_direction)
        if end_norm > 1e-12:
            end = extended[-1] + (end_direction / end_norm) * extension_length
            extended = np.vstack([extended, end])

    return extended


def extend_branch_centrelines_by_length(branch_centrelines, extension_length=0.0):
    """
    Extend only free branch endpoints, leaving shared junction endpoints fixed.
    """
    extension_length = float(max(extension_length, 0.0))
    if extension_length <= 0.0:
        return [
            (branch_id, np.asarray(centreline, dtype=float).copy())
            for branch_id, centreline in branch_centrelines
        ]

    endpoint_records = []
    for branch_id, centreline in branch_centrelines:
        centreline = np.asarray(centreline, dtype=float)
        if len(centreline) < 2:
            continue
        endpoint_records.append((branch_id, "start", centreline[0]))
        endpoint_records.append((branch_id, "end", centreline[-1]))

    def is_shared_endpoint(point):
        matches = 0
        for _, _, other_point in endpoint_records:
            if np.linalg.norm(point - other_point) <= 1e-6:
                matches += 1
        return matches > 1

    extended_branches = []
    for branch_id, centreline in branch_centrelines:
        centreline = np.asarray(centreline, dtype=float)
        if len(centreline) < 2:
            extended_branches.append((branch_id, centreline.copy()))
            continue
        extended_branches.append(
            (
                branch_id,
                extend_centreline_by_length(
                    centreline,
                    extension_length=extension_length,
                    extend_start=not is_shared_endpoint(centreline[0]),
                    extend_end=not is_shared_endpoint(centreline[-1]),
                ),
            )
        )

    return extended_branches


def prepare_sphere_tree(spheres, sphere_graph_k=2, graph_strategy="mst"):
    tree, centres = build_sphere_skeleton_graph(
        spheres,
        strategy=graph_strategy,
        k=sphere_graph_k,
    )

    if tree.number_of_nodes() == 0:
        return tree, centres

    for u, v in tree.edges:
        tree[u][v]["weight"] = float(np.linalg.norm(centres[u] - centres[v]))

    while tree.number_of_nodes() > 1 and not nx.is_connected(tree):
        components = [list(comp) for comp in nx.connected_components(tree)]
        best_edge = None
        best_weight = np.inf

        for comp_idx, comp_a in enumerate(components):
            pts_a = centres[np.asarray(comp_a, dtype=int)]
            for comp_b in components[comp_idx + 1:]:
                pts_b = centres[np.asarray(comp_b, dtype=int)]
                deltas = pts_a[:, None, :] - pts_b[None, :, :]
                distances = np.linalg.norm(deltas, axis=2)
                local_a, local_b = np.unravel_index(
                    int(np.argmin(distances)),
                    distances.shape,
                )
                weight = float(distances[local_a, local_b])
                if weight < best_weight:
                    best_weight = weight
                    best_edge = (comp_a[local_a], comp_b[local_b])

        if best_edge is None:
            break
        tree.add_edge(best_edge[0], best_edge[1], weight=best_weight)

    if not nx.is_tree(tree):
        tree = nx.minimum_spanning_tree(tree, weight="weight")

    return tree, centres


def prune_short_terminal_branches(tree, centres, min_length_ratio=0.08):
    """
    Remove short leaf spurs from a sphere tree.

    Small terminal branches are usually caused by local sphere-graph noise near
    tight bends or ends. They should not split a visually single branch model.
    """
    pruned = tree.copy()
    total_length = sum(
        data.get(
            "weight",
            float(np.linalg.norm(centres[u] - centres[v])),
        )
        for u, v, data in pruned.edges(data=True)
    )
    min_length = float(max(min_length_ratio, 0.0)) * max(total_length, 1e-12)

    changed = True
    while changed and pruned.number_of_nodes() > 2:
        changed = False
        leaves = [node for node in pruned.nodes if pruned.degree[node] == 1]

        for leaf in leaves:
            if leaf not in pruned or pruned.degree[leaf] != 1:
                continue

            path = [leaf]
            prev = None
            curr = leaf
            length = 0.0

            while True:
                neighbours = [n for n in pruned.neighbors(curr) if n != prev]
                if not neighbours:
                    break
                nxt = neighbours[0]
                length += float(
                    pruned[curr][nxt].get(
                        "weight",
                        np.linalg.norm(centres[curr] - centres[nxt]),
                    )
                )
                prev, curr = curr, nxt

                if pruned.degree[curr] != 2:
                    break
                path.append(curr)

            if pruned.degree[curr] <= 1:
                continue
            if length >= min_length:
                continue

            removable = [node for node in path if pruned.degree[node] <= 2]
            if removable and pruned.number_of_nodes() - len(removable) >= 2:
                pruned.remove_nodes_from(removable)
                changed = True

    return pruned


def build_tree_branch_centrelines(
    spheres,
    sphere_graph_k=2,
    graph_strategy="mst",
    spline_s=2.0,
    centreline_samples=200,
):
    """
    Create rooted spline segments from the tree-based sphere representation.

    The traversal starts from the lowest-z endpoint when possible so the trunk
    is represented by one spline until the first true junction. Each junction
    then spawns additional spline segments for its outgoing branches.
    """
    tree, centres = prepare_sphere_tree(
        spheres,
        sphere_graph_k=sphere_graph_k,
        graph_strategy=graph_strategy,
    )

    if tree.number_of_nodes() == 0:
        raise RuntimeError("Tree-based sphere graph is empty.")

    tree = prune_short_terminal_branches(tree, centres)

    endpoints = [node for node in tree.nodes if tree.degree[node] == 1]
    if endpoints:
        root = min(endpoints, key=lambda idx: centres[idx][2])
    else:
        root = min(tree.nodes, key=lambda idx: centres[idx][2])

    branch_paths = []
    visited_edges = set()

    def walk_segment(start_node, next_node):
        edge_key = tuple(sorted((start_node, next_node)))
        if edge_key in visited_edges:
            return

        path = [start_node, next_node]
        prev = start_node
        curr = next_node
        visited_edges.add(edge_key)

        while True:
            forward = [n for n in tree.neighbors(curr) if n != prev]
            if tree.degree[curr] != 2 or len(forward) != 1:
                break

            nxt = forward[0]
            next_edge = tuple(sorted((curr, nxt)))
            if next_edge in visited_edges:
                break
            path.append(nxt)
            visited_edges.add(next_edge)
            prev, curr = curr, nxt

        branch_paths.append(path)

        for child in [n for n in tree.neighbors(curr) if n != prev]:
            child_edge = tuple(sorted((curr, child)))
            if child_edge in visited_edges:
                continue
            walk_segment(curr, child)

    root_children = list(tree.neighbors(root))
    if not root_children:
        raise RuntimeError("Root node has no neighbours in the tree.")

    def trunk_score(child):
        prev = root
        curr = child
        length = np.linalg.norm(centres[curr] - centres[prev])

        while True:
            forward = [n for n in tree.neighbors(curr) if n != prev]
            if tree.degree[curr] != 2 or len(forward) != 1:
                break
            nxt = forward[0]
            length += np.linalg.norm(centres[nxt] - centres[curr])
            prev, curr = curr, nxt

        return length

    root_children = sorted(root_children, key=trunk_score, reverse=True)
    for child in root_children:
        walk_segment(root, child)

    branch_centrelines = []
    for branch_id, path in enumerate(branch_paths):
        branch_points = centres[np.asarray(path, dtype=int)]
        centreline = fit_branch_centreline(
            branch_points,
            spline_s=spline_s,
            centreline_samples=centreline_samples,
            preserve_endpoints=True,
        )
        if centreline is None or len(centreline) < 2:
            continue
        branch_centrelines.append((branch_id, centreline))

    if not branch_centrelines:
        raise RuntimeError("No valid tree-branch centrelines could be built.")

    return branch_centrelines


def detect_branching_from_spheres(spheres, sphere_graph_k=2, graph_strategy="mst"):
    """
    Detect whether the sphere adjacency graph contains branching.
    """
    tree, centres = prepare_sphere_tree(
        spheres,
        sphere_graph_k=sphere_graph_k,
        graph_strategy=graph_strategy,
    )
    tree = prune_short_terminal_branches(tree, centres)
    return any(tree.degree[node] > 2 for node in tree.nodes)


def tangents_from_centreline(centreline):
    """
    Compute unit tangents along the centreline.
    """
    if len(centreline) < 2:
        raise RuntimeError("Need at least two centreline points to define planes.")

    tangents = np.zeros_like(centreline)
    tangents[1:-1] = centreline[2:] - centreline[:-2]
    tangents[0] = centreline[1] - centreline[0]
    tangents[-1] = centreline[-1] - centreline[-2]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return tangents / norms


def tangent_for_single_sample(sample_point, reference_centreline):
    """
    Compute a fallback tangent for a single sampled plane origin by borrowing
    the local direction from the full reference centreline.
    """
    reference_centreline = np.asarray(reference_centreline, dtype=float)
    if len(reference_centreline) < 2:
        raise RuntimeError("Need at least two centreline points to define planes.")

    if len(reference_centreline) == 2:
        direction = reference_centreline[1] - reference_centreline[0]
    else:
        nearest_idx = int(
            np.argmin(np.linalg.norm(reference_centreline - sample_point, axis=1))
        )
        if nearest_idx == 0:
            direction = reference_centreline[1] - reference_centreline[0]
        elif nearest_idx == len(reference_centreline) - 1:
            direction = reference_centreline[-1] - reference_centreline[-2]
        else:
            direction = (
                reference_centreline[nearest_idx + 1]
                - reference_centreline[nearest_idx - 1]
            )

    norm = np.linalg.norm(direction)
    if np.isclose(norm, 0.0):
        raise RuntimeError("Could not determine a valid tangent for the plane.")
    return direction / norm


def plane_normals_from_samples(plane_origins, reference_centreline):
    """
    Compute plane normals from sampled plane origins. Supports the single-plane
    case by using the local tangent from the reference centreline.
    """
    if len(plane_origins) == 0:
        raise RuntimeError("No plane origins were generated.")
    if len(plane_origins) == 1:
        return np.asarray(
            [tangent_for_single_sample(plane_origins[0], reference_centreline)]
        )

    reference_centreline = np.asarray(reference_centreline, dtype=float)
    reference_tangents = tangents_from_centreline(reference_centreline)

    normals = []
    for origin in np.asarray(plane_origins, dtype=float):
        nearest_idx = int(
            np.argmin(np.linalg.norm(reference_centreline - origin, axis=1))
        )
        normals.append(reference_tangents[nearest_idx])

    return np.asarray(normals, dtype=float)


def find_arc_position_for_z(centreline, z_value=0.0):
    """
    Find the first arc-length position where the centreline reaches z = z_value.

    If the centreline does not cross the requested z level exactly, use the
    closest sampled centreline location.
    """
    z = centreline[:, 2]
    diffs = np.diff(centreline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    for idx in range(len(z) - 1):
        z0 = z[idx]
        z1 = z[idx + 1]

        if np.isclose(z0, z_value):
            return arc[idx]

        crosses_level = (z0 - z_value) * (z1 - z_value) < 0
        if crosses_level or np.isclose(z1, z_value):
            dz = z1 - z0
            if np.isclose(dz, 0.0):
                return arc[idx]
            t = (z_value - z0) / dz
            return arc[idx] + t * seg_lengths[idx]

    nearest_idx = int(np.argmin(np.abs(z - z_value)))
    return arc[nearest_idx]


def trim_centreline_from_z(centreline, start_z=0.0, start_z_tolerance=0.0):
    """
    Rebuild the centreline so its first point is the location at z = start_z.

    If the curve does not cross that z value exactly, prepend a projected start
    point on z = start_z using the initial local centreline direction.
    """
    if np.abs(centreline[0, 2] - start_z) <= start_z_tolerance:
        return centreline.copy(), 0.0

    start_arc = find_arc_position_for_z(centreline, z_value=start_z)

    diffs = np.diff(centreline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = arc[-1]

    if total_length <= 0:
        return centreline.copy(), start_arc

    remaining_arc = arc[arc > start_arc]
    sample_positions = np.concatenate(([start_arc], remaining_arc))

    trimmed = np.column_stack(
        [
            np.interp(sample_positions, arc, centreline[:, axis])
            for axis in range(3)
        ]
    )

    if not np.isclose(trimmed[0, 2], start_z):
        if len(trimmed) >= 2:
            direction = trimmed[1] - trimmed[0]
        elif len(centreline) >= 2:
            direction = centreline[1] - centreline[0]
        else:
            direction = np.array([0.0, 0.0, 1.0], dtype=float)

        dz = direction[2]
        if np.isclose(dz, 0.0):
            projected_start = trimmed[0].copy()
            projected_start[2] = start_z
        else:
            t = (start_z - trimmed[0, 2]) / dz
            projected_start = trimmed[0] + t * direction
            projected_start[2] = start_z

        trimmed = np.vstack([projected_start, trimmed])

    return trimmed, start_arc


def resample_centreline_by_spacing(centreline, spacing, start_arc=0.0):
    """
    Place samples along the centreline at fixed Euclidean arc-length spacing.
    """
    if spacing <= 0:
        raise ValueError("Plane spacing must be positive.")

    diffs = np.diff(centreline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = arc[-1]

    if total_length <= 0:
        raise RuntimeError("Centreline length is zero.")

    start_arc = float(np.clip(start_arc, 0.0, total_length))
    sample_positions = np.arange(
        start_arc,
        total_length + spacing * 0.5,
        spacing,
    )
    if len(sample_positions) == 0:
        sample_positions = np.array([start_arc], dtype=float)
    sample_positions[-1] = min(sample_positions[-1], total_length)

    sampled = np.column_stack(
        [
            np.interp(sample_positions, arc, centreline[:, axis])
            for axis in range(3)
        ]
    )
    return sampled, sample_positions


def estimate_local_shell_radius(centreline, mesh):
    """
    Estimate a local centreline-to-shell radius using nearest mesh vertices.
    """
    centreline = np.asarray(centreline, dtype=float)
    if len(centreline) == 0:
        return np.array([], dtype=float)

    mesh_vertices = np.asarray(mesh.vertices, dtype=float)
    if len(mesh_vertices) == 0:
        return np.zeros(len(centreline), dtype=float)

    tree = cKDTree(mesh_vertices)
    distances, _ = tree.query(centreline, k=1)
    return np.asarray(distances, dtype=float)


def estimate_centreline_curvature(centreline):
    """
    Estimate scalar curvature from discrete centreline samples.
    """
    centreline = np.asarray(centreline, dtype=float)
    if len(centreline) < 3:
        return np.zeros(len(centreline), dtype=float)

    tangents = tangents_from_centreline(centreline)
    diffs = np.diff(centreline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    curvature = np.zeros(len(centreline), dtype=float)
    tangent_diffs = np.diff(tangents, axis=0)
    edge_curvature = np.linalg.norm(tangent_diffs, axis=1) / np.maximum(
        seg_lengths,
        1e-12,
    )

    curvature[0] = edge_curvature[0]
    curvature[-1] = edge_curvature[-1]
    if len(centreline) > 2:
        curvature[1:-1] = 0.5 * (edge_curvature[:-1] + edge_curvature[1:])
    return curvature


def resample_centreline_adaptive_by_spacing(
    centreline,
    spacing,
    mesh,
    start_arc=0.0,
    min_spacing_factor=0.35,
    outer_corner_spacing_safety=1.0,
):
    """
    Reduce centreline spacing in high-curvature regions so the outer bend
    remains covered.
    """
    if spacing <= 0:
        raise ValueError("Plane spacing must be positive.")

    centreline = np.asarray(centreline, dtype=float)
    diffs = np.diff(centreline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = arc[-1]

    if total_length <= 0:
        raise RuntimeError("Centreline length is zero.")

    start_arc = float(np.clip(start_arc, 0.0, total_length))
    curvature = estimate_centreline_curvature(centreline)
    local_radius = estimate_local_shell_radius(centreline, mesh)

    bend_factor = np.maximum(curvature * local_radius, 0.0)
    coverage_ratio = 1.0 + outer_corner_spacing_safety * np.sqrt(bend_factor)
    local_spacing = spacing / np.maximum(coverage_ratio, 1.0)
    min_spacing = spacing * float(np.clip(min_spacing_factor, 0.05, 1.0))
    local_spacing = np.clip(local_spacing, min_spacing, spacing)

    if len(local_spacing) >= 3:
        kernel = np.array([0.25, 0.5, 0.25], dtype=float)
        padded = np.pad(local_spacing, 1, mode="edge")
        local_spacing = np.convolve(padded, kernel, mode="valid")

    sample_positions = [start_arc]
    current_arc = start_arc
    while current_arc < total_length - 1e-9:
        step = float(np.interp(current_arc, arc, local_spacing))
        next_arc = min(current_arc + max(step, min_spacing), total_length)
        if next_arc <= current_arc + 1e-9:
            break
        sample_positions.append(next_arc)
        current_arc = next_arc

    sample_positions = np.asarray(sample_positions, dtype=float)
    sampled = np.column_stack(
        [
            np.interp(sample_positions, arc, centreline[:, axis])
            for axis in range(3)
        ]
    )
    return sampled, sample_positions


def polyline_length(polyline):
    polyline = np.asarray(polyline, dtype=float)
    if len(polyline) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())


def build_local_normal_frames(origins, tangents):
    """
    Build a smooth local frame (tangent, normal, binormal) along a centreline
    or plane-origin sequence.
    """
    tangents = np.asarray(tangents, dtype=float)
    origins = np.asarray(origins, dtype=float)
    if len(origins) != len(tangents):
        raise ValueError("origins and tangents must have the same length.")
    if len(origins) == 0:
        raise ValueError("Need at least one origin to build local frames.")

    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)

    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if np.abs(np.dot(reference, tangents[0])) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)

    normal0 = reference - np.dot(reference, tangents[0]) * tangents[0]
    norm0 = np.linalg.norm(normal0)
    if np.isclose(norm0, 0.0):
        normal0 = np.array([1.0, 0.0, 0.0], dtype=float)
        normal0 = normal0 - np.dot(normal0, tangents[0]) * tangents[0]
        norm0 = np.linalg.norm(normal0)
    normal0 = normal0 / max(norm0, 1e-12)
    binormal0 = np.cross(tangents[0], normal0)
    binormal0 = binormal0 / max(np.linalg.norm(binormal0), 1e-12)

    normals[0] = normal0
    binormals[0] = binormal0

    for idx in range(1, len(origins)):
        prev_normal = normals[idx - 1]
        tangent = tangents[idx]
        normal = prev_normal - np.dot(prev_normal, tangent) * tangent
        norm = np.linalg.norm(normal)
        if np.isclose(norm, 0.0):
            fallback = binormals[idx - 1]
            normal = fallback - np.dot(fallback, tangent) * tangent
            norm = np.linalg.norm(normal)
        normal = normal / max(norm, 1e-12)
        binormal = np.cross(tangent, normal)
        binormal = binormal / max(np.linalg.norm(binormal), 1e-12)
        normals[idx] = normal
        binormals[idx] = binormal

    return normals, binormals


def select_surface_intersection_point(
    section_points,
    origin,
    normal_axis,
    binormal_axis,
    target_angle,
    angle_tolerance_rad,
):
    deltas = section_points - origin
    local_x = deltas @ normal_axis
    local_y = deltas @ binormal_axis
    radii = np.hypot(local_x, local_y)
    if np.all(radii <= 1e-9):
        return None

    angles = np.arctan2(local_y, local_x)
    angle_diff = np.abs(np.angle(np.exp(1j * (angles - target_angle))))

    candidates = np.where(angle_diff <= angle_tolerance_rad)[0]
    if len(candidates) == 0:
        candidates = np.array([int(np.argmin(angle_diff))], dtype=int)

    best = candidates[np.argmax(radii[candidates])]
    return section_points[best]


def generate_nurbs_like_surface_intersections(
    centrelines,
    plane_origins,
    plane_normals,
    shell_points,
    plane_index,
    num_surfaces=16,
    angle_tolerance_deg=20.0,
    min_points_per_surface=3,
    branch_index=None,
    plane_branch_index=None,
    cancel_callback=None,
):
    """
    Generate model-surface points from a family of centreline-normal swept
    surfaces. Each swept surface is represented by a fixed angular direction in
    the local normal plane and is sampled from the plane-section points.
    """
    if isinstance(centrelines, np.ndarray):
        centreline_items = [(0, centrelines)]
    else:
        centreline_items = list(centrelines)

    if plane_branch_index is None:
        plane_branch_index = np.zeros(len(plane_origins), dtype=int)

    angle_tolerance_rad = np.deg2rad(angle_tolerance_deg)
    angles = np.linspace(0.0, 2.0 * np.pi, num_surfaces, endpoint=False)
    surface_curves = []

    for branch_idx, (branch_id, _centreline) in enumerate(centreline_items):
        if cancel_callback is not None and branch_idx % 5 == 0:
            cancel_callback()
        branch_plane_ids = np.where(plane_branch_index == branch_id)[0]
        if len(branch_plane_ids) == 0:
            continue

        branch_origins = plane_origins[branch_plane_ids]
        branch_tangents = plane_normals[branch_plane_ids]
        normal_axes, binormal_axes = build_local_normal_frames(
            branch_origins,
            branch_tangents,
        )

        for surface_id, angle in enumerate(angles):
            if cancel_callback is not None:
                cancel_callback()
            selected_points = []
            selected_plane_ids = []

            for local_idx, global_plane_id in enumerate(branch_plane_ids):
                if cancel_callback is not None and local_idx % 50 == 0:
                    cancel_callback()
                section_points = shell_points[plane_index == global_plane_id]
                if len(section_points) == 0:
                    continue

                point = select_surface_intersection_point(
                    section_points,
                    branch_origins[local_idx],
                    normal_axes[local_idx],
                    binormal_axes[local_idx],
                    angle,
                    angle_tolerance_rad,
                )
                if point is None:
                    continue

                selected_points.append(point)
                selected_plane_ids.append(global_plane_id)

            if len(selected_points) < min_points_per_surface:
                continue

            selected_points = np.asarray(selected_points, dtype=float)
            selected_plane_ids = np.asarray(selected_plane_ids, dtype=int)
            surface_curve = fit_branch_centreline(
                selected_points,
                spline_s=0.0,
                centreline_samples=max(len(selected_points), 50),
                preserve_endpoints=True,
            )
            surface_curves.append(
                {
                    "branch_id": branch_id,
                    "surface_id": surface_id,
                    "angle_rad": angle,
                    "points": selected_points,
                    "curve": surface_curve,
                    "plane_ids": selected_plane_ids,
                }
            )

    return surface_curves


def resample_polyline_by_spacing(polyline, spacing):
    """
    Resample a section polyline to obtain evenly spaced shell points.
    """
    if len(polyline) < 2:
        return polyline

    diffs = np.diff(polyline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = arc[-1]

    if total_length <= 0:
        return polyline[:1]

    sample_positions = np.arange(0.0, total_length + spacing * 0.5, spacing)
    sample_positions[-1] = min(sample_positions[-1], total_length)

    return np.column_stack(
        [
            np.interp(sample_positions, arc, polyline[:, axis])
            for axis in range(3)
        ]
    )


def filter_points_by_centreline_distance(points, origin, max_distance):
    points = np.asarray(points, dtype=float)
    max_distance = float(max(max_distance, 0.0))
    if len(points) == 0 or max_distance <= 0.0:
        return points

    distances = np.linalg.norm(points - np.asarray(origin, dtype=float), axis=1)
    return points[distances <= max_distance]


def _polyline_is_closed(polyline, tolerance):
    polyline = np.asarray(polyline, dtype=float)
    if len(polyline) < 3:
        return False
    return (
        np.linalg.norm(polyline[0] - polyline[-1])
        <= max(float(tolerance), 1e-6)
    )


def split_polyline_by_centreline_distance(
    polyline,
    origin,
    max_distance,
    tolerance,
    min_points=2,
):
    polyline = np.asarray(polyline, dtype=float)
    max_distance = float(max(max_distance, 0.0))
    if len(polyline) < 2:
        return [polyline] if len(polyline) else []
    if max_distance <= 0.0:
        return [polyline]

    distances = np.linalg.norm(polyline - np.asarray(origin, dtype=float), axis=1)
    keep_mask = distances <= max_distance
    if not np.any(keep_mask):
        return []

    return _split_polyline_by_mask(
        polyline,
        keep_mask,
        closed=_polyline_is_closed(polyline, tolerance),
        min_points=min_points,
    )


def build_branch_ownership_tree(branch_centrelines):
    sample_points = []
    sample_branch_labels = []

    for branch_id, centreline in branch_centrelines:
        centreline = np.asarray(centreline, dtype=float)
        if len(centreline) == 0:
            continue
        sample_points.append(centreline)
        sample_branch_labels.append(
            np.full(len(centreline), int(branch_id), dtype=int)
        )

    if not sample_points:
        return None, None

    stacked_points = np.vstack(sample_points)
    stacked_labels = np.concatenate(sample_branch_labels)
    return cKDTree(stacked_points), stacked_labels


def assign_points_to_branch_centrelines(
    points,
    ownership_tree,
    ownership_labels,
    tie_tolerance=1e-6,
    neighbour_count=8,
):
    points = np.asarray(points, dtype=float)
    if (
        ownership_tree is None
        or ownership_labels is None
        or len(points) == 0
    ):
        return None

    ownership_labels = np.asarray(ownership_labels, dtype=int)
    k = max(1, min(int(neighbour_count), len(ownership_labels)))
    distances, indices = ownership_tree.query(points, k=k)

    if k == 1:
        distances = np.asarray(distances, dtype=float).reshape(-1, 1)
        indices = np.asarray(indices, dtype=int).reshape(-1, 1)
    else:
        distances = np.asarray(distances, dtype=float)
        indices = np.asarray(indices, dtype=int)

    assigned = ownership_labels[indices[:, 0]].astype(int, copy=True)
    for point_idx in range(len(points)):
        best_distance = float(distances[point_idx, 0])
        tied_mask = distances[point_idx] <= best_distance + float(tie_tolerance)
        tied_labels = ownership_labels[indices[point_idx, tied_mask]]
        if len(tied_labels) > 0:
            assigned[point_idx] = int(np.min(tied_labels))

    return assigned


def split_polyline_by_branch_ownership(
    polyline,
    ownership_tree,
    ownership_labels,
    owner_branch_id,
    tolerance,
    min_points=2,
):
    polyline = np.asarray(polyline, dtype=float)
    if len(polyline) < 2 or ownership_tree is None or ownership_labels is None:
        return [polyline]

    assigned_labels = assign_points_to_branch_centrelines(
        polyline,
        ownership_tree,
        ownership_labels,
        tie_tolerance=max(float(tolerance), 1e-6),
    )
    if assigned_labels is None:
        return [polyline]

    keep_mask = assigned_labels == int(owner_branch_id)
    if not np.any(keep_mask):
        return []

    return _split_polyline_by_mask(
        polyline,
        keep_mask,
        closed=_polyline_is_closed(polyline, tolerance),
        min_points=min_points,
    )


def _section_crosses_plane(section_points, plane_origin, plane_normal, tolerance):
    section_points = np.asarray(section_points, dtype=float)
    signed_distances = (section_points - np.asarray(plane_origin, dtype=float)) @ np.asarray(
        plane_normal,
        dtype=float,
    )
    return (
        float(np.min(signed_distances)) < -float(tolerance)
        and float(np.max(signed_distances)) > float(tolerance)
    )


def sections_overlap_by_plane_intersection(
    previous_points,
    previous_origin,
    previous_normal,
    current_points,
    current_origin,
    current_normal,
    tolerance,
):
    """
    Approximate whether two slice sections overlap by checking whether each
    section is cut by the other section's plane.

    If both discrete section contours cross the other plane, the plane/mesh
    intersections are no longer cleanly ordered along the centreline and are
    treated as overlapping slices.
    """
    return _section_crosses_plane(
        current_points,
        previous_origin,
        previous_normal,
        tolerance,
    ) and _section_crosses_plane(
        previous_points,
        current_origin,
        current_normal,
        tolerance,
    )


def _normalize_vector(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return vector / norm


def _split_shared_endpoint_groups(branch_centrelines, tolerance=1e-6):
    endpoint_groups = []

    for branch_id, centreline in branch_centrelines:
        centreline = np.asarray(centreline, dtype=float)
        if len(centreline) < 2:
            continue
        for position, point in (("start", centreline[0]), ("end", centreline[-1])):
            matched_group = None
            for group in endpoint_groups:
                if np.linalg.norm(point - group["point"]) <= tolerance:
                    matched_group = group
                    break

            if matched_group is None:
                matched_group = {
                    "point": np.asarray(point, dtype=float),
                    "members": [],
                }
                endpoint_groups.append(matched_group)

            matched_group["members"].append((branch_id, position))

    return endpoint_groups


def build_branch_endpoint_clip_planes(branch_centrelines, tolerance=1e-6):
    """
    Build fixed clipping planes at shared branch endpoints.

    Each branch is clipped to its own span between shared endpoints so slices
    near a junction cannot keep extending through the common junction region.
    """
    clip_planes_by_branch = {
        int(branch_id): []
        for branch_id, _ in branch_centrelines
    }
    shared_groups = [
        group
        for group in _split_shared_endpoint_groups(
            branch_centrelines,
            tolerance=tolerance,
        )
        if len(group["members"]) > 1
    ]
    if not shared_groups:
        return clip_planes_by_branch

    centreline_map = {
        int(branch_id): np.asarray(centreline, dtype=float)
        for branch_id, centreline in branch_centrelines
    }

    for group in shared_groups:
        junction_point = np.asarray(group["point"], dtype=float)
        for branch_id, position in group["members"]:
            centreline = centreline_map.get(int(branch_id))
            if centreline is None or len(centreline) < 2:
                continue

            if position == "start":
                tangent = _normalize_vector(centreline[1] - centreline[0])
                side_sign = 1.0
            else:
                tangent = _normalize_vector(centreline[-1] - centreline[-2])
                side_sign = -1.0

            if tangent is None:
                continue

            clip_planes_by_branch[int(branch_id)].append(
                (junction_point.copy(), tangent, side_sign)
            )

    return clip_planes_by_branch


def _preferred_halfspace_sign(
    current_origin,
    current_normal,
    reference_origin,
    reference_normal,
    tolerance,
):
    current_origin = np.asarray(current_origin, dtype=float)
    current_normal = np.asarray(current_normal, dtype=float)
    reference_origin = np.asarray(reference_origin, dtype=float)
    reference_normal = np.asarray(reference_normal, dtype=float)

    signed = float((current_origin - reference_origin) @ reference_normal)
    if abs(signed) > tolerance:
        return 1.0 if signed >= 0.0 else -1.0

    probe_distance = max(4.0 * float(tolerance), 1e-3)
    probe = current_origin + current_normal * probe_distance
    probe_signed = float((probe - reference_origin) @ reference_normal)
    if abs(probe_signed) > tolerance:
        return 1.0 if probe_signed >= 0.0 else -1.0

    return 1.0


def _split_polyline_by_mask(polyline, keep_mask, closed=False, min_points=2):
    polyline = np.asarray(polyline, dtype=float)
    keep_mask = np.asarray(keep_mask, dtype=bool)
    fragments = []
    start = None

    for idx, keep in enumerate(keep_mask):
        if keep and start is None:
            start = idx
        elif not keep and start is not None:
            fragment = polyline[start:idx]
            if len(fragment) >= min_points:
                fragments.append(fragment)
            start = None

    if start is not None:
        fragment = polyline[start:]
        if len(fragment) >= min_points:
            fragments.append(fragment)

    if (
        closed
        and len(fragments) >= 2
        and keep_mask[0]
        and keep_mask[-1]
    ):
        merged = np.vstack([fragments[-1], fragments[0]])
        middle = fragments[1:-1]
        fragments = [merged, *middle]

    return fragments


def clip_polyline_against_planes(
    polyline,
    current_origin,
    current_normal,
    clip_planes,
    tolerance,
):
    """
    Clip a sampled section polyline against already accepted slicing planes.

    Earlier planes are treated as authoritative. Later planes keep only the
    points that lie on the same preferred side of each previously accepted
    plane as the later plane's centreline origin.
    """
    polyline = np.asarray(polyline, dtype=float)
    if len(polyline) < 2 or not clip_planes:
        return [polyline]

    closed = _polyline_is_closed(polyline, tolerance)
    keep_mask = np.ones(len(polyline), dtype=bool)

    for clip_plane in clip_planes:
        if len(clip_plane) >= 3:
            reference_origin, reference_normal, side_sign = clip_plane[:3]
        else:
            reference_origin, reference_normal = clip_plane
            side_sign = None

        if side_sign is None:
            side_sign = _preferred_halfspace_sign(
                current_origin,
                current_normal,
                reference_origin,
                reference_normal,
                tolerance,
            )
        signed_distances = (polyline - np.asarray(reference_origin, dtype=float)) @ np.asarray(
            reference_normal,
            dtype=float,
        )
        keep_mask &= side_sign * signed_distances >= -float(tolerance)
        if not np.any(keep_mask):
            return []

    return _split_polyline_by_mask(
        polyline,
        keep_mask,
        closed=closed,
        min_points=2,
    )


def slice_shell_points_from_faces(
    mesh,
    centreline,
    plane_spacing,
    shell_point_spacing,
    start_z=0.0,
    anchor_to_start_z=True,
    min_section_points=8,
    adaptive_plane_spacing=True,
    adaptive_spacing_min_factor=0.35,
    outer_corner_spacing_safety=1.0,
    max_point_distance_from_centreline=0.0,
    cancel_callback=None,
    clip_against_planes=None,
    branch_ownership_tree=None,
    branch_ownership_labels=None,
    owner_branch_id=None,
    progress_label=None,
):
    """
    Intersect the mesh with planes normal to the centreline and sample shell
    points from the resulting face-section polylines.
    """
    if anchor_to_start_z:
        start_arc = find_arc_position_for_z(centreline, z_value=start_z)
    else:
        start_arc = 0.0
    if adaptive_plane_spacing:
        plane_origins, plane_distances = resample_centreline_adaptive_by_spacing(
            centreline,
            plane_spacing,
            mesh,
            start_arc=start_arc,
            min_spacing_factor=adaptive_spacing_min_factor,
            outer_corner_spacing_safety=outer_corner_spacing_safety,
        )
    else:
        plane_origins, plane_distances = resample_centreline_by_spacing(
            centreline,
            plane_spacing,
            start_arc=start_arc,
        )
    plane_normals = plane_normals_from_samples(plane_origins, centreline)
    if progress_label:
        logger.info(
            "%s: evaluating %s candidate slicing planes.",
            progress_label,
            len(plane_origins),
        )

    shell_points = []
    plane_index = []
    plane_fragment_index = []
    section_count = 0
    sampled_point_count = 0
    radius_rejected_count = 0
    overlap_rejected_count = 0
    clip_rejected_count = 0
    ownership_rejected_count = 0
    accepted_plane_origins = []
    accepted_plane_normals = []
    accepted_plane_distances = []
    accepted_plane_points = []
    max_point_distance_from_centreline = float(
        max(max_point_distance_from_centreline, 0.0)
    )
    overlap_tolerance = max(
        min(float(shell_point_spacing), float(plane_spacing)) * 0.25,
        1e-4,
    )
    static_clip_planes = list(clip_against_planes or [])
    clip_planes = list(static_clip_planes)
    plane_shell_radii = estimate_local_shell_radius(plane_origins, mesh)

    for idx, (origin, normal) in enumerate(zip(plane_origins, plane_normals)):
        if cancel_callback is not None and idx % 10 == 0:
            cancel_callback()
        remaining_arc = float(plane_distances[-1] - plane_distances[idx])
        terminal_relax_distance = max(
            float(plane_shell_radii[idx]) if idx < len(plane_shell_radii) else 0.0,
            float(plane_spacing),
        )
        relax_dynamic_clipping = remaining_arc <= terminal_relax_distance
        active_clip_planes = (
            static_clip_planes if relax_dynamic_clipping else clip_planes
        )
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            continue
        section_count += 1

        try:
            polylines = section.discrete
        except Exception:
            continue

        plane_chunks = []
        plane_chunk_fragment_ids = []

        for poly_idx, polyline in enumerate(polylines):
            if cancel_callback is not None and poly_idx % 25 == 0:
                cancel_callback()
            if polyline is None or len(polyline) < min_section_points:
                continue

            sampled_polyline = resample_polyline_by_spacing(
                np.asarray(polyline, dtype=float),
                shell_point_spacing,
            )
            if len(sampled_polyline) == 0:
                continue

            sampled_point_count += len(sampled_polyline)

            radius_filtered_fragments = split_polyline_by_centreline_distance(
                sampled_polyline,
                origin,
                max_point_distance_from_centreline,
                overlap_tolerance,
            )
            kept_radius_points = sum(
                len(fragment) for fragment in radius_filtered_fragments
            )
            radius_rejected_count += len(sampled_polyline) - kept_radius_points
            if not radius_filtered_fragments:
                continue

            for radius_filtered_polyline in radius_filtered_fragments:
                clipped_fragments = clip_polyline_against_planes(
                    radius_filtered_polyline,
                    origin,
                    normal,
                    active_clip_planes,
                    overlap_tolerance,
                )
                if not clipped_fragments:
                    clip_rejected_count += 1
                    continue

                for clipped_fragment in clipped_fragments:
                    if len(clipped_fragment) < 2:
                        continue
                    owned_fragments = split_polyline_by_branch_ownership(
                        clipped_fragment,
                        branch_ownership_tree,
                        branch_ownership_labels,
                        owner_branch_id,
                        overlap_tolerance,
                    )
                    if not owned_fragments:
                        ownership_rejected_count += 1
                        continue

                    for owned_fragment in owned_fragments:
                        if len(owned_fragment) < 2:
                            continue
                        fragment_id = len(plane_chunk_fragment_ids)
                        plane_chunks.append(owned_fragment)
                        plane_chunk_fragment_ids.append(
                            np.full(len(owned_fragment), fragment_id, dtype=int)
                        )

        if not plane_chunks:
            continue

        plane_points = np.vstack(plane_chunks)
        plane_points_fragment_index = np.concatenate(plane_chunk_fragment_ids)
        if len(plane_points) < 2:
            continue
        overlaps_previous = False
        recent_count = min(3, len(accepted_plane_points))
        for back in range(1, recent_count + 1):
            prev_points = accepted_plane_points[-back]
            prev_origin = accepted_plane_origins[-back]
            prev_normal = accepted_plane_normals[-back]
            if sections_overlap_by_plane_intersection(
                prev_points,
                prev_origin,
                prev_normal,
                plane_points,
                origin,
                normal,
                overlap_tolerance,
            ):
                overlaps_previous = True
                break

        if overlaps_previous:
            overlap_rejected_count += 1
            continue

        accepted_idx = len(accepted_plane_origins)
        accepted_plane_origins.append(np.asarray(origin, dtype=float))
        accepted_plane_normals.append(np.asarray(normal, dtype=float))
        accepted_plane_distances.append(float(plane_distances[idx]))
        accepted_plane_points.append(plane_points)
        clip_planes.append((np.asarray(origin, dtype=float), np.asarray(normal, dtype=float)))
        shell_points.append(plane_points)
        plane_index.append(np.full(len(plane_points), accepted_idx, dtype=int))
        plane_fragment_index.append(plane_points_fragment_index)

    if not shell_points:
        details = (
            f" sections={section_count}, sampled_points={sampled_point_count}, "
            f"radius_rejected={radius_rejected_count}, "
            f"clip_rejected={clip_rejected_count}, "
            f"ownership_rejected={ownership_rejected_count}, "
            f"overlap_rejected={overlap_rejected_count}, "
            f"max_radius={max_point_distance_from_centreline}"
        )
        raise RuntimeError(
            "No shell points were found from plane/mesh intersections."
            + details
        )

    if progress_label:
        logger.info(
            "%s: accepted %s planes and %s shell points.",
            progress_label,
            len(accepted_plane_origins),
            int(sum(len(points) for points in shell_points)),
        )

    if overlap_rejected_count > 0:
        logger.info(
            "Rejected %s overlapping slicing planes.",
            overlap_rejected_count,
        )
    if clip_rejected_count > 0:
        logger.info(
            "Clipped away %s section fragments against previously accepted planes.",
            clip_rejected_count,
        )
    if ownership_rejected_count > 0:
        logger.info(
            "Discarded %s section fragments due to branch ownership splitting.",
            ownership_rejected_count,
        )

    return (
        np.vstack(shell_points),
        np.concatenate(plane_index),
        np.concatenate(plane_fragment_index),
        np.asarray(accepted_plane_origins, dtype=float),
        np.asarray(accepted_plane_normals, dtype=float),
        np.asarray(accepted_plane_distances, dtype=float) - start_arc,
        start_arc,
    )


def slice_shell_points_from_branch_centrelines(
    mesh,
    branch_centrelines,
    plane_spacing,
    shell_point_spacing,
    start_z=0.0,
    start_z_tolerance=0.0,
    min_section_points=8,
    adaptive_plane_spacing=True,
    adaptive_spacing_min_factor=0.35,
    outer_corner_spacing_safety=1.0,
    max_point_distance_from_centreline=0.0,
    cancel_callback=None,
    branch_progress_callback=None,
):
    """
    Run the same plane-based shell-point extraction for multiple branch
    centrelines and keep branch ownership for each generated point.
    """
    all_shell_points = []
    all_plane_index = []
    all_plane_fragment_index = []
    all_branch_index = []
    all_plane_origins = []
    all_plane_normals = []
    all_plane_distances = []
    all_plane_branch_index = []
    branch_metadata = []
    ownership_tree, ownership_labels = build_branch_ownership_tree(branch_centrelines)
    branch_endpoint_clip_planes = build_branch_endpoint_clip_planes(
        branch_centrelines,
        tolerance=1e-5,
    )
    logger.info(
        "Slicing shell points across %s branch centrelines.",
        len(branch_centrelines),
    )

    plane_offset = 0

    branch_count = len(branch_centrelines)
    for branch_pos, (branch_id, centreline) in enumerate(branch_centrelines):
        if cancel_callback is not None and branch_pos % 5 == 0:
            cancel_callback()
        original_length = polyline_length(centreline)
        branch_label = (
            f"Branch {branch_pos + 1}/{branch_count} (id={branch_id})"
        )
        logger.info(
            "%s: starting with centreline length %.2f mm.",
            branch_label,
            original_length,
        )
        if branch_progress_callback is not None:
            branch_progress_callback(
                "start",
                branch_pos,
                branch_count,
                branch_id,
                {"centreline_length": float(original_length)},
            )
        anchor_branch_to_start_z = branch_id == 0
        if anchor_branch_to_start_z:
            trimmed_centreline, start_arc_before_trim = trim_centreline_from_z(
                centreline,
                start_z=start_z,
                start_z_tolerance=start_z_tolerance,
            )
        else:
            trimmed_centreline = np.asarray(centreline, dtype=float)
            start_arc_before_trim = 0.0

        trimmed_length = polyline_length(trimmed_centreline)

        try:
            (
                shell_points,
                plane_index,
                plane_fragment_index,
                plane_origins,
                plane_normals,
                plane_distances,
                start_arc,
            ) = slice_shell_points_from_faces(
                mesh,
                trimmed_centreline,
                plane_spacing=plane_spacing,
                shell_point_spacing=shell_point_spacing,
                start_z=start_z,
                anchor_to_start_z=anchor_branch_to_start_z,
                min_section_points=min_section_points,
                adaptive_plane_spacing=adaptive_plane_spacing,
                adaptive_spacing_min_factor=adaptive_spacing_min_factor,
                outer_corner_spacing_safety=outer_corner_spacing_safety,
                max_point_distance_from_centreline=max_point_distance_from_centreline,
                cancel_callback=cancel_callback,
                clip_against_planes=branch_endpoint_clip_planes.get(
                    int(branch_id),
                    [],
                ),
                branch_ownership_tree=ownership_tree,
                branch_ownership_labels=ownership_labels,
                owner_branch_id=branch_id,
                progress_label=branch_label,
            )
        except RuntimeError as exc:
            logger.warning("%s: %s", branch_label, exc)
            if branch_progress_callback is not None:
                branch_progress_callback(
                    "skipped",
                    branch_pos,
                    branch_count,
                    branch_id,
                    {"error": str(exc)},
                )
            continue

        all_shell_points.append(shell_points)
        all_plane_index.append(plane_index + plane_offset)
        all_plane_fragment_index.append(plane_fragment_index)
        all_branch_index.append(
            np.full(len(shell_points), branch_id, dtype=int)
        )
        all_plane_origins.append(plane_origins)
        all_plane_normals.append(plane_normals)
        all_plane_distances.append(plane_distances)
        all_plane_branch_index.append(
            np.full(len(plane_origins), branch_id, dtype=int)
        )
        branch_metadata.append(
            {
                "branch_id": branch_id,
                "centreline": trimmed_centreline,
                "original_length": original_length,
                "trimmed_length": trimmed_length,
                "plane_count": len(plane_origins),
                "anchor_to_start_z": anchor_branch_to_start_z,
                "start_arc_before_trim": start_arc_before_trim,
                "start_arc_after_trim": start_arc,
                "plane_offset": plane_offset,
                "shared_endpoint_clip_count": len(
                    branch_endpoint_clip_planes.get(int(branch_id), [])
                ),
            }
        )
        if branch_progress_callback is not None:
            branch_progress_callback(
                "complete",
                branch_pos,
                branch_count,
                branch_id,
                {
                    "plane_count": int(len(plane_origins)),
                    "shell_point_count": int(len(shell_points)),
                },
            )
        plane_offset += len(plane_origins)

    if not all_shell_points:
        raise RuntimeError(
            "No shell points were found for any tree-based centreline branch."
        )

    return (
        np.vstack(all_shell_points),
        np.concatenate(all_plane_index),
        np.concatenate(all_plane_fragment_index),
        np.concatenate(all_branch_index),
        np.vstack(all_plane_origins),
        np.vstack(all_plane_normals),
        np.concatenate(all_plane_distances),
        np.concatenate(all_plane_branch_index),
        branch_metadata,
    )


def export_shell_points_csv(
    output_path,
    shell_points,
    plane_index,
    plane_distances,
    branch_index=None,
):
    point_plane_distances = plane_distances[plane_index]
    if branch_index is None:
        data = np.column_stack([shell_points, plane_index, point_plane_distances])
        header = "x,y,z,plane_index,centreline_distance"
    else:
        data = np.column_stack(
            [
                shell_points,
                branch_index,
                plane_index,
                point_plane_distances,
            ]
        )
        header = "x,y,z,branch_index,plane_index,centreline_distance"
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")


def export_shell_layers_json(
    output_path,
    shell_points,
    plane_index,
    plane_origins,
    plane_normals,
    plane_distances,
    plane_fragment_index=None,
    branch_index=None,
    plane_branch_index=None,
    metadata=None,
):
    """
    Write plane-grouped shell points as one-based layers.
    """
    shell_points = np.asarray(shell_points, dtype=float)
    plane_index = np.asarray(plane_index, dtype=int)
    plane_origins = np.asarray(plane_origins, dtype=float)
    plane_normals = np.asarray(plane_normals, dtype=float)
    plane_distances = np.asarray(plane_distances, dtype=float)

    if plane_fragment_index is not None:
        plane_fragment_index = np.asarray(plane_fragment_index, dtype=int)
    if branch_index is not None:
        branch_index = np.asarray(branch_index, dtype=int)
    if plane_branch_index is not None:
        plane_branch_index = np.asarray(plane_branch_index, dtype=int)

    layers = []
    for plane_id in range(len(plane_origins)):
        mask = plane_index == plane_id
        points = shell_points[mask]
        fragment_ids = plane_fragment_index[mask] if plane_fragment_index is not None else None
        point_records = [
            dict(
                {
                    "point": int(point_id + 1),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                },
                **(
                    {"fragment_index": int(fragment_ids[point_id])}
                    if fragment_ids is not None
                    else {}
                ),
            )
            for point_id, point in enumerate(points)
        ]

        branch_id = None
        if plane_branch_index is not None and plane_id < len(plane_branch_index):
            branch_id = int(plane_branch_index[plane_id])
        elif branch_index is not None and np.any(mask):
            branch_id = int(branch_index[np.flatnonzero(mask)[0]])

        layer = {
            "layer": int(plane_id + 1),
            "plane_index": int(plane_id),
            "branch_index": branch_id,
            "centreline_distance": float(plane_distances[plane_id]),
            "origin": {
                "x": float(plane_origins[plane_id, 0]),
                "y": float(plane_origins[plane_id, 1]),
                "z": float(plane_origins[plane_id, 2]),
            },
            "normal": {
                "x": float(plane_normals[plane_id, 0]),
                "y": float(plane_normals[plane_id, 1]),
                "z": float(plane_normals[plane_id, 2]),
            },
            "point_count": int(len(points)),
            "fragment_count": (
                int(len(np.unique(fragment_ids)))
                if fragment_ids is not None and len(fragment_ids) > 0
                else (1 if len(points) > 0 else 0)
            ),
            "points": point_records,
        }
        layers.append(layer)

    payload = {
        "layer_count": int(len(layers)),
        "point_count": int(len(shell_points)),
        "coordinate_units": "mm",
        "layers": layers,
    }
    if metadata:
        payload["metadata"] = metadata

    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def draw_plot_context(
    ax,
    mesh,
    centrelines,
    spheres=None,
    title="",
):
    ax.add_collection3d(
        Poly3DCollection(
            mesh.triangles,
            facecolor="lightgray",
            edgecolor="none",
            alpha=0.12,
        )
    )

    if spheres:
        u, v = np.mgrid[0:2*np.pi:16j, 0:np.pi:8j]
        for center, radius in spheres:
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            ax.plot_surface(
                x,
                y,
                z,
                color="red",
                alpha=0.06,
                linewidth=0,
                shade=False,
            )

    if isinstance(centrelines, np.ndarray):
        centreline_iter = [(0, centrelines)]
    else:
        centreline_iter = list(centrelines)

    cmap = plt.get_cmap("tab10", max(1, len(centreline_iter)))
    for idx, (_, centreline) in enumerate(centreline_iter):
        label = "HitBox centreline" if idx == 0 else None
        ax.plot(
            centreline[:, 0],
            centreline[:, 1],
            centreline[:, 2],
            color=cmap(idx),
            linewidth=2,
            label=label,
        )

    ax.set_box_aspect([1, 1, 1])
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.view_init(elev=0, azim=90)
    ax.set_title(title)
    return cmap, centreline_iter


def plot_shell_points(
    mesh,
    centrelines,
    shell_points,
    plane_origins,
    spheres=None,
    branch_index=None,
    surface_curves=None,
    point_size=4,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    cmap, _ = draw_plot_context(
        ax,
        mesh,
        centrelines,
        spheres=spheres,
        title="Shell points from STL faces aligned to HitBox centreline",
    )

    if branch_index is None:
        ax.scatter(
            shell_points[:, 0],
            shell_points[:, 1],
            shell_points[:, 2],
            s=point_size,
            c="darkorange",
            alpha=0.75,
            label="Plane-aligned shell points",
        )
    else:
        colors = [cmap(int(i) % cmap.N) for i in branch_index]
        ax.scatter(
            shell_points[:, 0],
            shell_points[:, 1],
            shell_points[:, 2],
            s=point_size,
            c=colors,
            alpha=0.75,
            label="Branch-aligned shell points",
        )

    ax.scatter(
        plane_origins[:, 0],
        plane_origins[:, 1],
        plane_origins[:, 2],
        s=12,
        c="black",
        alpha=0.9,
        label="Plane origins",
    )
    ax.legend()
    plt.show()


def plot_surface_curves(
    mesh,
    centrelines,
    surface_curves,
    spheres=None,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    cmap, _ = draw_plot_context(
        ax,
        mesh,
        centrelines,
        spheres=spheres,
        title="Centreline-normal surface curves",
    )

    for idx, surface in enumerate(surface_curves):
        curve = surface["curve"]
        branch_id = surface["branch_id"]
        color = cmap(int(branch_id) % cmap.N)
        label = "Surface curves" if idx == 0 else None
        ax.plot(
            curve[:, 0],
            curve[:, 1],
            curve[:, 2],
            color=color,
            linewidth=1.5,
            alpha=0.9,
            label=label,
        )

    ax.legend()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample shell points from STL faces and index them by the "
            "HitBoxMethod centreline."
        )
    )
    parser.add_argument(
        "--mesh",
        default=CONFIG["mesh"],
        
        #STLfiles/EdgeCaseSplit.stl
        #STLfiles/CustomEdgeCaseThesis.stl
        #STLfiles/EdgeCaseSplitJoin.stl
        
        help="Path to the STL mesh.",
    )
    parser.add_argument(
        "--line-method",
        choices=("auto", "single", "tree"),
        default=CONFIG["line_method"],
        help="Choose automatic, single-spline, or tree-based branch centrelines.",
    )
    parser.add_argument(
        "--sphere-generation-method",
        choices=("auto", "skeleton_paths", "component_centroid"),
        default=CONFIG["sphere_generation_method"],
        help="Choose how HitBox spheres are generated before centreline fitting.",
    )
    parser.add_argument(
        "--plane-spacing",
        type=float,
        default=CONFIG["plane_spacing"],
        help="Distance between consecutive slicing planes along the centreline.",
    )
    parser.add_argument(
        "--shell-point-spacing",
        type=float,
        default=CONFIG["shell_point_spacing"],
        help="Spacing between sampled shell points within each section curve.",
    )
    parser.add_argument(
        "--start-z",
        type=float,
        default=CONFIG["start_z"],
        help="Anchor the first slicing plane at the centreline location for this z value.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=CONFIG["knn_k"],
        help="Neighbour count used by the HitBoxMethod graph.",
    )
    parser.add_argument(
        "--sphere-min-diameter",
        type=float,
        default=CONFIG["sphere_min_diameter"],
        help="Minimum allowed HitBox sphere diameter in mm. Use 0 to disable.",
    )
    parser.add_argument(
        "--sphere-max-diameter",
        type=float,
        default=CONFIG["sphere_max_diameter"],
        help="Maximum allowed HitBox sphere diameter in mm. Use 0 to disable.",
    )
    parser.add_argument(
        "--overlap-factor",
        type=float,
        default=CONFIG["overlap_factor"],
        help="Sphere overlap factor used when filtering generated HitBox spheres.",
    )
    parser.add_argument(
        "--spline-s",
        type=float,
        default=CONFIG["spline_s"],
        help="Spline smoothing used for the HitBox centreline.",
    )
    parser.add_argument(
        "--centreline-samples",
        type=int,
        default=CONFIG["centreline_samples"],
        help="Number of samples used to evaluate the centreline spline.",
    )
    parser.add_argument(
        "--tree-sphere-graph-k",
        type=int,
        default=CONFIG["tree_sphere_graph_k"],
        help="Sphere-graph neighbour count for the tree-based centreline method.",
    )
    parser.add_argument(
        "--graph-strategy",
        choices=("mst", "knn", "complete_mst"),
        default=CONFIG["graph_strategy"],
        help="Sphere graph strategy used for centreline ordering.",
    )
    parser.add_argument(
        "--csv",
        default=CONFIG["csv"],
        help="Optional CSV output path for the shell point table.",
    )
    parser.add_argument(
        "--json",
        default=CONFIG["json"],
        help="Optional JSON output path for layer-grouped shell points.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=CONFIG["no_plot"],
        help="Disable the 3D plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    mesh = load_mesh(args.mesh)
    centreline = compute_hitbox_centreline(
        mesh,
        knn_k=args.knn_k,
        overlap_factor=args.overlap_factor,
        sphere_min_diameter=args.sphere_min_diameter,
        sphere_max_diameter=args.sphere_max_diameter,
        sphere_generation_method=args.sphere_generation_method,
        spline_s=args.spline_s,
        centreline_samples=args.centreline_samples,
    )
    centreline, centreline_start_arc = trim_centreline_from_z(
        centreline,
        start_z=args.start_z,
        start_z_tolerance=CONFIG["start_z_tolerance"],
    )

    (
        shell_points,
        plane_index,
        plane_fragment_index,
        plane_origins,
        plane_normals,
        plane_distances,
        start_arc,
    ) = slice_shell_points_from_faces(
        mesh,
        centreline,
        plane_spacing=args.plane_spacing,
        shell_point_spacing=args.shell_point_spacing,
        start_z=args.start_z,
    )
    print(f"Centreline samples: {len(centreline)}")
    print(
        f"Centreline starts at z={args.start_z:.4f}"
        f" after trimming {centreline_start_arc:.4f} units"
    )
    print(f"Planes placed along centreline: {len(plane_origins)}")
    print(f"Shell points sampled from face intersections: {len(shell_points)}")
    print(
        f"First plane anchored at z={args.start_z:.4f}"
        f" on centreline arc distance {start_arc:.4f}"
    )
    print(
        "Centreline plane distances:"
        f" start={plane_distances.min():.4f},"
        f" end={plane_distances.max():.4f},"
        f" spacing≈{args.plane_spacing:.4f}"
    )
    print(
        "Plane normal vectors computed for each plane:"
        f" {len(plane_normals)}"
    )

    if args.csv:
        output_path = Path(args.csv)
        export_shell_points_csv(
            output_path,
            shell_points,
            plane_index,
            plane_distances,
        )
        print(f"Saved shell point table to {output_path}")

    if args.json:
        output_path = Path(args.json)
        export_shell_layers_json(
            output_path,
            shell_points,
            plane_index,
            plane_origins,
            plane_normals,
            plane_distances,
            plane_fragment_index=plane_fragment_index,
        )
        print(f"Saved layer JSON to {output_path}")

    if not args.no_plot:
        plot_shell_points(
            mesh,
            centreline,
            shell_points,
            plane_origins,
        )


def main_branch_aware():
    args = parse_args()

    mesh = load_mesh(args.mesh)
    spheres = compute_hitbox_spheres(
        mesh,
        knn_k=args.knn_k,
        overlap_factor=args.overlap_factor,
        min_diameter=args.sphere_min_diameter,
        max_diameter=args.sphere_max_diameter,
        sphere_generation_method=args.sphere_generation_method,
    )

    branch_index = None
    effective_line_method = args.line_method
    if effective_line_method == "auto":
        if detect_branching_from_spheres(
            spheres,
            sphere_graph_k=args.tree_sphere_graph_k,
        ):
            effective_line_method = "tree"
        else:
            effective_line_method = "single"
        print(
            f"Requested line method: auto -> using {effective_line_method}"
        )

    if effective_line_method == "single":
        centreline = fit_single_sphere_centreline(
            spheres,
            sphere_graph_k=args.tree_sphere_graph_k,
            graph_strategy=args.graph_strategy,
            spline_s=args.spline_s,
            centreline_samples=args.centreline_samples,
        )
        centreline, centreline_start_arc = trim_centreline_from_z(
            centreline,
            start_z=args.start_z,
            start_z_tolerance=CONFIG["start_z_tolerance"],
        )

        (
            shell_points,
            plane_index,
            plane_fragment_index,
            plane_origins,
            plane_normals,
            plane_distances,
            start_arc,
        ) = slice_shell_points_from_faces(
            mesh,
            centreline,
            plane_spacing=args.plane_spacing,
            shell_point_spacing=args.shell_point_spacing,
            start_z=args.start_z,
        )
        centrelines_for_plot = [(0, centreline)]
        plane_branch_index = np.zeros(len(plane_origins), dtype=int)
        surface_curves = generate_nurbs_like_surface_intersections(
            centrelines_for_plot,
            plane_origins,
            plane_normals,
            shell_points,
            plane_index,
            num_surfaces=CONFIG["nurbs_surface_count"],
            angle_tolerance_deg=CONFIG["nurbs_angle_tolerance_deg"],
            min_points_per_surface=CONFIG["nurbs_min_points"],
            plane_branch_index=plane_branch_index,
        )

        print(f"Line method: {effective_line_method}")
        print(f"Centreline samples: {len(centreline)}")
        print(
            f"Centreline starts at z={args.start_z:.4f}"
            f" after trimming {centreline_start_arc:.4f} units"
        )
        print(f"Planes placed along centreline: {len(plane_origins)}")
        print(f"Shell points sampled from face intersections: {len(shell_points)}")
        print(f"Centreline-normal surface curves generated: {len(surface_curves)}")
        print(
            f"First plane anchored at z={args.start_z:.4f}"
            f" on centreline arc distance {start_arc:.4f}"
        )
        print(
            "Centreline plane distances:"
            f" start={plane_distances.min():.4f},"
            f" end={plane_distances.max():.4f},"
            f" spacing~={args.plane_spacing:.4f}"
        )
        print(
            "Plane normal vectors computed for each plane:"
            f" {len(plane_normals)}"
        )
    else:
        branch_centrelines = build_tree_branch_centrelines(
            spheres,
            sphere_graph_k=args.tree_sphere_graph_k,
            spline_s=args.spline_s,
            centreline_samples=args.centreline_samples,
        )
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
        ) = slice_shell_points_from_branch_centrelines(
            mesh,
            branch_centrelines,
            plane_spacing=args.plane_spacing,
            shell_point_spacing=args.shell_point_spacing,
            start_z=args.start_z,
            start_z_tolerance=CONFIG["start_z_tolerance"],
        )
        centrelines_for_plot = [
            (meta["branch_id"], meta["centreline"])
            for meta in branch_metadata
        ]
        surface_curves = generate_nurbs_like_surface_intersections(
            centrelines_for_plot,
            plane_origins,
            plane_normals,
            shell_points,
            plane_index,
            num_surfaces=CONFIG["nurbs_surface_count"],
            angle_tolerance_deg=CONFIG["nurbs_angle_tolerance_deg"],
            min_points_per_surface=CONFIG["nurbs_min_points"],
            branch_index=branch_index,
            plane_branch_index=plane_branch_index,
        )

        print(f"Line method: {effective_line_method}")
        print(f"Tree branches used: {len(branch_metadata)}")
        print(f"Planes placed across branches: {len(plane_origins)}")
        print(f"Shell points sampled from branch face intersections: {len(shell_points)}")
        print(f"Centreline-normal surface curves generated: {len(surface_curves)}")
        print(
            "Branch plane distances:"
            f" min={plane_distances.min():.4f},"
            f" max={plane_distances.max():.4f},"
            f" spacing~={args.plane_spacing:.4f}"
        )
        print(
            "Plane normal vectors computed for all branch planes:"
            f" {len(plane_normals)}"
        )
        for meta in branch_metadata:
            print(
                f"Branch {meta['branch_id']}: "
                f"original_length={meta['original_length']:.4f}, "
                f"trimmed_length={meta['trimmed_length']:.4f}, "
                f"trimmed_by={meta['start_arc_before_trim']:.4f}, "
                f"planes={meta['plane_count']}, "
                f"anchor_to_z0={meta['anchor_to_start_z']}"
            )

    if args.csv:
        output_path = Path(args.csv)
        export_shell_points_csv(
            output_path,
            shell_points,
            plane_index,
            plane_distances,
            branch_index=branch_index,
        )
        print(f"Saved shell point table to {output_path}")

    if args.json:
        output_path = Path(args.json)
        export_shell_layers_json(
            output_path,
            shell_points,
            plane_index,
            plane_origins,
            plane_normals,
            plane_distances,
            plane_fragment_index=plane_fragment_index,
            branch_index=branch_index,
            plane_branch_index=plane_branch_index,
        )
        print(f"Saved layer JSON to {output_path}")

    if not args.no_plot:
        plot_shell_points(
            mesh,
            centrelines_for_plot,
            shell_points,
            plane_origins,
            spheres=spheres,
            branch_index=branch_index,
        )
        plot_surface_curves(
            mesh,
            centrelines_for_plot,
            surface_curves,
            spheres=spheres,
        )


if __name__ == "__main__":
    main_branch_aware()
