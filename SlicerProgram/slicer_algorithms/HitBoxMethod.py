import trimesh
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
from scipy.spatial import KDTree
from tqdm import tqdm as _tqdm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH_DIR = PACKAGE_ROOT / "mesh"

PARAMS = {
    "mesh_path": str(DEFAULT_MESH_DIR / "CustomEdgecaseThesis.stl"), # Mesh files are bundled with SlicerProgram/mesh.
    "point_graph_k": 4,
    "sphere_min_diameter": 0.0,
    "sphere_max_diameter": 0.0,
    "sphere_overlap_factor": 1,
    "sphere_step": 0,  # Legacy node-stride override. Use 0 for adaptive arc-length sampling.
    "sphere_neighborhood_scale": 0.05,
    "sphere_path_sample_spacing_factor": 0.5,
    "sphere_radius_percentile": 20,
    "sphere_min_pts": 1,
    "component_sphere_overlap_factor": 0.8,
    # "sphere_graph_k": 4,
    # "progressive_sphere_graph_k": 3,
    "tree_sphere_graph_k": 4,
    "sphere_skeleton_strategy": "complete_mst", # options: "knn", "complete_mst", "tree"
}

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


def _supports_progress_stream(stream) -> bool:
    if stream is None or not hasattr(stream, "write"):
        return False
    try:
        isatty = getattr(stream, "isatty", None)
        if callable(isatty) and not isatty():
            return False
    except Exception:
        return False
    try:
        flush = getattr(stream, "flush", None)
        if callable(flush):
            flush()
    except OSError:
        return False
    except Exception:
        return False
    return True


def tqdm(*args, **kwargs):
    stream = kwargs.get("file", sys.stderr)
    if "disable" not in kwargs:
        kwargs["disable"] = not _supports_progress_stream(stream)
    try:
        return _tqdm(*args, **kwargs)
    except OSError:
        safe_kwargs = dict(kwargs)
        safe_kwargs["disable"] = True
        safe_kwargs.pop("file", None)
        return _tqdm(*args, **safe_kwargs)



# -----------------------------
# Load mesh
# -----------------------------
def load_mesh(path=None):
    path = PARAMS["mesh_path"] if path is None else path
    return trimesh.load(path)

# -----------------------------
# Build k-NN graph (skeleton proxy)
# -----------------------------

import numpy as np
import matplotlib.pyplot as plt

def build_knn_graph(points, k=None, cancel_callback=None):
    tree = KDTree(points)
    G = nx.Graph()
    k = PARAMS["point_graph_k"] if k is None else k
    k = max(2, min(int(k), len(points)))

    for i in tqdm(range(len(points)), desc="Building k-NN graph"):
        if cancel_callback is not None and i % 100 == 0:
            cancel_callback()
        _, idx = tree.query(points[i], k=k)
        idx = np.atleast_1d(idx)
        for j in idx[1:]:
            G.add_edge(i, j)
    print("DEBUG: k-NN graph built with k =", k)
    print("DEBUG: KDTree structure:", tree)
    return G


# -----------------------------
# Find endpoints (branch-aware)
# -----------------------------
def find_endpoints(G, points, max_endpoints, degree_percentile):
    degrees = np.array([G.degree[n] for n in G.nodes])
    thr = np.percentile(degrees, degree_percentile)

    candidates = np.array(
        [n for n in G.nodes if G.degree[n] <= thr],
        dtype=int
    )

    print("DEBUG: low-degree candidates =", len(candidates))

    if candidates.size == 0:
        return []

    pts = points[candidates]
    centroid = pts.mean(axis=0)
    d = np.linalg.norm(pts - centroid, axis=1)

    keep = min(max_endpoints, candidates.size)
    pick = np.argpartition(d, -keep)[-keep:]

    endpoints = candidates[pick].tolist()
    print("DEBUG: endpoints after capping =", len(endpoints))

    return endpoints




def filter_close_spheres(
    spheres,
    overlap_factor
):
    """
    Remove spheres whose centres are too close relative to their radii.

    Parameters
    ----------
    spheres : list[(center, radius)]
    overlap_factor : float
        Fraction of (r_i + r_j) below which spheres are considered overlapping

    Returns
    -------
    filtered_spheres : list[(center, radius)]
    """

    if len(spheres) == 0:
        return spheres

    centres = np.array([c for c, _ in spheres])
    radii   = np.array([r for _, r in spheres])

    tree = KDTree(centres)
    keep = np.ones(len(spheres), dtype=bool)

    for i, (c, r) in enumerate(spheres):
        if not keep[i]:
            continue

        # conservative search radius
        search_radius = overlap_factor * (r + radii.max())
        neighbours = tree.query_ball_point(c, search_radius)

        for j in neighbours:
            if i == j or not keep[j]:
                continue

            d = np.linalg.norm(centres[i] - centres[j])
            min_dist = overlap_factor * (radii[i] + radii[j])

            if d < min_dist:
                # deterministic tie-breaking: keep larger sphere
                if radii[i] >= radii[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [s for s, k in zip(spheres, keep) if k]


def _normalize_diameter_bounds(min_diameter=None, max_diameter=None):
    min_diameter = (
        PARAMS["sphere_min_diameter"]
        if min_diameter is None
        else min_diameter
    )
    max_diameter = (
        PARAMS["sphere_max_diameter"]
        if max_diameter is None
        else max_diameter
    )
    min_diameter = float(max(min_diameter, 0.0))
    max_diameter = float(max(max_diameter, 0.0))
    if max_diameter > 0.0 and max_diameter < min_diameter:
        min_diameter, max_diameter = max_diameter, min_diameter
    return min_diameter, max_diameter


def filter_spheres_by_diameter(spheres, min_diameter=None, max_diameter=None):
    min_diameter, max_diameter = _normalize_diameter_bounds(
        min_diameter=min_diameter,
        max_diameter=max_diameter,
    )
    if len(spheres) == 0:
        return spheres

    filtered = []
    for center, radius in spheres:
        diameter = 2.0 * float(radius)
        if diameter + 1e-9 < min_diameter:
            continue
        if max_diameter > 0.0 and diameter - 1e-9 > max_diameter:
            continue
        filtered.append((center, radius))

    return filtered


def generate_component_centroid_spheres(
    points,
    G,
    overlap_factor=None,
    min_diameter=None,
    max_diameter=None,
    cancel_callback=None,
):
    overlap_factor = (
        PARAMS["component_sphere_overlap_factor"]
        if overlap_factor is None
        else overlap_factor
    )
    spheres = []
    print("Generating component-centroid spheres from connected components...")
    components = list(nx.connected_components(G))
    for idx, comp in enumerate(tqdm(components, desc="Component spheres")):
        if cancel_callback is not None and idx % 50 == 0:
            cancel_callback()
        pts = points[list(comp)]
        center = pts.mean(axis=0)
        radius = np.percentile(
            np.linalg.norm(pts - center, axis=1),
            40
        )
        spheres.append((center, radius))

    spheres = filter_spheres_by_diameter(
        spheres,
        min_diameter=min_diameter,
        max_diameter=max_diameter,
    )
    spheres = filter_close_spheres(
        spheres,
        overlap_factor
    )
    
    return spheres


def _path_length(points, path):
    if len(path) < 2:
        return 0.0
    path_points = points[np.asarray(path, dtype=int)]
    return float(np.linalg.norm(np.diff(path_points, axis=0), axis=1).sum())


def _longest_tree_path(tree, points):
    endpoints = [node for node in tree.nodes if tree.degree[node] <= 1]
    if len(endpoints) < 2:
        return list(tree.nodes)

    best_path = None
    best_length = -np.inf

    for idx, start in enumerate(endpoints):
        lengths = nx.single_source_dijkstra_path_length(tree, start, weight="weight")
        for end in endpoints[idx + 1:]:
            length = lengths.get(end)
            if length is None or length <= best_length:
                continue
            best_length = length
            best_path = nx.shortest_path(tree, start, end, weight="weight")

    return best_path if best_path is not None else list(tree.nodes)


def _sample_path_nodes_by_stride(path, step):
    step = max(1, int(step))
    sample_nodes = list(path[::step])
    if path and path[-1] not in sample_nodes:
        sample_nodes.append(path[-1])
    return sample_nodes


def _sample_path_nodes_by_spacing(points, path, spacing):
    if len(path) <= 2:
        return list(path)

    spacing = float(max(spacing, 1e-9))
    path = list(path)
    path_points = points[np.asarray(path, dtype=int)]
    segment_lengths = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])

    if total_length <= spacing:
        return [path[0], path[-1]]

    target_distances = np.arange(0.0, total_length, spacing)
    sample_indices = np.searchsorted(cumulative, target_distances, side="left")

    sample_nodes = []
    for sample_idx in sample_indices:
        sample_idx = min(int(sample_idx), len(path) - 1)
        node = path[sample_idx]
        if not sample_nodes or sample_nodes[-1] != node:
            sample_nodes.append(node)

    if sample_nodes[-1] != path[-1]:
        sample_nodes.append(path[-1])

    return sample_nodes


def generate_spheres(
    points,
    G,
    mesh_bounds,
    overlap_factor=None,
    min_diameter=None,
    max_diameter=None,
    step=None,
    neighborhood_scale=None,
    path_sample_spacing_factor=None,
    percentile=None,
    min_pts=None,
    cancel_callback=None,
):
    """
    Generate sphere samples along graph skeleton paths.
    """
    points = np.asarray(points, dtype=float)
    spheres = []
    overlap_factor = PARAMS["sphere_overlap_factor"] if overlap_factor is None else overlap_factor
    step = PARAMS["sphere_step"] if step is None else step
    neighborhood_scale = (
        PARAMS["sphere_neighborhood_scale"]
        if neighborhood_scale is None
        else neighborhood_scale
    )
    path_sample_spacing_factor = (
        PARAMS["sphere_path_sample_spacing_factor"]
        if path_sample_spacing_factor is None
        else path_sample_spacing_factor
    )
    percentile = PARAMS["sphere_radius_percentile"] if percentile is None else percentile
    min_pts = PARAMS["sphere_min_pts"] if min_pts is None else min_pts

    if len(points) == 0 or G.number_of_nodes() == 0:
        return spheres

    skeleton = build_skeleton_tree(G, points)
    paths = extract_tree_paths(skeleton)
    if not paths:
        paths = [_longest_tree_path(skeleton, points)]

    bounds = np.asarray(mesh_bounds, dtype=float)
    scale = float(np.linalg.norm(bounds[1] - bounds[0]))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        return spheres

    tree = KDTree(points)
    neighborhood_radius = max(neighborhood_scale * scale, 1e-9)
    sample_spacing = max(
        float(path_sample_spacing_factor) * neighborhood_radius,
        1e-9,
    )
    legacy_stride = 0 if step is None else max(0, int(step))

    print("Generating spheres from skeleton paths...")
    for path_idx, path in enumerate(tqdm(paths, desc="Generating spheres (paths)")):
        if cancel_callback is not None and path_idx % 10 == 0:
            cancel_callback()
        if len(path) == 0:
            continue

        if legacy_stride > 0:
            sample_nodes = _sample_path_nodes_by_stride(path, legacy_stride)
        else:
            sample_nodes = _sample_path_nodes_by_spacing(
                points,
                path,
                sample_spacing,
            )

        for node_idx, node in enumerate(sample_nodes):
            if cancel_callback is not None and node_idx % 50 == 0:
                cancel_callback()
            center_seed = points[int(node)]
            local_idx = tree.query_ball_point(center_seed, neighborhood_radius)

            if len(local_idx) < min_pts:
                query_k = min(max(min_pts, 2), len(points))
                _, nearest_idx = tree.query(center_seed, k=query_k)
                local_idx = np.atleast_1d(nearest_idx).astype(int).tolist()

            local = points[np.asarray(local_idx, dtype=int)]
            if len(local) < 2:
                continue

            center = local.mean(axis=0)
            distances = np.linalg.norm(local - center, axis=1)
            radius = float(np.percentile(distances, percentile))

            if np.isfinite(radius) and radius > 0:
                spheres.append((center, radius))

    spheres = filter_spheres_by_diameter(
        spheres,
        min_diameter=min_diameter,
        max_diameter=max_diameter,
    )
    spheres = filter_close_spheres(spheres, overlap_factor)
    return spheres


# -----------------------------
# Visualization
# -----------------------------
def plot_result(mesh, spheres):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.add_collection3d(
        Poly3DCollection(
            mesh.triangles,
            facecolor="cyan",
            edgecolor="none",
            alpha=0.15
        )
    )

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    for c, r in spheres:
        x = c[0] + r * np.cos(u) * np.sin(v)
        y = c[1] + r * np.sin(u) * np.sin(v)
        z = c[2] + r * np.cos(v)
        ax.plot_wireframe(x, y, z, color="red", alpha=0.4)

    ax.set_box_aspect([1, 1, 1])
    # Ortihographic projection for better visualization
    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass  # set_proj_type is available in matplotlib >=3.2
    # View angle
    ax.view_init(elev=0, azim=90)
    ax.set_title("Centre-line–aware hitbox spheres")
    plt.show()

import networkx as nx
import numpy as np


def build_skeleton_tree(G, points):
    T = nx.Graph()

    for u, v in G.edges:
        w = np.linalg.norm(points[u] - points[v])
        T.add_edge(u, v, weight=w)

    mst = nx.minimum_spanning_tree(T)
    return mst


# def build_sphere_graph(spheres, k=3):
#     centres = np.array([c for c, _ in spheres])

#     G = nx.Graph()

#     if len(centres) == 0:
#         return G, centres

#     for i in range(len(centres)):
#         G.add_node(i)

#     if len(centres) == 1:
#         return G, centres

#     tree = KDTree(centres)
#     k = min(max(2, int(k)), len(centres))

#     for i in tqdm(range(len(centres)), desc="Building sphere adjacency graph", unit="node"):
#         _, idx = tree.query(centres[i], k=k)
#         idx = np.atleast_1d(idx)
#         for j in idx[1:]:
#             if np.isfinite(j):
#                 G.add_edge(i, int(j))

#     return G, centres


def build_sphere_graph(spheres, k=None):
    """
    Build adjacency graph between sphere centres.
    """

    centres = np.array([c for c, _ in spheres])
    G = nx.Graph()

    if len(centres) == 0:
        return G, centres

    for i in range(len(centres)):
        G.add_node(i)

    if len(centres) == 1:
        return G, centres

    tree = KDTree(centres)
    k = PARAMS["sphere_graph_k"] if k is None else k
    k = max(2, min(int(k), len(centres)))

    for i in tqdm(
        range(len(centres)),
        desc="Building sphere adjacency graph",
        unit="node"
    ):
        _, idx = tree.query(centres[i], k=k)
        idx = np.atleast_1d(idx)
        for j in idx[1:]:
            G.add_edge(i, j)

    return G, centres


def build_complete_sphere_graph(spheres):
    """
    Build a complete weighted graph between sphere centres.
    """
    centres = np.array([c for c, _ in spheres])
    G = nx.Graph()

    for i in range(len(centres)):
        G.add_node(i)

    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            weight = float(np.linalg.norm(centres[i] - centres[j]))
            G.add_edge(i, j, weight=weight)

    return G, centres


def build_sphere_skeleton_graph(spheres, strategy=None, k=None):
    """
    Build a sphere skeleton graph according to a chosen strategy.

    Strategies:
    - mst: k-NN sphere graph followed by MST
    - knn: raw k-NN sphere graph
    - complete_mst: complete weighted graph followed by MST
    """
    strategy = PARAMS["sphere_skeleton_strategy"] if strategy is None else strategy
    strategy = str(strategy).lower()

    if strategy == "knn":
        graph, centres = build_sphere_graph(spheres, k=k)
        return graph, centres

    if strategy == "complete_mst":
        graph, centres = build_complete_sphere_graph(spheres)
        return nx.minimum_spanning_tree(graph, weight="weight"), centres

    graph, centres = build_sphere_graph(spheres, k=k)
    return build_skeleton_tree(graph, centres), centres

def extract_progressive_paths(G, points):
    """
    Progressive rooted traversal allowing branch remerging.
    """

    visited = set()
    paths = []

    root = np.argmin(points[:, 2])

    pbar = tqdm(
        total=len(points),
        desc="Progressive skeleton coverage",
        unit="node"
    )

    while len(visited) < len(points):

        # ----------------------------------
        # Choose start node
        # ----------------------------------
        if not visited:
            start = root
        else:
            unvisited = list(set(G.nodes) - visited)

            lowest_unvisited = min(
                unvisited,
                key=lambda i: points[i][2]
            )

            visited_pts = points[list(visited)]
            d = np.linalg.norm(
                visited_pts - points[lowest_unvisited],
                axis=1
            )

            start = list(visited)[np.argmin(d)]

        # ----------------------------------
        # Grow path
        # ----------------------------------
        endpoints = [n for n in G.nodes if G.degree[n] == 1]

        best_path = None
        best_len = -np.inf

        for ep in endpoints:
            try:
                path = nx.shortest_path(G, start, ep)
            except nx.NetworkXNoPath:
                continue

            if any(n in visited for n in path[1:]):
                continue

            L = sum(
                np.linalg.norm(points[a] - points[b])
                for a, b in zip(path[:-1], path[1:])
            )

            if L > best_len:
                best_len = L
                best_path = path

        if best_path is None:
            if start not in visited:
                visited.add(start)
                pbar.update(1)
            continue

        clean_path = [best_path[0]]

        for n in best_path[1:]:
            if n in visited:
                break
            clean_path.append(n)

        for n in clean_path:
            if n not in visited:
                visited.add(n)
                pbar.update(1)

        paths.append(clean_path)

    pbar.close()

    return paths

def plot_progressive_centrelines(spheres, mesh):


    Gs, centres = build_sphere_graph(
        spheres,
        k=PARAMS["progressive_sphere_graph_k"]
    )
    paths = extract_progressive_paths(Gs, centres)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.add_collection3d(
        Poly3DCollection(
            mesh.triangles,
            facecolor="cyan",
            edgecolor="none",
            alpha=0.15
        )
    )

    for path in tqdm(
        paths,
        desc="Spline fitting",
        unit="branch"
    ):

        pts = centres[path]

        if len(pts) < 4:
            ax.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=2)
            continue

        spline, _ = make_splprep(pts.T, s=2)
        u = np.linspace(0, 1, 100)
        curve = spline(u).T

        ax.plot(
            curve[:,0],
            curve[:,1],
            curve[:,2],
            linewidth=3
        )

    ax.set_box_aspect([1,1,1])

    try:
        ax.set_proj_type('ortho')
    except:
        pass

    ax.view_init(elev=0, azim=90)
    ax.set_title("Progressive centre-line extraction")

    plt.show()

def extract_tree_paths(T):
    """
    Extract branch paths from a tree graph.
    """
    paths = []
    visited_edges = set()

    for node in T.nodes:
        if T.degree[node] != 2:  # endpoint or junction
            for nbr in T.neighbors(node):
                edge = tuple(sorted((node, nbr)))
                if edge in visited_edges:
                    continue

                path = [node, nbr]
                prev, curr = node, nbr

                while T.degree[curr] == 2:
                    nxt = [n for n in T.neighbors(curr) if n != prev][0]
                    edge = tuple(sorted((curr, nxt)))
                    if edge in visited_edges:
                        break
                    path.append(nxt)
                    prev, curr = curr, nxt

                paths.append(path)

                for a, b in zip(path[:-1], path[1:]):
                    visited_edges.add(tuple(sorted((a, b))))

    return paths


def plot_tree_centrelines(spheres, mesh):
    # --- build sphere graph
    Gs, centres = build_sphere_graph(spheres, k=PARAMS["tree_sphere_graph_k"])

    # --- build skeleton tree
    T = build_skeleton_tree(Gs, centres)

    # --- extract paths
    paths = extract_tree_paths(T)

    # --- plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # mesh
    ax.add_collection3d(
        Poly3DCollection(
            mesh.triangles,
            facecolor="cyan",
            edgecolor="none",
            alpha=0.15
        )
    )

    # plot splines per branch
    for path in paths:

        pts = centres[path]

        if len(pts) < 4:
            ax.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=2)
            continue

        spline, _ = make_splprep(pts.T, s=2)
        u = np.linspace(0, 1, 100)
        curve = spline(u).T

        ax.plot(
            curve[:,0],
            curve[:,1],
            curve[:,2],
            linewidth=3
        )

    ax.set_box_aspect([1,1,1])

    try:
        ax.set_proj_type('ortho')
    except:
        pass

    ax.view_init(elev=0, azim=90)
    ax.set_title("Tree-based centre-line skeleton")

    plt.show()


def plot_centrelines(spheres, mesh):
    centres = np.array([c for c, _ in spheres])

    if len(centres) == 0:
        print("No spheres to plot.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if len(centres) == 1:
        ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2], color="blue", s=40)
    elif len(centres) < 4:
        ax.plot(centres[:, 0], centres[:, 1], centres[:, 2], color="blue", linewidth=2)
    else:
        spline, _ = make_splprep(centres.T, s=2)
        u_new = np.linspace(0.0, 1.0, 100)
        centreline = spline(u_new).T
        ax.plot(centreline[:, 0], centreline[:, 1], centreline[:, 2], color="blue", linewidth=2)

    ax.add_collection3d(
        Poly3DCollection(mesh.triangles, facecolor="cyan", edgecolor="none", alpha=0.15)
    )
    ax.set_box_aspect([1, 1, 1])

    try:
        ax.set_proj_type('ortho')
    except Exception:
        pass

    ax.view_init(elev=0, azim=90)
    ax.set_title("Centre-lines from hitbox spheres")
    plt.show()

def main():
    mesh = load_mesh()

    # CustomEdgeCaseThesis
    # EdgeCaseSplitJoin
    # EdgeCaseSplit
    points = mesh.vertices
    G = build_knn_graph(points)
    print("DEBUG: number of nodes =", G.number_of_nodes())
    print("DEBUG: number of edges =", G.number_of_edges())
    
    print("Watertight:", mesh.is_watertight)
    # G = nx.minimum_spanning_tree(G)

    spheres = []
    # spheres = generate_spheres(
    #     points,
    #     G,
    #     mesh.bounds,
    # )
    if not spheres:
        print("No spheres from skeleton paths, using component-centroid spheres")
        spheres = generate_component_centroid_spheres(points, G)

    # print("Centres and radii of generated spheres:", spheres)
    print(f"Generated {len(spheres)} spheres")


    plot_result(mesh, spheres)
    # plot_centrelines(spheres, mesh)
    plot_tree_centrelines(spheres, mesh)
    # plot_progressive_centrelines(spheres, mesh)


if __name__ == "__main__":
    main()
