import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from collections import defaultdict


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _union_find_groups(faces, u_edges):
    parent = list(range(len(faces)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    edge_to_faces = {}
    for fi, face in enumerate(faces):
        n = len(face)
        for k in range(n):
            e = tuple(sorted((face[k], face[(k + 1) % n])))
            edge_to_faces.setdefault(e, []).append(fi)

    for e in u_edges:
        if e in edge_to_faces and len(edge_to_faces[e]) == 2:
            union(edge_to_faces[e][0], edge_to_faces[e][1])

    groups = defaultdict(list)
    for fi in range(len(faces)):
        groups[find(fi)].append(fi)

    return list(groups.values())


def _panel_outline(face_group, faces, vertices, u_edges):
    edge_count = defaultdict(int)
    for fi in face_group:
        face = faces[fi]
        n = len(face)
        for k in range(n):
            e = tuple(sorted((face[k], face[(k + 1) % n])))
            edge_count[e] += 1

    outer_edges = [e for e, cnt in edge_count.items()
                   if cnt == 1 and e not in u_edges]

    if not outer_edges:
        return None

    adj = defaultdict(list)
    for a, b in outer_edges:
        adj[a].append(b)
        adj[b].append(a)

    start = outer_edges[0][0]
    ordered = [start]
    prev = None
    current = start
    for _ in range(len(outer_edges)):
        neighbors = adj[current]
        next_v = next((nb for nb in neighbors if nb != prev), None)
        if next_v is None or next_v == start:
            break
        ordered.append(next_v)
        prev = current
        current = next_v

    return vertices[ordered]


def _panel_centroid(face_group, faces, vertices):
    all_verts = []
    for fi in face_group:
        for vi in faces[fi]:
            all_verts.append(vertices[vi])
    return np.mean(all_verts, axis=0)


def _panel_local_axes(face_group, faces, vertices):
    edge_count = defaultdict(int)
    for fi in face_group:
        face = faces[fi]
        n = len(face)
        for k in range(n):
            e = tuple(sorted((face[k], face[(k + 1) % n])))
            edge_count[e] += 1

    outer_edges = [e for e, cnt in edge_count.items() if cnt == 1]

    if not outer_edges:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])

    best_len = -1
    best_vec = np.array([1.0, 0.0])
    for a, b in outer_edges:
        v = vertices[b] - vertices[a]
        length = np.linalg.norm(v)
        if length > best_len:
            best_len = length
            best_vec = v / length

    x_hat = best_vec
    y_hat = np.array([-x_hat[1], x_hat[0]])
    return x_hat, y_hat


# ----------------------------------------------------------------------
# Main plotting function
# ----------------------------------------------------------------------

def plot_decenter_heatmap_datapoints(
    fold_path,
    panel_map=None,
    panel_data=None,
    decenter_data=None,
    point_labels=None,
    cmap_name="coolwarm",
    title=None,
    colorbar_label="RSS Tip/Tilt [°]",
    vmin=None,
    vmax=None,
    save_path=None,
    scale_factor=1.0,
    use_local_frame=False,
    origin_dot_size=30,
    origin_dot_color="white",
    data_dot_size=50,
    data_dot_marker="o",
    show_crosshair=True,
    crosshair_size=0.04,
    crosshair_color="white",
    crosshair_lw=0.8,
):
    """
    Plots a flasher .fold file as a heat map with optional de-center scatter
    point overlays. Supports multiple points per panel (e.g. one per deployment).

    Parameters
    ----------
    fold_path : str
        Path to the .fold file.
    panel_data : array-like or None
        One scalar per physical panel (26 for this flasher). If None, random
        values are used for demonstration.
    decenter_data : dict or None
        Maps panel_idx -> list of (dx, dy) tuples, one per series (deployment).
        A single tuple is also accepted and treated as a one-element list.

        Example (4 deployments on every panel):
            decenter_data = {
                0: [(0.01, 0.02), (0.03, -0.01), (0.00, 0.04), (0.02, 0.01)],
                1: [(0.05, 0.02), (0.04, -0.03), (0.06, 0.01), (0.05, 0.00)],
                ...
            }

        Pass None to skip the overlay entirely.
        Pass {} to draw only the origin crosshairs with no data points.

    point_labels : list of str or None
        Labels for each series in the legend, e.g.:
            ["Deployment 1", "Deployment 2", "Deployment 3", "Deployment 4"]
        If None, series are labeled "Series 1", "Series 2", etc.

    scale_factor : float
        Multiplier applied to (dx, dy) before plotting. Tune so offsets are
        visible relative to panel size.

    use_local_frame : bool
        If True, (dx, dy) are in each panel's local frame (X along longest
        outer edge, Y perpendicular). If False (default), global plot frame.

    origin_dot_size, origin_dot_color : appearance of the (0,0) reference dot.
    data_dot_size, data_dot_marker    : appearance of data points.
    show_crosshair, crosshair_size, crosshair_color, crosshair_lw : crosshair.
    cmap_name, title, colorbar_label, vmin, vmax, save_path : standard options.
    """

    # Wong (2011) colorblind-safe palette — standard in many journals
    SERIES_COLORS = [
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#F0E442",  # yellow
    ]

    # ------------------------------------------------------------------
    # 1. Load .fold file
    # ------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fold_path_full = os.path.join(script_dir, fold_path)
    with open(fold_path_full, "r") as f:
        fold = json.load(f)

    vertices    = np.array(fold["vertices_coords"])
    edges_verts = fold["edges_vertices"]
    assignments = fold["edges_assignment"]
    faces       = fold["faces_vertices"]

    rotation_deg = -20
    angle = np.radians(rotation_deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
    vertices = vertices @ rot.T

    # ------------------------------------------------------------------
    # 2. Identify U/F edges
    # ------------------------------------------------------------------
    u_edges = {
        tuple(sorted(e))
        for e, a in zip(edges_verts, assignments)
        if a in ("U", "F")
    }

    # ------------------------------------------------------------------
    # 3. Group triangulated faces -> physical panels
    # ------------------------------------------------------------------
    panel_groups = _union_find_groups(faces, u_edges)
    n_panels = len(panel_groups)

    # ------------------------------------------------------------------
    # 4. Panel heat-map data
    # ------------------------------------------------------------------
    if panel_data is None:
        rng = np.random.default_rng(42)
        panel_data = rng.uniform(0, 3.0, size=n_panels)
        print(f"No panel_data supplied — using {n_panels} random values.")
    else:
        # Reorder by panel_map BEFORE stripping the index
        if panel_map is not None and hasattr(panel_data, 'index'):
            panel_data = np.array([
                float(panel_data.loc[panel_map[i]]) if panel_map[i] in panel_data.index else 0.0
                for i in range(n_panels)
            ])
        else:
            panel_data = np.asarray(panel_data, dtype=float)

        if len(panel_data) != n_panels:
            raise ValueError(
                f"panel_data length ({len(panel_data)}) must match "
                f"number of physical panels ({n_panels})."
            )
    # ------------------------------------------------------------------
    # 5. Colormap
    # ------------------------------------------------------------------
    if vmin is None:
        vmin = panel_data.min()
    if vmax is None:
        vmax = panel_data.max()

    cmap  = plt.get_cmap(cmap_name)
    cnorm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ------------------------------------------------------------------
    # 6. Plot filled panels
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))

    for panel_idx, face_group in enumerate(panel_groups):
        outline = _panel_outline(face_group, faces, vertices, u_edges)
        if outline is None:
            continue
        fc = cmap(cnorm(panel_data[panel_idx]))
        ax.add_patch(Polygon(outline, closed=True,
                             facecolor=fc, edgecolor="none", zorder=2))
        

    # ------------------------------------------------------------------
    # 7. Coordinate frame arrow on center pentagon
    # ------------------------------------------------------------------
    center_group = panel_groups[25]
    centroid_center = _panel_centroid(center_group, faces, vertices)
    arrow_length = 0.3
    arrow_kwargs = dict(head_width=0.06, head_length=0.04,
                        length_includes_head=True, zorder=6)
    ax.arrow(centroid_center[0], centroid_center[1],
             arrow_length, 0, fc="black", ec="black", **arrow_kwargs)
    ax.arrow(centroid_center[0], centroid_center[1],
             0, arrow_length, fc="black", ec="black", **arrow_kwargs)
    ax.text(centroid_center[0] + arrow_length * 1.2, centroid_center[1],
            "X", color="black", fontsize=10, fontweight="bold",
            zorder=6, ha="center")
    ax.text(centroid_center[0], centroid_center[1] + arrow_length * 1.2,
            "Y", color="black", fontsize=10, fontweight="bold",
            zorder=6, ha="center")

    # ------------------------------------------------------------------
    # 8. De-center overlay
    # ------------------------------------------------------------------
    n_series = 0

    if decenter_data is not None:
        # Determine number of series
        for v in decenter_data.values():
            pts = [v] if isinstance(v, tuple) else list(v)
            n_series = max(n_series, len(pts))

        # Resolve labels
        if point_labels is None:
            point_labels = [f"Series {i+1}" for i in range(n_series)]
        else:
            point_labels = list(point_labels)
            while len(point_labels) < n_series:
                point_labels.append(f"Series {len(point_labels)+1}")

        for panel_idx, face_group in enumerate(panel_groups):
            centroid = _panel_centroid(face_group, faces, vertices)

            # Crosshair
            if show_crosshair:
                cs = crosshair_size
                ax.plot([centroid[0] - cs, centroid[0] + cs],
                        [centroid[1],       centroid[1]],
                        color=crosshair_color, lw=crosshair_lw, zorder=5)
                ax.plot([centroid[0],       centroid[0]],
                        [centroid[1] - cs,  centroid[1] + cs],
                        color=crosshair_color, lw=crosshair_lw, zorder=5)

            # Origin dot
            ax.scatter(centroid[0], centroid[1],
                       s=origin_dot_size, c=origin_dot_color,
                       zorder=7, marker="o", edgecolors="gray", linewidths=0.5)

            # Data points
            if panel_idx in decenter_data:
                pts = decenter_data[panel_idx]
                if isinstance(pts, tuple):
                    pts = [pts]

                for i, (dx, dy) in enumerate(pts):
                    color = SERIES_COLORS[i % len(SERIES_COLORS)]

                    if use_local_frame:
                        x_hat, y_hat = _panel_local_axes(face_group, faces, vertices)
                        offset = (dx * x_hat + dy * y_hat) * scale_factor
                    else:
                        offset = np.array([dx, dy]) * scale_factor

                    data_pos = centroid + offset

                    ax.scatter(data_pos[0], data_pos[1],
                               s=data_dot_size, c=color,
                               marker=data_dot_marker, zorder=8,
                               edgecolors="white", linewidths=0.5)

                    ax.plot([centroid[0], data_pos[0]],
                            [centroid[1], data_pos[1]],
                            color=color, lw=0.7, zorder=6, alpha=0.5)

    # ------------------------------------------------------------------
    # 9. Colorbar
    # ------------------------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=18, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    # ------------------------------------------------------------------
    # 10. Legend for de-center series
    # ------------------------------------------------------------------
    if decenter_data is not None and n_series > 0:
        legend_elements = []

        for i in range(n_series):
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            legend_elements.append(
                Line2D([0], [0], marker=data_dot_marker, color="none",
                       markerfacecolor=color, markeredgecolor="white",
                       markeredgewidth=0.5, markersize=7,
                       label=point_labels[i])
            )

        # Origin reference
        legend_elements.append(
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=origin_dot_color,
                   markeredgecolor="gray", markeredgewidth=0.5,
                   markersize=6, label="Ideal position (0, 0)")
        )

        ax.legend(
            handles=legend_elements,
            loc="lower right",
            fontsize=9,
            framealpha=0.9,
            edgecolor="0.7",
            title="De-center overlay",
            title_fontsize=9,
        )

    # ------------------------------------------------------------------
    # 11. Formatting
    # ------------------------------------------------------------------
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle=":", alpha=0.3, zorder=0)
    ax.autoscale_view()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved to {save_path}")

    plt.show()

    return fig, ax


# ----------------------------------------------------------------------
# Helper: panel centroids
# ----------------------------------------------------------------------

def get_panel_centroids(fold_path):
    """Returns {panel_idx: (cx, cy)} for all physical panels."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fold_path_full = os.path.join(script_dir, fold_path)
    with open(fold_path_full, "r") as f:
        fold = json.load(f)

    vertices    = np.array(fold["vertices_coords"])
    edges_verts = fold["edges_vertices"]
    assignments = fold["edges_assignment"]
    faces       = fold["faces_vertices"]

    rotation_deg = -20
    angle = np.radians(rotation_deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
    vertices = vertices @ rot.T

    u_edges = {
        tuple(sorted(e))
        for e, a in zip(edges_verts, assignments)
        if a in ("U", "F")
    }
    panel_groups = _union_find_groups(faces, u_edges)

    return {
        i: tuple(_panel_centroid(g, faces, vertices))
        for i, g in enumerate(panel_groups)
    }


def build_decenter_dict(panel_map, *deployment_pairs):
    """
    Build a decenter_data dict for plot_flasher_heatmap().

    Parameters
    ----------
    panel_map : dict
        Maps panel_idx (int) -> aperture label (str), e.g. {0: "Aper1", ...}
    *deployment_pairs : tuples of (x_df, y_df, x_col, y_col)
        One tuple per deployment. Each tuple contains:
            x_df  : DataFrame indexed by aperture label with x decenter column
            y_df  : DataFrame indexed by aperture label with y decenter column
            x_col : str, column name in x_df
            y_col : str, column name in y_df

    Returns
    -------
    dict : {panel_idx: [(dx1, dy1), (dx2, dy2), ...]}
        One (dx, dy) tuple per deployment for each panel.

    Example
    -------
        decenter_data = build_decenter_dict(
            panel_map,
            (x_decenter_deployment1, y_decenter_deployment1,
             "x_decenter_deployment1", "y_decenter_deployment1"),
            (x_decenter_deployment2, y_decenter_deployment2,
             "x_decenter_deployment2", "y_decenter_deployment2"),
            (x_decenter_deployment3, y_decenter_deployment3,
             "x_decenter_deployment3", "y_decenter_deployment3"),
            (x_decenter_deployment4, y_decenter_deployment4,
             "x_decenter_deployment4", "y_decenter_deployment4"),
        )
    """
    decenter = {}
    for panel_idx in range(26):
        label = panel_map[panel_idx]
        points = []
        for x_df, y_df, x_col, y_col in deployment_pairs:
            if label in x_df.index and label in y_df.index:
                dx = float(x_df.loc[label, x_col])
                dy = float(y_df.loc[label, y_col])
            else:
                dx, dy = 0.0, 0.0  # fallback if panel missing from data
            points.append((dx, dy))
        decenter[panel_idx] = points
    return decenter