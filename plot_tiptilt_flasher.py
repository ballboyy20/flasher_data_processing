
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from collections import defaultdict
import pandas as pd


def tip_tilt_to_euler(tip_deg, tilt_deg):
    """Rigorous angle between panel normal and center normal (0,0,1)."""
    tip  = np.radians(tip_deg)
    tilt = np.radians(tilt_deg)
    nx = np.sin(tilt)
    ny = np.sin(tip)
    nz = np.sqrt(max(0.0, 1 - nx**2 - ny**2))
    return np.degrees(np.arccos(np.clip(nz, -1.0, 1.0)))

def _find_shared_edges(panel_groups, faces, vertices, u_edges):
    """
    Returns a list of (panel_idx_A, panel_idx_B, edge_coords)
    for every pair of adjacent physical panels.
    edge_coords is a (2,2) array: [start_vertex, end_vertex]
    """
    # For each panel, collect its outer boundary edges
    panel_outer_edges = {}
    for panel_idx, face_group in enumerate(panel_groups):
        edge_count = defaultdict(int)
        for fi in face_group:
            face = faces[fi]
            n = len(face)
            for k in range(n):
                e = tuple(sorted((face[k], face[(k+1) % n])))
                edge_count[e] += 1
        # outer edges of this panel = edges appearing once, not U/F edges
        panel_outer_edges[panel_idx] = {
            e for e, cnt in edge_count.items()
            if cnt == 1 and e not in u_edges
        }

    # Find shared edges between panels
    shared = []
    panel_indices = list(panel_outer_edges.keys())
    for i in range(len(panel_indices)):
        for j in range(i+1, len(panel_indices)):
            common = panel_outer_edges[i] & panel_outer_edges[j]
            for e in common:
                coords = vertices[list(e)]
                shared.append((i, j, coords))
    return shared


# ----------------------------------------------------------------------
# Internal helpers (shared with plot_decenter_flasher.py)
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


def _find_shared_edges(panel_groups, faces, vertices, u_edges):
    """
    Find all shared boundary edges between adjacent physical panels.

    Returns
    -------
    list of (panel_idx_A, panel_idx_B, edge_coords)
        edge_coords : (2, 2) array — [start_xy, end_xy]
    """
    # Collect outer boundary edges per panel
    panel_outer_edges = {}
    for panel_idx, face_group in enumerate(panel_groups):
        edge_count = defaultdict(int)
        for fi in face_group:
            face = faces[fi]
            n = len(face)
            for k in range(n):
                e = tuple(sorted((face[k], face[(k + 1) % n])))
                edge_count[e] += 1
        panel_outer_edges[panel_idx] = {
            e for e, cnt in edge_count.items()
            if cnt == 1 and e not in u_edges
        }

    # Find edges shared between two panels
    shared = []
    n = len(panel_groups)
    for i in range(n):
        for j in range(i + 1, n):
            common = panel_outer_edges[i] & panel_outer_edges[j]
            for e in common:
                coords = vertices[list(e)]  # shape (2, 2)
                shared.append((i, j, coords))

    return shared


# ----------------------------------------------------------------------
# Main plotting function
# ----------------------------------------------------------------------

def plot_euler_angle_heatmap(
    fold_path,
    panel_map=None,
    mask_panels=None,
    panel_data=None,
    edge_data=None,
    panel_cmap_name="viridis",
    edge_cmap_name="hot_r",
    figsize=(10, 9),
    title=None,
    panel_colorbar_label="Mean Euler Angle [°]",
    edge_colorbar_label="Mean Relative Angle [°]",
    panel_vmin=None,
    panel_vmax=None,
    edge_vmin=None,
    edge_vmax=None,
    edge_linewidth=4.0,
    save_path=None,
):
    """
    Plots a flasher .fold file with:
      - Panel fills colored by a scalar (e.g. mean absolute euler angle from center)
      - Shared edges between adjacent panels colored by a scalar
        (e.g. mean relative euler angle between the two panels)

    Two separate colorbars are shown — one for panel fill, one for edges.

    Parameters
    ----------
    fold_path : str
        Path to the .fold file.
    panel_map : dict or None
        Maps panel_idx (int) -> aperture label (str).
        If provided and panel_data is a pandas Series, data is reordered
        to match panel_map order automatically.
    mask_panels : list of int or None
        Panel indices to show as gray (no data). e.g. [25] for center.
    panel_data : array-like or pandas Series or None
        One scalar per physical panel (26 for this flasher).
        If None, random values are used for demonstration.
    edge_data : dict or None
        Maps (panel_idx_A, panel_idx_B) -> scalar value for that shared edge.
        Key order doesn't matter — (0,1) and (1,0) are treated the same.
        If a shared edge is not in edge_data, it is drawn in gray.
        If None, all shared edges are drawn in gray (no data).

        Example:
            edge_data = {
                (0, 1): 0.45,
                (1, 5): 0.23,
                (5, 10): 0.88,
                ...
            }

    panel_cmap_name : str
        Colormap for panel fills. Default "viridis".
    edge_cmap_name : str
        Colormap for edge lines. Default "hot_r" — distinct from panel cmap.
    figsize : tuple
        Figure size in inches.
    panel_colorbar_label : str
        Label for the panel fill colorbar.
    edge_colorbar_label : str
        Label for the edge line colorbar.
    panel_vmin, panel_vmax : float or None
        Color scale limits for panel fills.
    edge_vmin, edge_vmax : float or None
        Color scale limits for edge lines.
    edge_linewidth : float
        Line width for shared edge lines. Default 4.0.
    save_path : str or None
        If provided, saves figure at 300 dpi.
    """

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
    # 4. Panel data — reorder by panel_map if pandas Series passed
    # ------------------------------------------------------------------
    if panel_data is None:
        rng = np.random.default_rng(42)
        panel_data = rng.uniform(0, 3.0, size=n_panels)
        print(f"No panel_data supplied — using {n_panels} random values.")
    else:
        if panel_map is not None and hasattr(panel_data, 'index'):
            panel_data = np.array([
                float(panel_data.loc[panel_map[i]])
                if panel_map[i] in panel_data.index else 0.0
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
    # 4b. Mask specified panels
    # ------------------------------------------------------------------
    if mask_panels is not None:
        mask = [i in mask_panels for i in range(n_panels)]
        panel_data = np.ma.array(panel_data, mask=mask)

    panel_cmap = plt.get_cmap(panel_cmap_name).copy()
    panel_cmap.set_bad(color="lightgray")

    if panel_vmin is None:
        panel_vmin = float(np.nanmin(panel_data))
    if panel_vmax is None:
        panel_vmax = float(np.nanmax(panel_data))

    panel_cnorm = mcolors.Normalize(vmin=panel_vmin, vmax=panel_vmax)

    # ------------------------------------------------------------------
    # 5. Edge data setup
    # ------------------------------------------------------------------
    # Normalize edge_data keys so (a,b) and (b,a) both work
    edge_data_norm = {}
    if edge_data is not None:
        for (a, b), val in edge_data.items():
            edge_data_norm[tuple(sorted((a, b)))] = val

    # Determine edge color scale
    if edge_data_norm:
        edge_vals = list(edge_data_norm.values())
        if edge_vmin is None:
            edge_vmin = min(edge_vals)
        if edge_vmax is None:
            edge_vmax = max(edge_vals)
    else:
        edge_vmin = edge_vmin or 0.0
        edge_vmax = edge_vmax or 1.0

    edge_cmap = plt.get_cmap(edge_cmap_name)
    edge_cnorm = mcolors.Normalize(vmin=edge_vmin, vmax=edge_vmax)

    # ------------------------------------------------------------------
    # 6. Find shared edges between physical panels
    # ------------------------------------------------------------------
    shared_edges = _find_shared_edges(panel_groups, faces, vertices, u_edges)

    # ------------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # --- Panel fills ---
    for panel_idx, face_group in enumerate(panel_groups):
        outline = _panel_outline(face_group, faces, vertices, u_edges)
        if outline is None:
            continue
        if np.ma.is_masked(panel_data[panel_idx]):
            fc = "lightgray"
        else:
            fc = panel_cmap(panel_cnorm(panel_data[panel_idx]))
        ax.add_patch(Polygon(outline, closed=True,
                             facecolor=fc, edgecolor="none", zorder=2))

    # --- Shared edge lines ---
    for (idx_a, idx_b, coords) in shared_edges:
        key = tuple(sorted((idx_a, idx_b)))
        if key in edge_data_norm:
            color = edge_cmap(edge_cnorm(edge_data_norm[key]))
        else:
            color = "gray"

        ax.plot([coords[0, 0], coords[1, 0]],
                [coords[0, 1], coords[1, 1]],
                color=color, lw=edge_linewidth, zorder=4,
                solid_capstyle="round")

    # --- Coordinate frame on center pentagon ---
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
    # 8. Two colorbars — panel fill (left) and edge lines (right)
    # ------------------------------------------------------------------
    # Panel fill colorbar
    sm_panel = plt.cm.ScalarMappable(cmap=panel_cmap, norm=panel_cnorm)
    sm_panel.set_array([])
    cbar_panel = fig.colorbar(sm_panel, ax=ax, shrink=0.7, aspect=30,
                              pad=0.01, location="right")
    cbar_panel.set_label(panel_colorbar_label, fontsize=12, fontweight="bold")
    cbar_panel.ax.tick_params(labelsize=9)

    # Edge line colorbar — placed to the right of the panel colorbar
    sm_edge = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_cnorm)
    sm_edge.set_array([])
    cbar_edge = fig.colorbar(sm_edge, ax=ax, shrink=0.7, aspect=30,
                             pad=0.01, location="right")
    cbar_edge.set_label(edge_colorbar_label, fontsize=12, fontweight="bold")
    cbar_edge.ax.tick_params(labelsize=9)



    # ------------------------------------------------------------------
    # 9. Formatting
    # ------------------------------------------------------------------
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle=":", alpha=0.3, zorder=0)
    ax.autoscale_view()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=False,
                    facecolor="white")
        print(f"Saved to {save_path}")

    plt.show()
    return fig, ax


# ----------------------------------------------------------------------
# Helper: build edge_data dict from tip/tilt DataFrames
# ----------------------------------------------------------------------

def build_edge_data_dict(shared_edges, panel_map,
                         tip_dfs, tilt_dfs,
                         tip_cols, tilt_cols):

    def tip_tilt_to_normal(tip_deg, tilt_deg):
        tip  = np.radians(tip_deg)
        tilt = np.radians(tilt_deg)
        nx = np.sin(tilt)
        ny = np.sin(tip)
        nz = np.sqrt(max(0.0, 1 - nx**2 - ny**2))  # max() guards against float rounding
        return np.array([nx, ny, nz])

    edge_data = {}

    for (idx_a, idx_b, _) in shared_edges:
        label_a = panel_map.get(idx_a)
        label_b = panel_map.get(idx_b)

        if label_a is None or label_b is None:
            continue

        relative_angles = []
        for tip_df, tilt_df, tip_col, tilt_col in zip(tip_dfs, tilt_dfs, tip_cols, tilt_cols):
            if (label_a in tip_df.index and label_b in tip_df.index and
                    label_a in tilt_df.index and label_b in tilt_df.index):

                tip_a  = float(tip_df.loc[label_a,  tip_col])
                tip_b  = float(tip_df.loc[label_b,  tip_col])
                tilt_a = float(tilt_df.loc[label_a, tilt_col])
                tilt_b = float(tilt_df.loc[label_b, tilt_col])

                n_a = tip_tilt_to_normal(tip_a, tilt_a)
                n_b = tip_tilt_to_normal(tip_b, tilt_b)
                angle = np.degrees(np.arccos(np.clip(np.dot(n_a, n_b), -1.0, 1.0)))
                relative_angles.append(angle)

        if relative_angles:
            edge_data[(idx_a, idx_b)] = float(np.mean(relative_angles))

    return edge_data


# ----------------------------------------------------------------------
# Helper: get shared edges without running the full plot
# ----------------------------------------------------------------------

def get_shared_edges(fold_path):
    """
    Returns list of (panel_idx_A, panel_idx_B, edge_coords) for all
    adjacent physical panel pairs. Use this to build edge_data in main.py.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fold_path_full = os.path.join(script_dir, fold_path)
    with open(fold_path_full, "r") as f:
        fold = json.load(f)

    vertices    = np.array(fold["vertices_coords"])
    edges_verts = fold["edges_vertices"]
    assignments = fold["edges_assignment"]
    faces       = fold["faces_vertices"]

    rotation_deg = -20
    angle_r = np.radians(rotation_deg)
    rot = np.array([[np.cos(angle_r), -np.sin(angle_r)],
                    [np.sin(angle_r),  np.cos(angle_r)]])
    vertices = vertices @ rot.T

    u_edges = {
        tuple(sorted(e))
        for e, a in zip(edges_verts, assignments)
        if a in ("U", "F")
    }
    panel_groups = _union_find_groups(faces, u_edges)
    return _find_shared_edges(panel_groups, faces, vertices, u_edges)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)

    # Dummy panel data (mean euler angle per panel)
    panel_vals = rng.uniform(0, 3.0, size=26)

    # Dummy edge data — you'd normally build this with build_edge_data_dict()
    shared = get_shared_edges("r1_h2_m5_flasher.fold")
    edge_vals = {(a, b): rng.uniform(0, 2.0) for a, b, _ in shared}

    plot_euler_angle_heatmap(
        fold_path="r1_h2_m5_flasher.fold",
        panel_data=panel_vals,
        edge_data=edge_vals,
        mask_panels=[25],
        panel_cmap_name="viridis",
        edge_cmap_name="hot_r",
        figsize=(10, 9),
        title="Euler Angle Heatmap — Panel Fill + Edge Relative Angle",
        panel_colorbar_label="Mean Euler Angle [°]",
        edge_colorbar_label="Mean Relative Angle [°]",
        panel_vmin=0,
        edge_vmin=0,
        edge_linewidth=5.0,
        save_path="euler_angle_heatmap.png",
    )