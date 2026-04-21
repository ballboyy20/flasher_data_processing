import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from collections import defaultdict
import pandas as pd




def load_panel_data_returns_pd_dataframe(excel_path, sheet_name=0, label_col=0, value_col=1, 
                    value_name="value", skiprows=0, nrows=None, usecols=None,center_value=0.0):
    """
    Reads aperture data from Excel and returns a DataFrame indexed by aperture label.

    Parameters
    ----------
    excel_path : str
        Path to the Excel file.
    sheet_name : int or str
        Sheet tab name or index. Default 0 (first sheet).
    label_col : int
        Column index containing aperture labels after usecols filtering. Default 0.
    value_col : int
        Column index containing values after usecols filtering. Default 1.
    value_name : str
        Name for the value column in the returned DataFrame. e.g. "tip", "tilt"
    skiprows : int
        Number of rows to skip at the top before reading. Default 0.
    nrows : int or None
        Number of rows to read. Use this to stop before summary rows. Default None (all).
    usecols : str or None
        Excel columns to read, e.g. "A,B" or "A,C". Default None (all columns).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, excel_path)

    df = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        header=None,
        skiprows=skiprows,
        nrows=nrows,
        usecols=usecols,
    )

    # After usecols filtering, reset column positions to 0, 1, 2...
    df.columns = range(len(df.columns))

    df = df.rename(columns={label_col: "label", value_col: value_name})
    df = df.set_index("label")[[value_name]]
    df.loc["Aper0"] = center_value  # add this line
    return df

def df_to_panel_array(df, column, panel_map):
    """Convert a labeled DataFrame column to ordered numpy array for the plotter."""
    
    return np.array([df.loc[panel_map[i], column] if panel_map[i] in df.index else 0.0
                     for i in range(26)])




# ----------------------------------------------------------------------
# Internal helpers for union-find panel grouping
# ----------------------------------------------------------------------

def _union_find_groups(faces, u_edges):
    """Group triangulated faces into physical panels via union-find on U edges."""
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
    """Return ordered outline coordinates for a group of triangulated faces."""
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


# ----------------------------------------------------------------------
# Main functions
# ----------------------------------------------------------------------

def plot_flasher_heatmap(
    fold_path,
    panel_data=None,
    cmap_name="coolwarm",
    title=None,
    colorbar_label="RSS Tip/Tilt [°]",
    vmin=None,
    vmax=None,
    save_path=None,
):
    """
    Plots a flasher .fold file as a heat map, filling each physical panel
    with a color corresponding to a data value (e.g. RSS of tip and tilt).

    Triangulated sub-faces are merged into physical panels using union-find
    on the unassigned (U) edges. No cross-bar lines are drawn.

    Parameters
    ----------
    fold_path : str
        Path to the .fold file. Relative paths resolve from this script's directory.
    panel_data : array-like or None
        One scalar value per physical panel (26 for this flasher).
        If None, random values are generated for demonstration.
    cmap_name : str
        Matplotlib colormap name. Default "coolwarm".
    title : str
        Figure title.
    colorbar_label : str
        Label for the colorbar axis.
    vmin, vmax : float or None
        Color scale limits. If None, computed from panel_data.
    save_path : str or None
        If provided, saves the figure to this path.
    """

    # 1. Load .fold file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fold_path = os.path.join(script_dir, fold_path)
    with open(fold_path, "r") as f:
        fold = json.load(f)

    vertices    = np.array(fold["vertices_coords"])
    edges_verts = fold["edges_vertices"]
    assignments = fold["edges_assignment"]
    faces       = fold["faces_vertices"]

    # After loading vertices, rotate the entire geometry
    rotation_deg = -20  # adjust until oriented correctly ROTATE WHOLE FLASHER HERE
    angle = np.radians(rotation_deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
    vertices = vertices @ rot.T

    # 2. Identify U edges (internal diagonals)
    u_edges = {
        tuple(sorted(e))
        for e, a in zip(edges_verts, assignments)
        if a in ("U", "F")
    }

    # 3. Group triangulated faces into physical panels
    panel_groups = _union_find_groups(faces, u_edges)
    n_panels = len(panel_groups)

    # 4. Panel data
    if panel_data is None:
        rng = np.random.default_rng(42)
        panel_data = rng.uniform(0, 3.0, size=n_panels)
        print(f"No panel_data supplied — using {n_panels} random values.")
    else:
        panel_data = np.asarray(panel_data, dtype=float)
        if len(panel_data) != n_panels:
            raise ValueError(
                f"panel_data length ({len(panel_data)}) must match "
                f"number of physical panels ({n_panels}). "
                f"Pass size={n_panels} when generating your array."
            )

    # 5. Color map
    if vmin is None:
        vmin = panel_data.min()
    if vmax is None:
        vmax = panel_data.max()

    cmap  = plt.get_cmap(cmap_name)
    cnorm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 6. Plot filled panels
    fig, ax = plt.subplots(figsize=(9, 8))

    for panel_idx, face_group in enumerate(panel_groups):
        outline = _panel_outline(face_group, faces, vertices, u_edges)
        if outline is None:
            continue
        fc = cmap(cnorm(panel_data[panel_idx]))
        ax.add_patch(Polygon(outline, closed=True,
                             facecolor=fc, edgecolor="none", zorder=2))
        
    # 7. Draw coordinate frame on center pentagon
    # Find centroid of center panel (panel index 25 = Aper0)
    center_panel_idx = 25
    center_group = panel_groups[center_panel_idx]
    all_verts = []
    for fi in center_group:
        for vi in faces[fi]:
            all_verts.append(vertices[vi])
    centroid = np.mean(all_verts, axis=0)

    # Coordinate frame settings — adjust these
    arrow_length = 0.3      # length of each axis arrow
    rotation_deg = 0.0      # rotate the frame (degrees) — adjust until correct ROTATE COORD FRAME HERE

    angle = np.radians(rotation_deg)
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])

    x_axis = rot @ np.array([1, 0]) * arrow_length
    y_axis = rot @ np.array([0, 1]) * arrow_length

    arrow_kwargs = dict(head_width=0.06, head_length=0.04, length_includes_head=True, zorder=6)

    ax.arrow(centroid[0], centroid[1], x_axis[0], x_axis[1],
             fc="black", ec="black", **arrow_kwargs)
    ax.arrow(centroid[0], centroid[1], y_axis[0], y_axis[1],
             fc="black", ec="black", **arrow_kwargs)

    ax.text(centroid[0] + x_axis[0] * 1.2, centroid[1] + x_axis[1] * 1.2,
            "X", color="black", fontsize=10, fontweight="bold", zorder=6, ha="center")
    ax.text(centroid[0] + y_axis[0] * 1.2, centroid[1] + y_axis[1] * 1.2,
            "Y", color="black", fontsize=10, fontweight="bold", zorder=6, ha="center")


    # # 7. Draw panel boundary lines only
    # plotted = set()
    # for (i, j), asgn in zip(edges_verts, assignments):
    #     if asgn in ("U", "F"):
    #         continue
    #     key = tuple(sorted((i, j)))
    #     if key in plotted:
    #         continue
    #     plotted.add(key)
    #     x = [vertices[i][0], vertices[j][0]]
    #     y = [vertices[i][1], vertices[j][1]]
    #     ax.plot(x, y, color="black", lw=1.5, linestyle="-", zorder=4)

    # 8. Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=18, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    # # 9. Legend
    # legend_elements = [
    #     Line2D([0], [0], color="black", lw=2.2, ls="-", label="Panel boundary"),
    # ]
    # ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.85)

    # 10. Formatting
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, linestyle=":", alpha=0.3, zorder=0)
    ax.autoscale_view()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", transparent=True)
        print(f"Saved to {save_path}")

    plt.show()
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    return fig, ax


