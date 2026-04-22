#%%
import numpy as np
import pandas as pd
from plot_decenter_flasher import *
from plot_tiptilt_flasher import *



fold_path = "r1_h2_m5_flasher.fold"

#### LOAD TIP/TILT DATA FROM EXCEL FILE ####
panel_map = {
    0:  "Ap1G3",
    1:  "Ap3G2",
    2:  "Ap5G1",
    3:  "Ap2G3",
    4:  "Ap4G2",
    5:  "Ap1G2",
    6:  "Ap3G1",
    7:  "Ap5G5",
    8:  "Ap2G2",
    9:  "Ap4G1",
    10: "Ap1G1",
    11: "Ap3G5",
    12: "Ap5G4",
    13: "Ap2G1",
    14: "Ap4G5",
    15: "Ap1G5",
    16: "Ap3G4",
    17: "Ap5G3",
    18: "Ap2G5",
    19: "Ap4G4",
    20: "Ap1G4",
    21: "Ap3G3",
    22: "Ap5G2",
    23: "Ap2G4",
    24: "Ap4G3",
    25: "Aper0"
}

tip_deployment1 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment1",
    label_col=0, value_col=1,
    value_name="tip_deployment1",
    skiprows=6, nrows=25, usecols="B,C"
)

tilt_deployment1 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment1",
    label_col=0, value_col=1,
    value_name="tilt_deployment1",
    skiprows=6, nrows=25, usecols="B,D"
)

tip_deployment2 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment2",
    label_col=0, value_col=1,
    value_name="tip_deployment2",
    skiprows=6, nrows=25, usecols="B,C"
)

tilt_deployment2 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment2",
    label_col=0, value_col=1,
    value_name="tilt_deployment2",
    skiprows=6, nrows=25, usecols="B,D"
)

tip_deployment3 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment3",
    label_col=0, value_col=1,
    value_name="tip_deployment3",
    skiprows=6, nrows=25, usecols="B,C"
)

tilt_deployment3 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment3",
    label_col=0, value_col=1,
    value_name="tilt_deployment3",
    skiprows=6, nrows=25, usecols="B,D"
)

tip_deployment4 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment4",
    label_col=0, value_col=1,
    value_name="tip_deployment4",
    skiprows=6, nrows=25, usecols="B,C"
)

tilt_deployment4 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment4",
    label_col=0, value_col=1,
    value_name="tilt_deployment4",
    skiprows=6, nrows=25, usecols="B,D"
)

euler_angle_deployment1 = tip_deployment1["tip_deployment1"].combine(
    tilt_deployment1["tilt_deployment1"],
    lambda t, tl: tip_tilt_to_euler(t, tl)
)

euler_angle_deployment2 = tip_deployment2["tip_deployment2"].combine(
    tilt_deployment2["tilt_deployment2"],
    lambda t, tl: tip_tilt_to_euler(t, tl)
)

euler_angle_deployment3 = tip_deployment3["tip_deployment3"].combine(
    tilt_deployment3["tilt_deployment3"],
    lambda t, tl: tip_tilt_to_euler(t, tl)
)

euler_angle_deployment4 = tip_deployment4["tip_deployment4"].combine(
    tilt_deployment4["tilt_deployment4"],
    lambda t, tl: tip_tilt_to_euler(t, tl)
)

mean_euler_angle = pd.DataFrame({
    "mean_euler_angle": pd.concat([
        euler_angle_deployment1,
        euler_angle_deployment2,
        euler_angle_deployment3,
        euler_angle_deployment4],
        axis=1).mean(axis=1)
})

max_euler_angle_value = mean_euler_angle["mean_euler_angle"].max()

shared_edges = get_shared_edges(fold_path)

# Build edge data from your tip/tilt DataFrames
edge_data = build_edge_data_dict(
    shared_edges, panel_map,
    tip_dfs=[tip_deployment1, tip_deployment2, tip_deployment3, tip_deployment4],
    tilt_dfs=[tilt_deployment1, tilt_deployment2, tilt_deployment3, tilt_deployment4],
    tip_cols=["tip_deployment1", "tip_deployment2", "tip_deployment3", "tip_deployment4"],
    tilt_cols=["tilt_deployment1", "tilt_deployment2", "tilt_deployment3", "tilt_deployment4"],
)

plot_euler_angle_heatmap(
        fold_path=fold_path,
        panel_map=panel_map,
        mask_panels=[25],
        panel_data=mean_euler_angle["mean_euler_angle"],
        edge_data=edge_data,
        panel_cmap_name="viridis",
        edge_cmap_name="hot_r",
        figsize=(10, 9),
        panel_colorbar_label="Mean Euler Angle [°]",
        edge_colorbar_label="Mean Relative Angle [°]",
        panel_vmax=max_euler_angle_value,
        edge_linewidth=5.0,
        save_path="euler_angle_heatmap.png",
    )

print("The max euler angle value across all deployments is:", max_euler_angle_value)
print("The max panel to panel relative angle across all deployments is:", max(edge_data["mean_relative_angle"]))
#%%