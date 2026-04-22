#%%
import numpy as np
import pandas as pd
from plot_tip_tilt_flasher import *
from plot_decenter_flasher import *

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

#####################################################
#### TIP AND TILT HEATMAPS AND MEAN TIP/TILT HEATMAPS ACROSS DEPLOYMENTS ###
#####################################################

# # Mean tip across deployments
# mean_tip = pd.DataFrame({
#     "mean_tip": pd.concat([
#         tip_deployment1["tip_deployment1"],
#         tip_deployment2["tip_deployment2"],
#         tip_deployment3["tip_deployment3"],
#         tip_deployment4["tip_deployment4"]],
#         axis=1).mean(axis=1)
# })

# get max average tip/tilt value across panels to set symmetric vmin/vmax for diverging colormap
# max_abs_value = max(mean_tip["mean_tip"].abs().max(), mean_tilt["mean_tilt"].abs().max())


#####################################################
#### LOCAL DE CENTERING CALCULATION AND PLOTTING ####
#####################################################

x_decenter_deployment1 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment1",
    label_col=0, value_col=1,
    value_name="x_decenter_deployment1",
    skiprows=6, nrows=25, usecols="B,F"
)

y_decenter_deployment1 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment1",
    label_col=0, value_col=1,
    value_name="y_decenter_deployment1",
    skiprows=6, nrows=25, usecols="B,G"
)

x_decenter_deployment2 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment2",
    label_col=0, value_col=1,
    value_name="x_decenter_deployment2",
    skiprows=6, nrows=25, usecols="B,F"
)

y_decenter_deployment2 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment2",
    label_col=0, value_col=1,
    value_name="y_decenter_deployment2",
    skiprows=6, nrows=25, usecols="B,G"
)

x_decenter_deployment3 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment3",
    label_col=0, value_col=1,
    value_name="x_decenter_deployment3",
    skiprows=6, nrows=25, usecols="B,F"
)

y_decenter_deployment3 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment3",
    label_col=0, value_col=1,
    value_name="y_decenter_deployment3",
    skiprows=6, nrows=25, usecols="B,G"
)

x_decenter_deployment4 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment4",
    label_col=0, value_col=1,
    value_name="x_decenter_deployment4",
    skiprows=6, nrows=25, usecols="B,F"
)

y_decenter_deployment4 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment4",
    label_col=0, value_col=1,
    value_name="y_decenter_deployment4",
    skiprows=6, nrows=25, usecols="B,G"
)


decenter_magnitude_deployment1 = np.sqrt(x_decenter_deployment1["x_decenter_deployment1"]**2 + y_decenter_deployment1["y_decenter_deployment1"]**2)
decenter_magnitude_deployment2 = np.sqrt(x_decenter_deployment2["x_decenter_deployment2"]**2 + y_decenter_deployment2["y_decenter_deployment2"]**2)
decenter_magnitude_deployment3 = np.sqrt(x_decenter_deployment3["x_decenter_deployment3"]**2 + y_decenter_deployment3["y_decenter_deployment3"]**2)
decenter_magnitude_deployment4 = np.sqrt(x_decenter_deployment4["x_decenter_deployment4"]**2 + y_decenter_deployment4["y_decenter_deployment4"]**2)

# Mean decenter magnitude across deployments
mean_magnitude_decenter = pd.DataFrame({
    "mean_magnitude_decenter": pd.concat([
        decenter_magnitude_deployment1.abs(),
        decenter_magnitude_deployment2.abs(),
        decenter_magnitude_deployment3.abs(),
        decenter_magnitude_deployment4.abs()],
        axis=1).mean(axis=1)
})

max_decenter_value = mean_magnitude_decenter["mean_magnitude_decenter"].max()

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

plot_decenter_heatmap_datapoints(
    fold_path=fold_path,
    panel_map=panel_map,
    panel_data=mean_magnitude_decenter["mean_magnitude_decenter"],
    decenter_data=decenter_data,
    point_labels=["Deployment 1", "Deployment 2", "Deployment 3", "Deployment 4"],
    cmap_name="viridis",
    title="",
    colorbar_label="Mean Decenter Magnitude [mm]",
    vmin=0,
    vmax=max_decenter_value,
    save_path="mean_decenter_heatmap_with_datapoints.png",
    scale_factor=0.15,
    use_local_frame=False,
    origin_dot_color="black",
    data_dot_size=60,
    show_crosshair=True,
    crosshair_size=0.04,
)

#%%