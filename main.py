#%%
import numpy as np
import pandas as pd
from plot_tip_tilt_flasher import *

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

# Mean tip across deployments
mean_tip = pd.DataFrame({
    "mean_tip": pd.concat([
        tip_deployment1["tip_deployment1"],
        tip_deployment2["tip_deployment2"],
        tip_deployment3["tip_deployment3"],
        tip_deployment4["tip_deployment4"]],
        axis=1).mean(axis=1)
})

# Mean tilt across deployments
mean_tilt = pd.DataFrame({
    "mean_tilt": pd.concat([
        tilt_deployment1["tilt_deployment1"],
        tilt_deployment2["tilt_deployment2"],
        tilt_deployment3["tilt_deployment3"],
        tilt_deployment4["tilt_deployment4"]],
        axis=1).mean(axis=1)
})

# get max average tip/tilt value across panels to set symmetric vmin/vmax for diverging colormap
max_abs_value = max(mean_tip["mean_tip"].abs().max(), mean_tilt["mean_tilt"].abs().max())

plot_flasher_heatmap(
    fold_path=fold_path,
    panel_data=df_to_panel_array(df=mean_tip, column="mean_tip", panel_map=panel_map),
    cmap_name="coolwarm",  # diverging colormap makes sense since tip/tilt can be positive or negative
    colorbar_label="Mean Tip [°]",
    vmin=-max_abs_value,
    vmax=max_abs_value,
    save_path="mean_tip_heatmap.png"
)

plot_flasher_heatmap(
    fold_path=fold_path,
    panel_data=df_to_panel_array(df=mean_tilt, column="mean_tilt", panel_map=panel_map),
    cmap_name="coolwarm",
    colorbar_label="Mean Tilt [°]",
    vmin=-max_abs_value,
    vmax=max_abs_value,
    save_path="mean_tilt_heatmap.png"
)
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

# Mean x_decenter across deployments
mean_x_decenter = pd.DataFrame({
    "mean_x_decenter": pd.concat([
        x_decenter_deployment1["x_decenter_deployment1"],
        x_decenter_deployment2["x_decenter_deployment2"],
        x_decenter_deployment3["x_decenter_deployment3"],
        x_decenter_deployment4["x_decenter_deployment4"]],
        axis=1).mean(axis=1)
})

# Mean y_decenter across deployments
mean_y_decenter = pd.DataFrame({
    "mean_y_decenter": pd.concat([
        y_decenter_deployment1["y_decenter_deployment1"],
        y_decenter_deployment2["y_decenter_deployment2"],
        y_decenter_deployment3["y_decenter_deployment3"],
        y_decenter_deployment4["y_decenter_deployment4"]],
        axis=1).mean(axis=1)
})

# get max average decenter value across panels to set symmetric vmin/vmax for diverging colormap
max_abs_decenter = max(mean_x_decenter["mean_x_decenter"].abs().max(), mean_y_decenter["mean_y_decenter"].abs().max())

plot_flasher_heatmap(
    fold_path=fold_path,
    panel_data=df_to_panel_array(df=mean_x_decenter, column="mean_x_decenter", panel_map=panel_map),
    cmap_name="coolwarm",
    colorbar_label="Mean X Decenter [mm]",
    vmin=-max_abs_decenter,
    vmax=max_abs_decenter,
    save_path="mean_x_decenter_heatmap.png"
)

plot_flasher_heatmap(
    fold_path=fold_path,
    panel_data=df_to_panel_array(df=mean_y_decenter, column="mean_y_decenter", panel_map=panel_map),
    cmap_name="coolwarm",
    colorbar_label="Mean Y Decenter [mm]",
    vmin=-max_abs_decenter,
    vmax=max_abs_decenter,
    save_path="mean_y_decenter_heatmap.png"
)

print("These are the scale values")
print(f"Max abs mean tip: {max_abs_value:.2f}°")
print(f"Max abs mean tilt: {max_abs_value:.2f}°")
print(f"Max abs mean x_decenter: {max_abs_decenter:.2f} mm")
print(f"Max abs mean y_decenter: {max_abs_decenter:.2f} mm")

#### RSS Calculation and Plotting ####
#### PLOTS MEAN RSS ACROSS DEPLOYMENTS 1-3, STD DEV OF RSS ACROSS DEPLOYMENTS 1-3, AND CV OF RSS ACROSS DEPLOYMENTS 1-3 ####
#### I DON'T THINK ITS MEANINGFUL BECAUSE WE ONLY HAVE 4 DEPLOYMENTS

# # Join and compute RSS
# rss_deployment1 = tip_deployment1.join(tilt_deployment1)
# rss_deployment1["rss"] = np.sqrt(rss_deployment1["tip_deployment1"]**2 + rss_deployment1["tilt_deployment1"]**2)

# rss_deployment2 = tip_deployment2.join(tilt_deployment2)
# rss_deployment2["rss"] = np.sqrt(rss_deployment2["tip_deployment2"]**2 + rss_deployment2["tilt_deployment2"]**2)

# rss_deployment3 = tip_deployment3.join(tilt_deployment3)
# rss_deployment3["rss"] = np.sqrt(rss_deployment3["tip_deployment3"]**2 + rss_deployment3["tilt_deployment3"]**2)

# # Combine all deployments into one DataFrame with the mean of the three deployments
# mean_rss_deployments1_3 = pd.DataFrame({
#     "rootsumsquared": (rss_deployment1["rss"] + rss_deployment2["rss"] + rss_deployment3["rss"]) / 3
# }) 

# std_rss_deployments1_3 = pd.DataFrame({
#     "rss_std": pd.concat([rss_deployment1["rss"], rss_deployment2["rss"], rss_deployment3["rss"]], axis=1).std(axis=1)
# })


# combined_data_rss_mean1_3 = df_to_panel_array(
#     df=mean_rss_deployments1_3,
#     column="rootsumsquared",
#     panel_map=panel_map
# )
# combined_data_rss_stddev1_3 = df_to_panel_array(
#     df=std_rss_deployments1_3,
#     column="rss_std",
#     panel_map=panel_map
# )


# print("Plotting flasher heatmap...")

# plot_flasher_heatmap(
#     fold_path=fold_path,
#     panel_data=combined_data_rss_mean1_3,   # length 26
#     cmap_name="viridis",
#     colorbar_label="RSS Tip/Tilt Values (mean of 1-3) [°]",
#     vmin=0,
#     vmax=np.max(combined_data_rss_mean1_3),
#     save_path="mean_rss_heatmap.png"
# )

# plot_flasher_heatmap(
#     fold_path=fold_path,
#     panel_data=combined_data_rss_stddev1_3,   # length 26
#     cmap_name="viridis",
#     colorbar_label="RSS Tip/Tilt Values (std dev of 1-3) [°]",
#     vmin=0,
#     vmax=np.max(combined_data_rss_stddev1_3),
#     save_path="mean_rss_stddev_heatmap.png"
# )

# cv_rss = pd.DataFrame({
#     "rss_cv": std_rss_deployments1_3["rss_std"] / mean_rss_deployments1_3["rootsumsquared"]
# }).fillna(0)  # avoid divide-by-zero for Aper0

# plot_flasher_heatmap(
#     fold_path=fold_path,
#     panel_data=df_to_panel_array(df=cv_rss, column="rss_cv", panel_map=panel_map),
#     cmap_name="viridis",
#     colorbar_label="CV of RSS Tip/Tilt (std/mean) [°]",
#     vmin=0,
#     vmax=cv_rss["rss_cv"].max()
# )
# %%
