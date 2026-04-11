import numpy as np
import pandas as pd
from plot_tip_tilt_flasher import *

fold_path = "r1_h2_m5_flasher.fold"


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

tilt_deployment2 = load_panel_data_returns_pd_dataframe(
    excel_path="combined_tiptilt_data.xlsx",
    sheet_name="Deployment2",
    label_col=0, value_col=1,
    value_name="tilt_deployment2",
    skiprows=6, nrows=25, usecols="B,D"
)

# Join and compute RSS
rss_deployment1 = tip_deployment1.join(tilt_deployment1)
rss_deployment1["rss"] = np.sqrt(rss_deployment1["tip_deployment1"]**2 + rss_deployment1["tilt_deployment1"]**2)

rss_deployment2 = tip_deployment2.join(tilt_deployment2)
rss_deployment2["rss"] = np.sqrt(rss_deployment2["tip_deployment2"]**2 + rss_deployment2["tilt_deployment2"]**2)

rss_deployment3 = tip_deployment3.join(tilt_deployment3)
rss_deployment3["rss"] = np.sqrt(rss_deployment3["tip_deployment3"]**2 + rss_deployment3["tilt_deployment3"]**2)

# Combine all deployments into one DataFrame with the mean of the three deployments
mean_rss_deployments1_3 = pd.DataFrame({
    "rss": (rss_deployment1["rss"] + rss_deployment2["rss"] + rss_deployment3["rss"]) / 3
}) 


combined_data_rss_mean1_3 = df_to_panel_array(
    df=mean_rss_deployments1_3,
    column="rss",
    panel_map=panel_map
)

print("Plotting flasher heatmap...")

plot_flasher_heatmap(
    fold_path=fold_path,
    panel_data=combined_data_rss_mean1_3,   # length 26
    cmap_name="viridis",
    colorbar_label="RSS Tip/Tilt Values Deployment 1 [°]",
    vmin=0,
    vmax=np.max(combined_data_rss_mean1_3)
)
