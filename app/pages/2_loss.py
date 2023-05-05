import streamlit as st
import pandas as pd
import altair as alt
import os
import sys
import re

wdir = os.path.dirname(os.getcwd())
sys.path.append(wdir)

log_dir = os.path.join(wdir, "src/lightning_logs/pos_tagging")

# Find the latest version by sorting subdirectories based on their version number
versions = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
versions.sort(key=lambda d: int(re.search(r'version_(\d+)', d).group(1)), reverse=True)

# Find the latest version that contains the 'train_loss_step' column
for version in versions:
    log_path = os.path.join(log_dir, version, "metrics.csv")
    metrics_df = pd.read_csv(log_path)
    if "train_loss_step" in metrics_df.columns:
        break
# Reshape the DataFrame
metrics_df = metrics_df.melt(id_vars=["epoch", "step"], var_name="Loss Type", value_name="Loss Value")
metrics_df = metrics_df.dropna(subset=["Loss Value"])  # Drop rows with NaN loss values
metrics_df["Loss Type"] = metrics_df["Loss Type"].replace({"train_loss_epoch": "train_loss", "val_loss_epoch": "val_loss"})
# Filter out train_loss_step rows
metrics_df = metrics_df[metrics_df["Loss Type"] != "train_loss_step"]

loss_chart = alt.Chart(metrics_df).mark_line().encode(
    x=alt.X("epoch:Q", title="Epoch"),
    y=alt.Y("Loss Value:Q", title="Loss"),
    color="Loss Type:N"
).properties(
    width=800,
    height=400
)

st.subheader("Training and Validation Losses")
st.write(loss_chart)

latest_version = versions[0]  # Pick the latest version
log_path = os.path.join(log_dir, latest_version, "metrics.csv")

metrics_df = pd.read_csv(log_path)
# Get the latest test_acc value
test_acc_df = metrics_df
latest_test_acc = test_acc_df.iloc[-1]["test_acc"]

st.subheader("Test Accuracy")
st.markdown(f"<h2 style='font-size: 32px;'>{latest_test_acc:.2%}</h2>", unsafe_allow_html=True)
