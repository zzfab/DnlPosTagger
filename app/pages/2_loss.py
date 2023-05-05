import streamlit as st
import pandas as pd
import altair as alt
from pytorch_lightning.loggers import CSVLogger


csv_logger = CSVLogger("lightning_logs", name="pos_tagging")
log_path = csv_logger.experiment.metrics_file_path
metrics_df = pd.read_csv(log_path)


loss_chart = alt.Chart(metrics_df).mark_line().transform_fold(
    ["train_loss", "val_loss"],
    as_=["Loss Type", "Loss Value"]
).encode(
    x=alt.X("epoch:Q", title="Epoch"),
    y=alt.Y("Loss Value:Q", title="Loss"),
    color="Loss Type:N"
)

st.subheader("Training and Validation Losses")
st.write(loss_chart)