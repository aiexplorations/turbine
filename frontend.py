from sensor_dataframe import return_df_from_db
import pandas as pd
import streamlit as st

# This is a script to set up a streamlit frontend UI for a data science project
# Using streamlit components such as containers for building up the UI

header = st.container()

dataset = st.container()

visualizations = st.container()

features = st.container()

transformations = st.container()

model_training = st.container()

model_saving = st.container()

with header:
    st.title("Turbine: Sensor Data Analytics Application")

with dataset:
    st.header("Sensor dataset summary")
    data_df = return_df_from_db()
    print(data_df.head())
    st.write(data_df)