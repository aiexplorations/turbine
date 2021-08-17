from visualize import visualize_data
from sensor_dataframe import generate_stats, return_df_from_db
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# This is a script to set up a streamlit frontend UI for a data science project
# Using streamlit components such as containers for building up the UI

header = st.container()

dataset_base = st.container()

dataset_stats = st.container()

visualizations = st.container()

features = st.container()

transformations = st.container()

model_training = st.container()

model_saving = st.container()

# Retrieve data from database and compute statistics

data_df = return_df_from_db()

sensor_wise_stats = generate_stats(data_df)

plots = visualize_data(data_df)

# Display on streamlit

with header:
    st.title("""
    # Turbine: Sensor Data Analytics Application
    A simple application for end-to-end machine learning.
    
    """)

with dataset_base:
    st.header("Sensor dataset head")
    st.write(data_df.head())

with dataset_stats:
    st.header("Sensor-wise dataset summary")
    st.write(sensor_wise_stats)

with visualizations:
    st.header("Visualizations of the sensor data")
    for sensor_id, plot in plots.items():
        st.subheader(f"Visualization for sensor id: {sensor_id}")
        st.pyplot(plot)
