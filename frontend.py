from visualize import visualize_data
from sensor_dataframe import generate_stats, return_df_from_db, transform_df
import numpy as np
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
metric_columns = ['x1', 'x2', 'target']

sensor_wise_stats = generate_stats(data_df, metric_columns)


# Display on streamlit

with header:
    st.title("Turbine: Sensor Data Analytics Application")

with dataset_base:
    st.header("Sensor dataset head")
    st.write(data_df.head())

with dataset_stats:
    st.header("Sensor-wise dataset summary")
    st.write(sensor_wise_stats)

with visualizations:
    st.header("Visualizations of the sensor data")
    kind = st.selectbox(f"Specify the kind of visualization you'd like to see for the features",
                    ("run chart", "histogram"))
    plots = visualize_data(data_df, metric_columns, kind)
    for plot_id, plot in plots.items():
        st.subheader(f"Visualization for sensor id: {plot_id}")
        st.pyplot(plot)

with features:
    st.header("Feature engineering options")
    transforms = {}
    for col in metric_columns:
        transformation = st.selectbox(f"Select the transform for {col}", 
                                      ("none", "log", "sqrt", "exp"))
        transforms.update({col: transformation})
    
    transformed_df = transform_df(data_df, transforms)
    st.subheader("Transformed dataframe (sample)")
    st.write(transformed_df.sample(10))
        
    
    

    