from eda import bivariate_plot
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

eda = st.container()


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
    st.title("""
    # Turbine: Sensor Data Analytics Application
    A simple application for end-to-end machine learning.
    
    """)

with dataset_base:
    st.header('''
    Sensor dataset 
    This is a sample of the data present in the database.
    Only the first few rows are shown.
    ''')
    st.write(data_df.head())

with dataset_stats:
    st.header('''
    Sensor-wise dataset summary
    These are sample statistics generated for each sensor
    Key statistics generated are mean, median, standard deviation, kurtosis and skewness.
    ''')
    st.write(sensor_wise_stats)

with visualizations:
    st.header('''
    Visualizations of the sensor data - univariate analysis
    ''')
    kind = st.selectbox(f"Specify the kind of visualization you'd like to see for the features",
                    ("run chart", "histogram"))
    plots = visualize_data(data_df, metric_columns, kind)
    for plot_id, plot in plots.items():
        st.subheader(f"Visualization for sensor id: {plot_id}")
        st.pyplot(plot)

with eda:
    st.header('''
    Exploratory data analysis
    This section can display joint density plots, and other kinds of charts
    Correlation analysis may also be performed here.
    ''')
    eda_type = st.selectbox(f"Specify kind of bivariate plot",
                            ("joint density plot", "scatter plot"))
    first = st.selectbox(f"Specify first variable",
                            metric_columns)
    second = st.selectbox(f"Specify second variable",
                            metric_columns)

    if first == second:
        st.write("Warning: Different variables should be selected to see a bivariate analysis")
    else:
        st.subheader(f"{eda_type} visualization for {second} vs {first}")
        eda_plot = bivariate_plot(data_df, first, second, eda_type)
        st.pyplot(eda_plot)



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
        
    
    

    