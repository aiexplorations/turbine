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
sensor_ids = data_df['device_id'].unique().tolist()
sensor_wise_stats = generate_stats(data_df, metric_columns)


# Display on streamlit

with header:
    st.markdown("""
    # Turbine: Sensor Data Analytics Application
    A simple application for end-to-end machine learning.
    
    """)

with dataset_base:
    st.header('''
    Sensor dataset 
    **Note:**
    * This is a sample of the data present in the database.
    * Only the first few rows are shown.
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
    sel_viz, disp_viz = st.columns(2)
    
    sensor = sel_viz.selectbox(f"Specify the sensor to be visualized",
                    sensor_ids)
    metric = sel_viz.selectbox(f"Specify the metric to be visualized",
                    metric_columns)
    kind = sel_viz.selectbox(f"Specify the kind of visualization",
                    ("run chart", "histogram"))
    plots = visualize_data(data_df, metric_columns, kind)

    st.subheader(f"Visualization for sensor id:")
    st.pyplot(plots[sensor+"_"+metric+"_"+kind])

with eda:
    st.header('''
    Exploratory data analysis
    This section can display joint density plots, and other kinds of charts
    Correlation analysis may also be performed here.
    ''')

    sel_column, disp_column = st.columns(2)

    eda_type = sel_column.selectbox(f"Specify kind of bivariate plot",
                            ("joint density plot", "scatter plot"))
    first = sel_column.selectbox(f"Specify first variable",
                            metric_columns)
    second = sel_column.selectbox(f"Specify second variable",
                            metric_columns)

    if first == second:
        st.write("Warning: Different variables should be selected to see a bivariate analysis")
    else:
        disp_column.subheader(f"{eda_type} visualization for {second} vs {first}")
        eda_plot = bivariate_plot(data_df, first, second, eda_type)
        disp_column.pyplot(eda_plot)



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
        
    
    

    