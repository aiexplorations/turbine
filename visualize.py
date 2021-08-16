from sensor_dataframe import return_df_from_db
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns



def visualize_data(sensor_df):
    unique_sensor_ids = sensor_df.device_id.unique().tolist()
    device_wise_df = {}
    for sensor_id in unique_sensor_ids:
        device_wise_df.update({sensor_id: sensor_df[sensor_df['device_id']==sensor_id]})
    
    plots = {}
    
    # Generate run charts for each sensor ID of the value
    for sensor_id in unique_sensor_ids:

        plt.figure(figsize = (16,4))
        plt.title(f"Run chart for sensor id : {sensor_id}")
        plt.plot(device_wise_df[sensor_id]['value'])
        plots.update({sensor_id: plt})

    print(plots)
    return plots
