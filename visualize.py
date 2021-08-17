from typing import OrderedDict
from sensor_dataframe import return_df_from_db
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns



def visualize_data(sensor_df, metric_columns, kind):
    unique_sensor_ids = sensor_df.device_id.unique().tolist()
    device_wise_df = {}
    for sensor_id in unique_sensor_ids:
        device_wise_df.update({sensor_id: sensor_df[sensor_df['device_id']==sensor_id]})
    
    plots = OrderedDict()

    #print(unique_sensor_ids)    
    # Generate run charts for each sensor ID
    for metric_column in metric_columns:

        for sensor_id in unique_sensor_ids:

            plots[str(metric_column) + "_" + str(sensor_id)+ "_" + "chart"] = plt.figure(figsize = (16,4))
            
            if kind == "run chart":
                plt.title(f"Run chart for {metric_column} sensor id : {sensor_id}")
                plt.plot(device_wise_df[sensor_id][metric_column])
            elif kind == "histogram":
                plt.title(f"Histogram for {metric_column} sensor id: {sensor_id}")
                plt.hist(device_wise_df[sensor_id][metric_column], bins = 20)


    print(plots)
    return plots

