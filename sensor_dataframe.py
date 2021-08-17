from typing import OrderedDict
import pandas as pd
import numpy as np
import sqlite3


# Verify that result of SQL query is stored in the dataframe
def return_df_from_db():
    # Read sqlite query results into a pandas DataFrame
    connection = sqlite3.connect("sensor_data.db")
    sensor_df = pd.read_sql_query("SELECT * from sensor_data", connection)

    try:
        print(sensor_df.head())
        print("Data from the sensor database successful imported into dataframe")
    except:
        raise("Error: unable to read data from database into dataframe")
    connection.close()

    return sensor_df

# From this point on, we are using the data in the dataframe for various transformations and the like

def compute_stats(df, col):
    mean = df[col].mean()
    median = df[col].median()
    stdev = df[col].std()
    kurtosis = df[col].kurtosis()
    skewness = df[col].skew()

    stats = {
        col+"_mean": mean,
        col+"_median": median,
        col+"_stdev": stdev,
        col+"_kurtosis": kurtosis,
        col+"_skewness": skewness
    }

    return stats


def generate_stats(sensor_df, metric_columns):
    '''
    Build summary statistics for each sensor in the dataframe
    '''
    print(pd.to_datetime(sensor_df.timestamp))
    print("Printed time stamps")
    unique_sensor_ids = sensor_df.device_id.unique().tolist()
    device_wise_dfs = {}
    device_wise_stats = {}
    for sensor_id in unique_sensor_ids:
        device_wise_dfs.update({ sensor_id: sensor_df[sensor_df['device_id']==sensor_id] })

    for sensor_id, device_df in device_wise_dfs.items():
        
        start_date =    pd.to_datetime(device_df.timestamp).min().date()
        end_date =      pd.to_datetime(device_df.timestamp).max().date()
        
        metric_stats = OrderedDict({
                "start_date":   start_date,
                "end_date":     end_date,
            })

        for metric_col in metric_columns:
            metric_stats.update(compute_stats(sensor_df, metric_col))
        
               
        device_stats = {sensor_id: metric_stats}
        
        
        device_wise_stats.update(device_stats)
        

    print ("Device wise stats")
    print(device_wise_stats)
    #print(pd.DataFrame.from_dict(device_wise_stats, orient='index'))
    result = pd.DataFrame.from_dict(device_wise_stats, orient='index')
    result.index = result.index.rename("sensor_id")
    return result



def transform_df(sensor_df, transforms):

    '''
    Generate transformations based on the "transforms" dictionary and add these to the parent dataframe
    '''


    for col, transform in transforms.items():
        if transform == "none":
            continue
        elif transform == "log":
            transformed_column = np.log(sensor_df[col].astype(float)).rename("log_"+col)
            sensor_df = sensor_df.merge(transformed_column, left_index=True, right_index=True, how="left")
        elif transform == "sqrt":
            transformed_column = np.sqrt(sensor_df[col].astype(float)).rename("sqrt_"+col)
            sensor_df = sensor_df.merge(transformed_column, left_index=True, right_index=True, how="left")
        elif transform == "exp":
            transformed_column = np.exp(sensor_df[col].astype(float)).rename("exp_"+col)
            sensor_df = sensor_df.merge(transformed_column, left_index=True, right_index=True, how="left")
    
    return sensor_df
