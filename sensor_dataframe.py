import pandas as pd
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

def generate_stats(sensor_df):
    '''
    Build summary statistics for each sensor in the dataframe
    '''
    print(sensor_df.timestamp)
    unique_sensor_ids = sensor_df.device_id.unique().tolist()
    device_wise_dfs = {}
    device_wise_stats = {}
    for sensor_id in unique_sensor_ids:
        device_wise_dfs.update({ sensor_id: sensor_df[sensor_df['device_id']==sensor_id] })

    for sensor_id, device_df in device_wise_dfs.items():
        
        start_date =    pd.to_datetime(device_df.timestamp).min()
        end_date =      pd.to_datetime(device_df.timestamp).max()
        mean =          device_df.value.mean()
        median =        device_df.value.median()
        stdev =         device_df.value.std()
        kurtosis =      device_df.value.kurtosis()
        skewness =      device_df.value.skew()
        
        device_stats = {
            sensor_id: {
                "start_date":   start_date,
                "end_date":     end_date,
                "mean":         mean,
                "median":       median,
                "stdev":        stdev,
                "kurtosis":     kurtosis,
                "skewness":     skewness,
            }
        }
        
        device_wise_stats.update(device_stats)

    #print(pd.DataFrame.from_dict(device_wise_stats, orient='index'))
    result = pd.DataFrame.from_dict(device_wise_stats, orient='index')
    result.index = result.index.rename("sensor_id")
    return result



def transform_df():


    pass
