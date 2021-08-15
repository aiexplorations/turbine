import pandas as pd
import sqlite3

# Read sqlite query results into a pandas DataFrame
connection = sqlite3.connect("sensor_data.db")
sensor_df = pd.read_sql_query("SELECT * from sensor_data", connection)

# Verify that result of SQL query is stored in the dataframe
try:
    print(sensor_df.head())
    print("Data from the sensor database successful imported into dataframe")
except:
    raise("Error: unable to read data from database into dataframe")

connection.close()

print(sensor_df.dtypes)

# From this point on, we are using the data in the dataframe for various transformations and the like


