from math import ceil
import sqlite3
from matplotlib.pyplot import sca
from datetime import datetime
import numpy as np
import pandas as pd
from collections import OrderedDict
from numpy.core.fromnumeric import mean
import logging


def connect_to_db(db_name):
    """
    Connect to the database of the application
    """
    try:
        connection = sqlite3.connect("../"+db_name)
        # Creating a cursor
        cursor = connection.cursor()
        msg = "Connected to database "+str(db_name)
        logging.log(logging.INFO, msg)
        return connection, cursor
    
    except:
        logging.log(logging.ERROR, "Could not find database")


# Create a table
def create_table(cursor, table_name, fields_dict, drop_existing_table):
    
    if drop_existing_table==True:
        cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
        logging.log(logging.WARN, "Dropped existing table")
    
    fields_strings = [str(key)+" "+str(value['type']) for key, value in fields_dict.items()]
    fields_text = "("+",".join(fields_strings)+")"
    table_create_string = f'CREATE TABLE IF NOT EXISTS {table_name}'
    db_command = f"""{table_create_string} {fields_text}"""

    try:
        cursor.execute(db_command)
        logging.log(logging.INFO, "Table created successfully")
    except:
        logging.log(logging.ERROR, "Unable to create database table")


def gen_data(specs, size, coeffs):

    '''
    Data generating function for continuous, discrete data, dates and device data
    '''    
    #print(specs)
    
    if specs["kind"] == "continuous":
        return np.random.normal(
            loc=specs['loc'],
            scale=specs['scale'],
            size=size
            )
    elif specs["kind"] == "device" and specs["devices_up"] == "all":
        return np.random.choice(specs["devices_list"])
    
    elif specs["kind"] == "dates":
        return pd.date_range(start=specs["start"], periods=size, )
    
    elif specs["kind"] == "device_status":
        return np.random.choice(specs['values'])

    elif specs["kind"] in ["categorical", "numeric"]:
        logging.log(logging.INFO, "Target variables are generated using generated data")

    else:
        logging.log(logging.ERROR, "Could not generate data, check input")


def generate_data(size, fields_dict, start_date, coeffs):
    
    # populate a data dictionary with all the generated values, which then gets converted to a dataframe
    data_dict = {}
    
    for field, specs in fields_dict.items():
        data_dict[field] = gen_data(specs, size, coeffs)
    

    if fields_dict["y_num"]["measure"] == "sum":
        data_dict["y_num"] = coeffs[0]*data_dict["x1"] + coeffs[1]*data_dict["x2"] + np.random.normal(loc=0, scale=0.2, size=size)
    elif fields_dict["y_num"]["measure"] == "prod":
        data_dict["y_num"] = coeffs[0]*data_dict["x1"] * coeffs[1]*data_dict["x2"] * np.random.normal(loc=0, scale=0.2, size=size)
    else:
        logging.log(logging.ERROR, "Unable to create target variable, check input")
    
    if fields_dict["y_cat"]["measure"] == "mean":
        data_dict["y_cat"] = data_dict["y_num"] > np.mean(data_dict['y_num'])
    elif fields_dict["y_cat"]["measure"] == "median":
        data_dict["y_cat"] = data_dict["y_num"] > np.median(data_dict['y_num'])        
    else:
        logging.log(logging.ERROR, "Unable to create target variable, check input")
    
    data_df = pd.DataFrame(data_dict)

    return data_df


def populate_database(df, table_name, connection):
    
    df.to_sql(name=table_name, con=connection, if_exists="replace")

    logging.log(logging.INFO, f"Added {len(df)} rows to the database")


if __name__ == '__main__':
    connection, cursor = connect_to_db("sensor_data.db")

    sample_fields_dict = OrderedDict({
        "x1": { 
            "kind": "continuous",
            "type": "REAL",
            "loc": 10,
            "scale": 1,
            },
        "x2": {
            "kind": "continuous",
            "type": "REAL",
            "loc":   10,
            "scale":  2,
            },
        "devices": {
            "kind": "device",
            "type": "TEXT",
            "devices_list": ["001", "002", "003", "004"],
            "devices_up": "all"
            },
        "timestamp": {
            "kind": "dates",
            "type": "TEXT",
            "start": "2020-01-01",
            "freq":   "d"
            },
        "y_num": {
            "kind": "numeric",
            "type": "REAL",
            "var1": "x1",
            "var2": "x2",
            "measure": "sum"
        },
        "y_cat": {
            "kind": "categorical",
            "type": "TEXT",
            "var1": "x1",
            "var2": "x2",
            "measure": "mean"
        },
        "device_status":{
            "kind": "device_status",
            "type": "INTEGER",
            "values": [0,1]
        }
    })

    table_name = "turbine_sensor_data"

    create_table(cursor, table_name, sample_fields_dict, drop_existing_table=True)
    data_df = generate_data(100, sample_fields_dict, "2010-01-01", [1,2])
    populate_database(data_df, table_name, connection)