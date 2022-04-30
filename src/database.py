from math import ceil
import sqlite3
from matplotlib.pyplot import sca
from datetime import datetime
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import mean

# connection = sqlite3.connect(':memory:')

def connect_to_db(db_name):
    """
    Connect to the database of the application
    """
    connection = sqlite3.connect('sensor_data.db')
    # Creating a cursor
    cursor = connection.cursor()

    return cursor


# Create a table

def create_table(cursor, table_name, fields_dict):
    
    fields_strings = [str(key)+" "+str(value) for key, value in enumerate(fields_dict))]
    
    fields_text = "("+fields_strings.join(",")+")"
    
    table_create_string = f'CREATE TABLE IF NOT EXISTS {table_name}'
    
    db_command = f"""{table_create_string} {fields_text}"""
    
    cursor.execute(db_command)

def generate_data(size, fields_dict, start_date, coeffs):
    
    data_dict = {}
    for field, specs in fields_dict:
        data_dict[field] = gen_data(specs, size, coeffs)
    
    data_df = pd.DataFrame(data_dict)

    return data_df

sample_fields_dict = {
    "x1": { 
        "kind": "continuous",
        "loc": 10,
        "scale":    1
        },
    "x2": {
        "kind":     "continuous",
        "loc":      10,
        "scale":    2,
        },
    "devices": {
        "kind": "device",
        "devices_list": ["001", "002", "003", "004"],
        "devices_up": "all"
        },
    "dates": {
        "start": "2020-01-01",
        "freq":   "d"
        },
}


def gen_data(specs, size, coeffs):
    
    x1 = np.random.normal(loc = 10, scale= 1, size=SIZE)
    x2 = np.random.normal(loc = 10, scale= 2, size=SIZE)
    coeff1, coeff2 = 0.5, -1.5
    device_status = np.repeat(1,SIZE)
    continuous_target = coeff1*x1 + coeff2*x2 + np.random.normal(loc=0, scale=0.2, size=SIZE)
    categorical_target = continuous_target > np.mean(continuous_target)
    dates = pd.date_range(start = "2010-01-01", periods= SIZE, freq="d")
    device_ids = np.random.choice(["001", "002", "003", "004"], size = SIZE, replace=True)

many_records_df = pd.DataFrame({
    "timestamp": [x.strftime('%Y-%m-%d') for x in dates],
    "device_id": device_ids,
    "device_status": device_status,
    "x1": x1,
    "x2": x2,
    "y_num": continuous_target,
    "y_cat": categorical_target
})

many_entries = many_records_df.to_records(index=False).tolist()

#print(many_entries)

'''
many_entries = [
    ('2021-01-01', '001', 1, 0.1 ),
    ('2021-01-02', '001', 1, 0.1 ),
    ('2021-01-03', '001', 1, 0.1 ),
    ('2021-01-04', '001', 1, 0.1 ),
    ('2021-01-05', '001', 1, 0.1 ),
    ('2021-01-01', '002', 1, 0.5 ),
    ('2021-01-02', '002', 1, 0.5 ),
    ('2021-01-03', '002', 1, 0.5 ),
    ('2021-01-04', '002', 1, 0.5 ),
    ('2021-01-05', '002', 1, 0.5 ),
    ('2021-01-01', '003', 1, 0.2 ),
    ('2021-01-02', '003', 1, 0.2 ),
    ('2021-01-03', '003', 1, 0.2 ),
    ('2021-01-04', '003', 1, 0.2 ),
    ('2021-01-05', '003', 1, 0.2 ),
    ('2021-01-01', '004', 1, 1.5 ),
    ('2021-01-02', '004', 1, 1.5 ),
    ('2021-01-03', '004', 1, 1.5 ),
    ('2021-01-04', '004', 1, 1.5 ),
    ('2021-01-05', '004', 1, 1.5 )
]
'''

cursor.executemany("INSERT INTO sensor_data (timestamp, device_id, device_status, x1, x2, y_num, y_cat) VALUES (?, ?, ?, ?, ?, ?, ?)", many_entries)

print(f"Added {cursor.rowcount} rows to the database")

connection.commit()