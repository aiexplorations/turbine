import sqlite3
from matplotlib.pyplot import sca
from datetime import datetime
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import mean

# connection = sqlite3.connect(':memory:')

connection = sqlite3.connect('sensor_data.db')

# Creating a cursor
cursor = connection.cursor()

# Create a table

cursor.execute("""
CREATE TABLE IF NOT EXISTS sensor_data 
(
    timestamp TEXT,
    device_id TEXT,
    device_status INTEGER,
    x1 REAL,
    x2 REAL,
    target REAL
)""")

SIZE = 1000

x1 = np.random.normal(loc = 10, scale= 2, size=SIZE)
x2 = np.random.normal(loc = 10, scale= 3, size=SIZE)
coeff1, coeff2 = 1.5, -2.5
device_status = np.repeat(1,SIZE)
target = coeff1*x1 + coeff2*x2 + np.random.normal(loc=0, scale=0.2, size=SIZE)
dates = pd.date_range(start = "2010-01-01", periods= SIZE, freq="d")
device_ids = np.random.choice(["001", "002", "003", "004"], size = SIZE, replace=True)

many_records_df = pd.DataFrame({
    "timestamp": [x.strftime('%Y-%m-%d') for x in dates],
    "device_id": device_ids,
    "device_status": device_status,
    "x1": x1,
    "x2": x2,
    "target": target
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

cursor.executemany("INSERT INTO sensor_data (timestamp, device_id, device_status, x1, x2, target) VALUES (?, ?, ?, ?, ?, ?)", many_entries)

print(f"Added {cursor.rowcount} rows to the database")

connection.commit()