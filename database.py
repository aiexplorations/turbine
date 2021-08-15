import sqlite3

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
    value REAL

)""")



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


cursor.executemany("INSERT INTO sensor_data (timestamp, device_id, device_status, value) VALUES (?, ?, ?, ?)", many_entries)

print("Added rows to the database")

connection.commit()