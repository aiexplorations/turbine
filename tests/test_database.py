import unittest
import pandas as pd
from collections import OrderedDict
from src.database import generate_data, connect_to_db, create_table, populate_database
import logging


class TestDatabase(unittest.TestCase):

    def test_generate_data(self):
        size = 10
        fields_dict = OrderedDict({
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

        start_date = "2012-01-01"
        coeffs = [10,20]
        try:
            data_df = generate_data(size, fields_dict, start_date, coeffs)
            logging.log(logging.INFO, "Able to generate data")
            logging.log(logging.INFO, data_df.head())
        except:
            raise(Exception("Unable to generate data"))