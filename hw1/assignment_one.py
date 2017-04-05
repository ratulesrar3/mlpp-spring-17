# MLPP 2017 Assignment 1
#
# Ratul Esrar, ratulesrar3

import numpy as np
import pandas as pd
import datetime
import urllib

# Query for Graffiti Removal dataset
RAW_GRAFFITI = ("https://data.cityofchicago.org/resource/cdmx-wzbz.json")

RAW_BUILDINGS = ("https://data.cityofchicago.org/resource/nht9-a9dz.json")

RAW_SANITATION = ("https://data.cityofchicago.org/resource/qhpu-7pmc.json")

RAW_POTHOLES = ("https://data.cityofchicago.org/resource/cu76-24fu.json")

def read_data(query):
	raw_data = pd.read_json(query)
	return raw_data