import os
import sys
sys.path.append("..")

import numpy as np

import json

A2D2_LOCATION = '/mnt/ml-data-storage/jens/A2D2/' 

DATA_DIR = '/mnt/ml-data-storage/jens/A2D2/'
IMAGE_DIRS = os.path.join(DATA_DIR, 'camera_lidar_semantic')

from geopy.geocoders import Nominatim
from IPython.display import clear_output
import time 

geolocator = Nominatim(user_agent='A2D2-data')

runs = os.listdir(DATA_DIR)
for r in runs: 
    if os.path.isdir(os.path.join(DATA_DIR,r)) and r != 'camera_lidar_semantic': 
        bus_file = os.path.join(DATA_DIR, r, 'bus', r.replace('_', '')) + '_bus_signals.json'
        with open(bus_file) as f: 
            data = json.load(f)
            for i in range(len(data)):
                if 'latitude_degree' in data[i]['flexray'].keys():
                    lat, long = data[i]['flexray']['latitude_degree']['values'], data[i]['flexray']['longitude_degree']['values']
                    location = geolocator.reverse(str(lat[0])+ ","+str(long[0]))
                    try:
                        city = location.raw['address']
                        pos = city['city']
                    except:
                        try: 
                            pos = city['village']
                        except:
                            #print(location.raw['address'])
                            continue
                            #try: 
                            #    pos = city['town']
                            #except: 
                            #    continue
                    print(r, pos, "postcode:", location.raw['address']['postcode'])
                    break 
