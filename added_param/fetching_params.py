import pandas as pd
import numpy as np
import requests
import math
import time

# Function to convert lat/lon to tile coordinates
def lat_lon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x_tile, y_tile

# Function to convert timestamp to Unix time
def timestamp_to_unix(timestamp):
    return int(pd.Timestamp(timestamp).timestamp())

# Load your dataset
df = pd.read_csv('added_param/preprocessed_data.csv')

api_key = "77759dbfac0444c5ab2150250221805"
zoom_level = 5  # Example zoom level

# Placeholder for precipitation data
df['precipitation'] = np.nan

for index, row in df.iterrows():
    lat, lon, timestamp = row['latitude'], row['longitude'], row['timestamp']
    x_tile, y_tile = lat_lon_to_tile(lat, lon, zoom_level)
    unix_timestamp = timestamp_to_unix(timestamp)
    
    url = f"https://maps.openweathermap.org/maps/2.0/radar/{zoom_level}/{x_tile}/{y_tile}?appid={api_key}&tm={unix_timestamp}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        precipitation_value = "extracted_value" 
        
        df.at[index, 'precipitation'] = precipitation_value
    except requests.RequestException as e:
        print(f"Request failed: {e}")
