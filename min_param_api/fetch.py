import requests_cache
import pandas as pd
from retry_requests import retry
import openmeteo_requests

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

latitude = 52.52
longitude = 13.41
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": "2024-03-03",
    "end_date": "2024-03-17",
    "hourly": ["temperature_2m", "relative_humidity_2m"]
}
responses = openmeteo.weather_api(url, params=params)

# Process the first location
response = responses[0]

# Generating pointId
pointId = f"{latitude}_{longitude}" 

altitude = response.Elevation()

# Process hourly data
hourly = response.Hourly()
hourly_data = {
    "timestamp": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ).strftime('%Y-%m-%dT%H:%M:%S.%f'),
    "temperature": hourly.Variables(0).ValuesAsNumpy(),
    "humidity": hourly.Variables(1).ValuesAsNumpy(),
    "altitude": [altitude] * len(hourly.Variables(0).ValuesAsNumpy()),
    "latitude": [latitude] * len(hourly.Variables(0).ValuesAsNumpy()),
    "longitude": [longitude] * len(hourly.Variables(0).ValuesAsNumpy()),
}

hourly_dataframe = pd.DataFrame(data=hourly_data)
hourly_dataframe.insert(0, 'dataId', range(1, 1 + len(hourly_dataframe)))
hourly_dataframe.insert(1, 'pointId', pointId)

print(hourly_dataframe.head())

hourly_dataframe.to_csv('min_param_api/fetched_data.csv', index=False)
