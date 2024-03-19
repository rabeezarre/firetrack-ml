import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('min_param/raws.csv')

# Convert 'ObservedDate' to the required datetime format
df['ObservedDate'] = pd.to_datetime(df['ObservedDate'], format='%Y/%m/%d %H:%M:%S+00').dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

# Replace 'NO DATA' and 'NaN' with np.nan to handle missing values before conversion
df.replace({'NO DATA': pd.NA, 'NaN': pd.NA, ' deg. F': '', '%': ''}, inplace=True, regex=True)

# Safely convert 'AirTempStandPlace' to float
df['AirTempStandPlace'] = pd.to_numeric(df['AirTempStandPlace'], errors='coerce')

# Convert temperature from Fahrenheit to Celsius
df['AirTempStandPlace'] = ((df['AirTempStandPlace'] - 32) * 5/9).round(4)

# Convert 'RelativeHumidity' to float
df['RelativeHumidity'] = pd.to_numeric(df['RelativeHumidity'], errors='coerce')

# Select and rename the required columns
processed_df = df[['OBJECTID', 'ObservedDate', 'AirTempStandPlace', 'RelativeHumidity', 'Elevation']].copy()
processed_df.rename(columns={
    'OBJECTID': 'pointId',
    'ObservedDate': 'timestamp',
    'AirTempStandPlace': 'temperature',
    'RelativeHumidity': 'humidity',
    'Elevation': 'altitude'
}, inplace=True)

# Remove all rows with any empty values
processed_df.dropna(inplace=True)

# Add 'dataId' as an autogenerated index starting from 1
processed_df.reset_index(drop=True, inplace=True)
processed_df.index += 1
processed_df['dataId'] = processed_df.index

# Reorder the columns to match the desired output
final_processed_df = processed_df[['dataId', 'pointId', 'timestamp', 'temperature', 'humidity', 'altitude']]

final_processed_df.to_csv('min_param/processed_data.csv', index=False)
