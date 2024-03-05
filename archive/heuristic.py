import pandas as pd

# Load the dataset from a CSV file
file_path = 'processed_data.csv'
data = pd.read_csv(file_path)

# Define thresholds for categorizing wildfire risk
high_temp_threshold = 30  # degrees Celsius
low_humidity_threshold = 30  # percent

# Function to categorize wildfire risk based on temperature and humidity
def categorize_risk(row):
    if row['temperature'] >= high_temp_threshold and row['humidity'] <= low_humidity_threshold:
        return 'High'
    elif row['temperature'] < high_temp_threshold and row['humidity'] > low_humidity_threshold:
        return 'Low'
    else:
        return 'Medium'

# Apply the function to each row in the dataset to determine wildfire risk
data['wildfire_risk'] = data.apply(categorize_risk, axis=1)

# Save the updated dataset with risk categorization to a new CSV file (optional)
updated_file_path = 'heuristic_dataset.csv'  # Update this to your desired output file path
data.to_csv(updated_file_path, index=False)

# Display the first few rows of the updated dataframe
print(data.head())
