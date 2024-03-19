import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Cluster 0 -> Medium risk: Higher altitude, moderate temperature, and humidity
# Cluster 1 -> High risk: Lower altitude, higher temperature, and lower humidity
# Cluster 2 -> Low risk: Moderate altitude, lower temperature, and higher humidity

cluster_to_risk = {
    0: 'Medium',
    1: 'High',
    2: 'Low'
}

fetched_data = pd.read_csv('min_param_api/fetched_data.csv')

scaler = joblib.load('min_param_api/scaler_model.pkl')
kmeans = joblib.load('min_param_api/kmeans_model.pkl')
clf = joblib.load('min_param_api/random_forest_model.pkl')

# Preprocess the fetched dataset using the loaded scaler
features = fetched_data[['temperature', 'humidity', 'altitude']]
features_scaled = scaler.transform(features)  # Use transform, not fit_transform

fetched_data['cluster'] = kmeans.predict(features_scaled)
fetched_data['wildfire_risk'] = fetched_data['cluster'].map(cluster_to_risk)

# Summary of predicted risks
risk_summary = fetched_data['wildfire_risk'].value_counts()
print("Summary of Predicted Wildfire Risks:")
print(risk_summary)

# Visualize the distribution of wildfire risks
plt.figure(figsize=(8, 6))
sns.countplot(data=fetched_data, x='wildfire_risk', hue='wildfire_risk', dodge=False, palette='viridis', order=fetched_data['wildfire_risk'].value_counts().index)
plt.title('Distribution of Predicted Wildfire Risks')
plt.xlabel('Wildfire Risk')
plt.ylabel('Count')
plt.legend(title='Wildfire Risk', loc='upper right', labels=fetched_data['wildfire_risk'].value_counts().index)
plt.show()

# Detailed analysis
mean_values = fetched_data.groupby('wildfire_risk')[['temperature', 'humidity', 'altitude']].mean()
print("\nMean Temperature, Humidity, and Altitude by Wildfire Risk:")
print(mean_values)

# Geographical distribution of predicted wildfire risks
if 'latitude' in fetched_data.columns and 'longitude' in fetched_data.columns:
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='longitude', y='latitude', hue='wildfire_risk', data=fetched_data, palette='viridis', s=100)
    plt.title('Geographical Distribution of Predicted Wildfire Risks')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Wildfire Risk')
    plt.grid(True)
    plt.show()

fetched_data.to_csv('min_param_api/fetched_data_with_predictions.csv', index=False)
