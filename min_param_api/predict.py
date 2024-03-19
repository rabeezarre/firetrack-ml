import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import joblib  # for loading the saved models

# Load the new fetched dataset
fetched_data = pd.read_csv('min_param_api/fetched_data.csv')

# Assuming you have saved your trained scaler, KMeans, and RandomForest models
# Load them back
scaler = joblib.load('scaler_model.pkl')
kmeans = joblib.load('kmeans_model.pkl')
clf = joblib.load('random_forest_model.pkl')

# Preprocess the fetched dataset using the loaded scaler
features = fetched_data[['temperature', 'humidity', 'altitude']]
features_scaled = scaler.transform(features)  # Use transform, not fit_transform

# Apply K-Means clustering to assign clusters to the new data
fetched_data['cluster'] = kmeans.predict(features_scaled)

# Map clusters to risk labels (ensure `cluster_to_risk` dictionary is correctly defined as before)
fetched_data['wildfire_risk'] = fetched_data['cluster'].map(cluster_to_risk)

# Now you can use the Random Forest classifier to predict wildfire risk
# This step might be redundant if you're already assigning risk based on clusters, unless you're
# using the Random Forest model for a different set of predictions or further refinement.

# To illustrate, if you were to use the Random Forest Classifier for predictions:
# Ensure your labels are encoded if clf expects numerical labels
# y_pred = clf.predict(features_scaled)
# Then you can compare these predictions to actual labels, if you have them, or use the predictions as is.

# Since it looks like your process is to assign risk based on clusters (and not directly predict with RandomForest),
# the above prediction step with RandomForest might not be needed unless your workflow requires it for other reasons.

# Note: The saving and loading of models (using joblib in this example) is crucial for applying trained models to new data.
# Ensure you have saved your models after training them on your initial dataset.
