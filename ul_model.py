import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'processed_data.csv'
data = pd.read_csv(file_path)

# Preprocess the data: Select and scale relevant features
features = data[['temperature', 'humidity', 'altitude']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply K-Means clustering to infer wildfire risk categories
kmeans = KMeans(n_clusters=4, random_state=42)
data['cluster'] = kmeans.fit_predict(features_scaled)

# Assign risk levels based on clusters
risk_labels = {0: 'Higher Risk', 1: 'Lower Risk', 2: 'Medium Risk', 3: 'Medium to Low Risk'}
data['wildfire_risk'] = data['cluster'].map(risk_labels)

# Prepare data for model training
X = data[['temperature', 'humidity', 'altitude']]
y = data['wildfire_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print model performance
print(f"Model Accuracy: {accuracy*100:.2f}%")
print("Classification Report:")
print(report)
