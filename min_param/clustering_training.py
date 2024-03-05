import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the new dataset
new_data = pd.read_csv('processed_data.csv')

# Preprocess the dataset
features = new_data[['temperature', 'humidity', 'altitude']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply K-Means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
new_data['cluster'] = kmeans.fit_predict(features_scaled)

# Analyze the cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids, columns=['temperature', 'humidity', 'altitude'])
print(centroids_df)

#   temperature   humidity     altitude
# 0     1.076485  69.164681  6546.063712   //medium
# 1    15.723165  45.911864  1375.912994
# 2     2.151868  82.814414  2317.348777

# Labeling clusters based on environmental factors
centroids_df['wildfire_risk'] = pd.cut(centroids_df['temperature'], 
                                        bins=3, 
                                        labels=['Medium', 'High', 'Low'])
risk_labels = centroids_df.sort_values(by='temperature')['wildfire_risk'].values
cluster_to_risk = {i: risk for i, risk in enumerate(risk_labels)}
new_data['wildfire_risk'] = new_data['cluster'].map(cluster_to_risk)

# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=new_data, x='temperature', y='humidity', hue='wildfire_risk', palette='viridis')
plt.title('Temperature vs. Humidity by Wildfire Risk Label')
plt.show()

# Preparing data for training the model
X = features
y = new_data['wildfire_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))