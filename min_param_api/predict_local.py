import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the new dataset
new_data = pd.read_csv('min_param_api/fetched_data.csv')

# Step 2: Preprocess the dataset
features = new_data[['temperature', 'humidity', 'altitude']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 3: Apply K-Means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
new_data['cluster'] = kmeans.fit_predict(features_scaled)

# Analyze the cluster centroids (optional analysis)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids, columns=['temperature', 'humidity', 'altitude'])
print("Cluster centroids:")
print(centroids_df)

# Labeling clusters based on environmental factors
# Adjust these labels based on your specific analysis and understanding of the data
centroids_df['wildfire_risk'] = pd.cut(centroids_df['temperature'], 
                                        bins=3, 
                                        labels=['Medium', 'High', 'Low'])
risk_labels = centroids_df.sort_values(by='temperature')['wildfire_risk'].values
cluster_to_risk = {i: risk for i, risk in enumerate(risk_labels)}
new_data['wildfire_risk'] = new_data['cluster'].map(cluster_to_risk)

# Step 4: Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=new_data, x='temperature', y='humidity', hue='wildfire_risk', palette='viridis')
plt.title('Temperature vs. Humidity by Wildfire Risk Label')
plt.show()

# Step 5: Preparing data for training the model
X = features
y = new_data['wildfire_risk'].astype('category').cat.codes  # Encoding the labels to numeric values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Training a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 7: Predicting and evaluating the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
