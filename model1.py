import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = 'raws.csv'
data = pd.read_csv(file_path)

# Selecting and preprocessing the data
selected_columns = ['RainAccumulation', 'WindSpeedMPH', 'AirTempStandPlace', 'RelativeHumidity']
df_selected = data[selected_columns].copy()

# Replace 'NO DATA' with NaN to allow conversion
df_selected = df_selected.replace('NO DATA', float('nan'))

# Convert selected columns to numeric, stripping units
df_selected['RainAccumulation'] = df_selected['RainAccumulation'].str.replace(' inches', '').astype(float)
df_selected['WindSpeedMPH'] = df_selected['WindSpeedMPH'].str.replace(' mph', '').astype(float)
df_selected['AirTempStandPlace'] = df_selected['AirTempStandPlace'].str.replace(' deg. F', '').astype(float)
df_selected['RelativeHumidity'] = df_selected['RelativeHumidity'].str.replace('%', '').astype(float)

# Filling missing values with the median of each column
df_selected.fillna(df_selected.median(), inplace=True)

# Creating a dummy binary target variable for demonstration
median_temp = df_selected['AirTempStandPlace'].median()
df_selected['WildfireRisk'] = (df_selected['AirTempStandPlace'] > median_temp).astype(int)

# Split the data into features and target variable
X = df_selected.drop('WildfireRisk', axis=1)
y = df_selected['WildfireRisk']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Predict on the testing set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Display the model performance
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report_result)
