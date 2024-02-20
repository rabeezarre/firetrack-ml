import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset
df = pd.read_csv('processed_data.csv')

# Step 2: Data exploration and preprocessing
# Note: Adjust preprocessing steps based on actual dataset exploration results
# Example preprocessing steps:
# Impute missing values if any
imputer = SimpleImputer(strategy='mean')
df_filled = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['float64', 'int64'])))
df_filled.columns = df.select_dtypes(include=['float64', 'int64']).columns
df_filled.index = df.index

# Encode categorical variables if necessary
# Scale/normalize numerical features if necessary
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_filled), columns=df_filled.columns)

# Assuming 'Risk' is the target variable
X = df_scaled.drop('Risk', axis=1)
y = df['Risk']

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model selection
model = RandomForestClassifier(random_state=42)

# Step 5: Training the model
model.fit(X_train, y_train)

# Step 6: Model evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
