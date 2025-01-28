import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 1000

# Generate synthetic data
data = {
    'Age': np.random.randint(18, 65, num_samples),
    'Employment_Years': np.random.randint(1, 40, num_samples),
    'Income': np.random.randint(5000, 50000, num_samples),
    'Debt': np.random.randint(0, 25000, num_samples),
    'Marital_Status': np.random.randint(0, 2, num_samples),
    'Education_Level': np.random.randint(0, 2, num_samples),
    'Utility_Payment_History': np.random.randint(0, 2, num_samples),
    'Mobile_Phone_Usage': np.random.randint(500, 5000, num_samples),
    'Target': np.random.randint(0, 2, num_samples)
}

# Create a DataFrame
df_unorganized = pd.DataFrame(data)

# Save the dataset to a CSV file (optional)
df_unorganized.to_csv('credit_risk_unorganized.csv', index=False)

# Display the first few rows of the dataset
print(df_unorganized.head())

# Features and target
X = df_unorganized.drop('Target', axis=1)  # Features
y = df_unorganized['Target']              # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display the classification report
print(classification_report(y_test, y_pred))

# Save the trained model and scaler using joblib
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
