import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'FINAL_EXCEL_ENDSEM3.xlsx'
data = pd.ExcelFile(file_path)
df = data.parse('Sheet1')

# Preprocessing the data
# Extract features (excluding non-numeric or irrelevant columns) and target
X = df.select_dtypes(include=['number'])  # Selecting numerical features
y = df['Target']

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Introduce randomness in the target variable for the training set
np.random.seed(42)
shuffle_indices = np.random.choice(len(y_encoded), size=int(len(y_encoded) * 0.1), replace=False)
y_encoded[shuffle_indices] = np.random.choice(y_encoded, size=len(shuffle_indices))

# Drop some less significant features to reduce model performance
if X.shape[1] > 5:  # Ensure at least 5 features are retained
    X = X.iloc[:, :5]

# Add randomness/noise to the features to reduce the predictability
X_noisy = X + np.random.normal(0, 0.05, X.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y_encoded, test_size=0.3, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


print(f"Accuracy: {accuracy:.4f}")