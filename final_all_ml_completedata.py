import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.decomposition import PCA 
from mpl_toolkits.mplot3d import Axes3D
 # Import PCA here
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Function to evaluate and display results
def evaluate_model(y_test, y_pred, label_encoder, model_name):
    print(f"\n=== {model_name} Evaluation ===")
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

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
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# Load and preprocess the dataset
file_path = "FINAL_EXCEL_ENDSEM3.xlsx"
data = pd.ExcelFile(file_path)
df = data.parse("Sheet1")

# Extract features (numerical columns) and target
X = df.select_dtypes(include=["number"])
y = df["Target"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Introduce randomness in the target variable for 10% of the data
np.random.seed(42)
shuffle_indices = np.random.choice(len(y_encoded), size=int(len(y_encoded) * 0.1), replace=False)
y_encoded[shuffle_indices] = np.random.choice(y_encoded, size=len(shuffle_indices))

# Select top 5 features if more than 5 exist
if X.shape[1] > 5:
    X = X.iloc[:, :5]

# Add noise to features
X_noisy = X + np.random.normal(0, 0.05, X.shape)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y_encoded, test_size=0.3, random_state=42)

# Standardize feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Model 1: SVM ===
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
evaluate_model(y_test, y_pred_svm, label_encoder, "Support Vector Machine (SVM)")

# === Model 2: Random Forest ===
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
evaluate_model(y_test, y_pred_rf, label_encoder, "Random Forest Classifier")

# === Model 3: Logistic Regression ===
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)
y_pred_lr = logistic_model.predict(X_test_scaled)
evaluate_model(y_test, y_pred_lr, label_encoder, "Logistic Regression")

# === Model 4: XGBoost ===
xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    random_state=42,
    eval_metric="mlogloss",
)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
evaluate_model(y_test, y_pred_xgb, label_encoder, "XGBoost Classifier")

# Store metrics for comparison
metrics = {
    "Model": ["SVM", "Random Forest", "Logistic Regression", "XGBoost"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_xgb),
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_svm, average="weighted"),
        f1_score(y_test, y_pred_rf, average="weighted"),
        f1_score(y_test, y_pred_lr, average="weighted"),
        f1_score(y_test, y_pred_xgb, average="weighted"),
    ],
    "Precision": [
        precision_score(y_test, y_pred_svm, average="weighted"),
        precision_score(y_test, y_pred_rf, average="weighted"),
        precision_score(y_test, y_pred_lr, average="weighted"),
        precision_score(y_test, y_pred_xgb, average="weighted"),
    ],
    "Recall": [
        recall_score(y_test, y_pred_svm, average="weighted"),
        recall_score(y_test, y_pred_rf, average="weighted"),
        recall_score(y_test, y_pred_lr, average="weighted"),
        recall_score(y_test, y_pred_xgb, average="weighted"),
    ],
}

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(metrics)


# Grouped bar chart with annotations
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.2
index = np.arange(len(metrics_df))

# Plot each metric
for i, metric in enumerate(["Accuracy", "F1 Score", "Precision", "Recall"]):
    ax.bar(
        index + i * bar_width,
        metrics_df[metric],
        bar_width,
        label=metric,
    )

# Add annotations
for i, metric in enumerate(["Accuracy", "F1 Score", "Precision", "Recall"]):
    for j, value in enumerate(metrics_df[metric]):
        ax.text(
            j + i * bar_width,
            value + 0.01,
            f"{value:.2f}",
            ha="center",
            fontsize=10,
        )


# Add legend and labels
ax.set_title("Model Performance Comparison", fontsize=16)
ax.set_xlabel("Models", fontsize=14)
ax.set_ylabel("Scores", fontsize=14)
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(metrics_df["Model"], fontsize=12)
ax.legend()
plt.show()




from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize

# Binarize the labels for multiclass
y_test_binarized = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))

plt.figure(figsize=(10, 8))

# Loop through models and calculate ROC for each
for model_name, model in [
    ("SVM", svm_model),
    ("Random Forest", rf_model),
    ("Logistic Regression", logistic_model),
    ("XGBoost", xgb_model),
]:
    if hasattr(model, "predict_proba"):  # Check if model supports predict_proba
        y_prob = model.predict_proba(X_test_scaled)
    else:  # For SVM, use decision_function
        y_prob = model.decision_function(X_test_scaled)
        # Ensure y_prob is properly shaped for multiclass
        if len(label_encoder.classes_) > 2:
            y_prob = np.column_stack([1 - y_prob, y_prob]) if y_prob.ndim == 1 else y_prob

    # Calculate ROC for each class and micro-average
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(label_encoder.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
        roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_prob[:, i])

    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_prob.ravel())
    roc_auc["micro"] = roc_auc_score(y_test_binarized, y_prob, average="micro")

    # Plot micro-average ROC curve
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"{model_name} (AUC = {roc_auc['micro']:.2f})",
    )

# Plot aesthetics
plt.title("ROC Curve Comparison (Micro-Average)", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend(fontsize=12)
plt.grid()
plt.show()



# Feature importance for Random Forest
plt.figure(figsize=(8, 6))
importances = rf_model.feature_importances_
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Feature importance for XGBoost
xgb_importances = xgb_model.feature_importances_
plt.figure(figsize=(8, 6))
sns.barplot(x=xgb_importances, y=X.columns)
plt.title("Feature Importance - XGBoost")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Reduce to 3 dimensions for 3D visualization
pca_3d = PCA(n_components=3)
X_train_pca_3d = pca_3d.fit_transform(X_train_scaled)
X_test_pca_3d = pca_3d.transform(X_test_scaled)

# Fit SVM model on 3D reduced data
svm_model_pca_3d = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_model_pca_3d.fit(X_train_pca_3d, y_train)

# Create a 3D grid for decision boundary
x_min, x_max = X_train_pca_3d[:, 0].min() - 1, X_train_pca_3d[:, 0].max() + 1
y_min, y_max = X_train_pca_3d[:, 1].min() - 1, X_train_pca_3d[:, 1].max() + 1
z_min, z_max = X_train_pca_3d[:, 2].min() - 1, X_train_pca_3d[:, 2].max() + 1

xx, yy, zz = np.meshgrid(
    np.arange(x_min, x_max, 0.5),
    np.arange(y_min, y_max, 0.5),
    np.arange(z_min, z_max, 0.5),
)

grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
Z = svm_model_pca_3d.predict(grid)
Z = Z.reshape(xx.shape)

# Plot the decision boundary in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot the grid points and decision boundary
ax.contourf(xx[:, :, 0], yy[:, :, 0], Z[:, :, 0], zdir='z', offset=z_min - 1, alpha=0.5, cmap="coolwarm")

# Scatter the training points
scatter = ax.scatter(
    X_train_pca_3d[:, 0],
    X_train_pca_3d[:, 1],
    X_train_pca_3d[:, 2],
    c=y_train,
    cmap="coolwarm",
    edgecolor="k",
    s=40,
)



# Labels and title
ax.set_title("SVM Decision Boundary (3D PCA-reduced Data)", fontsize=16)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

# Add legend for classes
legend1 = ax.legend(*scatter.legend_elements(), title="Classes", loc="upper left")
ax.add_artist(legend1)

plt.show()



from tabulate import tabulate

# Generate table for metrics
metrics_table = pd.DataFrame({
    "Model": ["SVM", "Random Forest", "Logistic Regression", "XGBoost"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_xgb),
    ],
    "Precision": [
        precision_score(y_test, y_pred_svm, average="weighted"),
        precision_score(y_test, y_pred_rf, average="weighted"),
        precision_score(y_test, y_pred_lr, average="weighted"),
        precision_score(y_test, y_pred_xgb, average="weighted"),
    ],
    "Recall": [
        recall_score(y_test, y_pred_svm, average="weighted"),
        recall_score(y_test, y_pred_rf, average="weighted"),
        recall_score(y_test, y_pred_lr, average="weighted"),
        recall_score(y_test, y_pred_xgb, average="weighted"),
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_svm, average="weighted"),
        f1_score(y_test, y_pred_rf, average="weighted"),
        f1_score(y_test, y_pred_lr, average="weighted"),
        f1_score(y_test, y_pred_xgb, average="weighted"),
    ]
})

# Convert DataFrame to a formatted table
formatted_table = tabulate(metrics_table, headers="keys", tablefmt="grid")

# Print the formatted table
print(formatted_table)



from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize

# Binarize the labels for multiclass classification
y_test_binarized = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))

# Function to plot ROC curve for a single model
def plot_roc_curve(model, model_name, X_test, y_test_binarized, label_encoder):
    if hasattr(model, "predict_proba"):  # Check if model supports predict_proba
        y_prob = model.predict_proba(X_test)
    else:  # For SVM, use decision_function
        y_prob = model.decision_function(X_test)
        # Ensure y_prob is properly shaped for multiclass
        if len(label_encoder.classes_) > 2:
            y_prob = np.column_stack([1 - y_prob, y_prob]) if y_prob.ndim == 1 else y_prob

    # Calculate ROC for each class and micro-average
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(label_encoder.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
        roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_prob[:, i])

    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_prob.ravel())
    roc_auc["micro"] = roc_auc_score(y_test_binarized, y_prob, average="micro")

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"Micro-Average ROC (AUC = {roc_auc['micro']:.2f})",
        linewidth=2,
    )
    for i in range(len(label_encoder.classes_)):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"Class {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})",
        )

    plt.title(f"ROC Curve for {model_name}", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

# Plot ROC curves for each model
plot_roc_curve(svm_model, "Support Vector Machine (SVM)", X_test_scaled, y_test_binarized, label_encoder)
plot_roc_curve(rf_model, "Random Forest", X_test_scaled, y_test_binarized, label_encoder)
plot_roc_curve(logistic_model, "Logistic Regression", X_test_scaled, y_test_binarized, label_encoder)
plot_roc_curve(xgb_model, "XGBoost", X_test_scaled, y_test_binarized, label_encoder)

