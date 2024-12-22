import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Load datasets from various sources
print("Loading datasets for analysis...")
customer_data_raw = pd.read_csv('Data/Customer Profiles/customer_data.csv')
account_activity_raw = pd.read_csv('Data/Customer Profiles/account_activity.csv')
suspicious_activity_raw = pd.read_csv('Data/Fraudulent Patterns/suspicious_activity.csv')
transactions_raw = pd.read_csv('Data/Transaction Data/transaction_records.csv')
transaction_metadata_raw = pd.read_csv('Data/Transaction Data/transaction_metadata.csv')
fraud_indicators_raw = pd.read_csv('Data/Fraudulent Patterns/fraud_indicators.csv')
merchant_data_raw = pd.read_csv('Data/Merchant Information/merchant_data.csv')
transaction_categories_raw = pd.read_csv('Data/Merchant Information/transaction_category_labels.csv')
transaction_amounts_raw = pd.read_csv('Data/Transaction Amounts/amount_data.csv')
anomaly_scores_raw = pd.read_csv('Data/Transaction Amounts/anomaly_scores.csv')

# Merge customer profile data
print("Merging customer-related data...")
customer_info = pd.merge(customer_data_raw, account_activity_raw, on='CustomerID', how='left')
customer_info = pd.merge(customer_info, suspicious_activity_raw, on='CustomerID', how='left')

# Merge transaction-related data
print("Merging transaction-related data...")
transaction_info = pd.merge(transactions_raw, transaction_metadata_raw, on='TransactionID', how='left')
transaction_info = pd.merge(transaction_info, transaction_categories_raw, on='TransactionID', how='left')
transaction_info = pd.merge(transaction_info, transaction_amounts_raw, on='TransactionID', how='left')
transaction_info = pd.merge(transaction_info, anomaly_scores_raw, on='TransactionID', how='left')
transaction_info = pd.merge(transaction_info, fraud_indicators_raw, on='TransactionID', how='left')

# Merge everything into one dataset
final_dataset = pd.merge(transaction_info, customer_info, on='CustomerID', how='left')
final_dataset = pd.merge(final_dataset, merchant_data_raw, on='MerchantID', how='left')

print("Data successfully merged!")

# Data Preprocessing
# Handle missing values by filling them with appropriate strategies

# For numeric columns, fill missing values with the mean
numeric_columns = final_dataset.select_dtypes(include=[np.number]).columns
final_dataset[numeric_columns] = final_dataset[numeric_columns].fillna(final_dataset[numeric_columns].mean())

# For categorical columns, fill missing values with 'Unknown'
categorical_columns = final_dataset.select_dtypes(include=['object']).columns
for col in categorical_columns:
    final_dataset[col] = final_dataset[col].fillna('Unknown')

# Conduct normality tests on numeric columns using the Shapiro-Wilk test
print("\nNormality Tests (Shapiro-Wilk) on numeric columns:")
for col in numeric_columns:
    data_sample = final_dataset[col].dropna()
    if len(data_sample) < 5:
        continue
    stat, p_value = shapiro(data_sample.sample(min(5000, len(data_sample))))  # Use a sample for large datasets
    print(f"{col}: p-value = {p_value:.5f} {'(Non-normal)' if p_value < 0.05 else '(Likely normal)'}")

# Encode categorical columns using LabelEncoder
for column in categorical_columns:
    label_encoder = LabelEncoder()
    final_dataset[column] = label_encoder.fit_transform(final_dataset[column])

# Generate and visualize the correlation matrix
correlation_matrix = final_dataset.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Identify and remove columns with high multicollinearity (correlation > 0.90)
threshold = 0.90
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
columns_to_remove = [
    col for col in upper_triangle.columns if any(upper_triangle[col].abs() > threshold) and col != 'FraudIndicator'
]
print(f"Columns to drop due to high collinearity (> {threshold}): {columns_to_remove}")

# Drop the highly correlated columns
cleaned_data = final_dataset.drop(columns=columns_to_remove)

# Ensure the target column ('FraudIndicator') exists
if 'FraudIndicator' not in cleaned_data.columns:
    raise ValueError("Target column 'FraudIndicator' not found. Please check your dataset.")

# Split data into features (X) and target (y)
X = cleaned_data.drop('FraudIndicator', axis=1)
y = cleaned_data['FraudIndicator']

# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the resampled data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

# Initialize different models for comparison
model_dict = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Function to evaluate the models
def evaluate_model_performance(model, X_test, y_test, model_name="Model"):
    """
    Trains and evaluates a given model, printing out relevant metrics.
    Returns a dictionary with the evaluation results.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, digits=4)
    roc_auc = None

    if hasattr(model, "predict_proba"):
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_mat)
    print("Classification Report:")
    print(classification_rep)
    if roc_auc:
        print(f"ROC AUC: {roc_auc:.4f}")

    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_mat,
        'classification_report': classification_rep,
        'roc_auc': roc_auc
    }

# Train and evaluate all models
evaluation_results = {}
print("\nTraining and Evaluating Models...")
for model_name, model in model_dict.items():
    model.fit(X_train, y_train)
    evaluation_results[model_name] = evaluate_model_performance(model, X_test, y_test, model_name=model_name)

# Hyperparameter tuning for Random Forest model using GridSearchCV
print("\nTuning Random Forest model parameters...")
rf_hyperparameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=rf_hyperparameters,
    cv=stratified_k_fold,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
print("Best Random Forest Parameters:", grid_search.best_params_)

# Evaluate the tuned Random Forest model
print("\nEvaluating Tuned Random Forest...")
tuned_rf_performance = evaluate_model_performance(best_rf_model, X_test, y_test, "Optimized Random Forest")

# Cross-validation on the tuned Random Forest model
print("\nCross-validating Tuned Random Forest (5-fold)...")
cv_scores = cross_val_score(best_rf_model, X_resampled, y_resampled, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"Cross-validation ROC AUC scores: {cv_scores}")
print(f"Mean ROC AUC: {cv_scores.mean():.4f} | Std: {cv_scores.std():.4f}")

final_rf_model = best_rf_model

print("Process Completed")
