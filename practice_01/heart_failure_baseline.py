"""
Heart Failure Patient Mortality Prediction
Practice Baseline Classification Pipeline using scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def main():
    # 1. Load Dataset
    data_file = 'heart_failure_clinical_records_ml_task.xlsx'
    print(f"Loading data from {data_file}...")
    df = pd.read_excel(data_file, sheet_name='Patient Clinical Cohort')

    print("\n--- Dataset Summary ---")
    print(df.describe())
    print("\n--- First 5 Rows ---")
    print(df.head())

    # 2. Select Features and Target
    features = ['serum_creatinine', 'serum_sodium', 'creatinine_phosphokinase', 'platelets']
    X = df[features]
    y = df['DEATH_EVENT']

    print(f"\nFeatures selected: {features}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")

    # 3. Stratified Partitioning (80/20 train/test split)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_indx, test_indx in split.split(X, y):
        X_train, X_test = X.iloc[train_indx], X.iloc[test_indx]
        y_train, y_test = y.iloc[train_indx], y.iloc[test_indx]

    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # 4. Feature Scaling (MinMaxScaler)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model Initialization & Training
    reg = LogisticRegression(random_state=42)
    reg.fit(X_train_scaled, y_train)

    # 6. Evaluation on Test Set
    test_score = reg.score(X_test_scaled, y_test)
    y_pred = reg.predict(X_test_scaled)

    # Accuracy calculations
    correct_count = accuracy_score(y_test, y_pred, normalize=False)
    acc_norm = accuracy_score(y_test, y_pred, normalize=True)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1_val = f1_score(y_test, y_pred)

    print("\n--- Model Evaluation (Hold-out Test Set) ---")
    print("Test Accuracy Score (reg.score):", test_score)
    print("Correct Predictions Count:", correct_count, f"/ {len(y_test)}")
    print("Accuracy Score (Normalized):", acc_norm)
    print("Precision Score:", prec)
    print("Recall Score:", rec)
    print("F1 Score:", f1_val)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Survived (0)', 'Deceased (1)']))

    # 7. Stratified 5-Fold Cross-Validation on Training Data
    scores = cross_val_score(reg, X_train_scaled, y_train, cv=5)
    print("\n--- 5-Fold Cross-Validation Scores ---")
    print("Fold Scores:", scores)
    print("Mean CV Score:", scores.mean())

    # 8. Model Coefficients (Risk Drivers)
    coef_summary = pd.DataFrame({
        'Biomarker': features,
        'Coefficient (Log-Odds)': reg.coef_[0],
        'Odds Ratio (e^Beta)': np.exp(reg.coef_[0])
    }).sort_values(by='Coefficient (Log-Odds)', ascending=False)

    print("\n--- Model Learned Coefficients ---")
    print(coef_summary.to_string(index=False))

if __name__ == '__main__':
    main()
