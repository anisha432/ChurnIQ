"""
ChurnIQ - Model Training Script
Run independently: python train_model.py
"""

import os
import sys
from utils import (
    load_telco_data, clean_data, engineer_features,
    preprocess_for_training, train_logistic_regression,
    train_random_forest, evaluate_model, get_feature_importance,
    select_best_model, save_artifacts
)


def main():
    print("=" * 60)
    print("  ChurnIQ — Model Training Pipeline")
    print("=" * 60)

    print("\n[Step 1/7] Loading dataset...")
    df = load_telco_data()

    print("\n[Step 2/7] Cleaning data...")
    df_clean = clean_data(df)

    print("\n[Step 3/7] Engineering features...")
    df_featured = engineer_features(df_clean)

    print("\n[Step 4/7] Preprocessing (encode, scale, split)...")
    X_train, X_test, y_train, y_test, scaler, feature_cols, num_cols = \
        preprocess_for_training(df_featured)

    print("\n[Step 5/7] Training models...")
    print("  Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    print("  Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)

    print("\n[Step 6/7] Evaluating models...")
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    lr_fi = get_feature_importance(lr_model, feature_cols, "Logistic Regression")
    rf_fi = get_feature_importance(rf_model, feature_cols, "Random Forest")

    print("\n[Step 7/7] Selecting best model & saving artifacts...")
    all_results = [lr_results, rf_results]
    best = select_best_model(all_results)
    best_name = best['Model']

    if best_name == "Logistic Regression":
        best_model = lr_model
        best_fi = lr_fi
    else:
        best_model = rf_model
        best_fi = rf_fi

    print(f"\n  Best Model: {best_name}")
    print(f"  Accuracy: {best['Accuracy']:.4f}")
    print(f"  ROC-AUC:  {best['ROC-AUC']:.4f}")
    print(f"  F1 Score: {best['F1 Score']:.4f}")

    save_artifacts(best_model, scaler, feature_cols, num_cols, all_results, best_name)

    print("\n  Top 10 Important Features:")
    for i, row in best_fi.head(10).iterrows():
        print(f"    {i+1:2d}. {row['feature']:<40s} {row['importance']:.4f}")

    print("\n" + "=" * 60)
    print("  Training Complete! Run 'streamlit run app.py' to launch.")
    print("=" * 60)


if __name__ == "__main__":
    main()