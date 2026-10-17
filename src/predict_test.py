"""
Generate predictions.csv for test_unlabelled.csv.
Matches sample_submission.csv shape and column headers exactly.
"""

import sys
import os
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
import joblib

from features import prepare_data

def generate_predictions():
    print("Loading test data and preparing features...")
    train_df, test_df = prepare_data()

    # Enhanced policy interaction features
    for df in [train_df, test_df]:
        df['is_brand_new_partner'] = (df['partner_tenure_days'] <= 45).astype(int)
        df['new_partner_small_claim'] = df['is_new_partner'] * df['is_under_2000']
        df['brand_new_small_claim'] = df['is_brand_new_partner'] * df['is_under_2000']
        df['bunching_small_claim'] = df['is_bunching_1800_2000'] * df['is_new_partner']

    # Load artifacts
    model = joblib.load('artifacts/final_lr_weighted_model.joblib')
    scaler = joblib.load('artifacts/final_scaler.joblib')
    feature_cols = joblib.load('artifacts/feature_cols.joblib')

    # Prepare test matrix
    X_test = test_df[feature_cols].fillna(0)
    X_test_scaled = scaler.transform(X_test)

    # Predict calibrated probability of fraud
    test_scores = model.predict_proba(X_test_scaled)[:, 1]

    # Create raw predictions dataframe
    raw_sub = pd.DataFrame({
        'claim_id': test_df['claim_id'],
        'score': np.round(test_scores, 4)
    })

    # Strictly preserve the exact row order of sample_submission.csv
    sample_sub = pd.read_csv('data/sample_submission.csv')
    sub = sample_sub[['claim_id']].merge(raw_sub, on='claim_id', how='left')
    assert len(sub) == len(sample_sub), f"Shape mismatch: {len(sub)} vs {len(sample_sub)}"
    assert list(sub.columns) == list(sample_sub.columns), f"Columns mismatch: {sub.columns} vs {sample_sub.columns}"
    assert (sub['claim_id'].values == sample_sub['claim_id'].values).all(), "claim_id sequence mismatch"
    assert sub['score'].isna().sum() == 0, "Null scores detected"
    assert (sub['score'] >= 0.0).all() and (sub['score'] <= 1.0).all(), "Scores out of [0, 1] bounds"

    # Save to root as predictions.csv
    out_path = 'predictions.csv'
    sub.to_csv(out_path, index=False)
    print(f"SUCCESS: Saved {len(sub)} predictions to {out_path}")
    print("\nSummary of predicted scores on test set:")
    print(sub['score'].describe())
    print("\nTop 10 highest-scored claims on test set:")
    print(sub.sort_values('score', ascending=False).head(10).to_string(index=False))

    return sub

if __name__ == '__main__':
    generate_predictions()
