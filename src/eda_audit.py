"""
EDA and Data Trap Audit for Kestrel Home Warranty Fraud.
Computes aggregate statistics without leaking raw customer data.
"""

import pandas as pd
import numpy as np

def run_audit():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test_unlabelled.csv')
    partners = pd.read_csv('data/partners.csv')
    products = pd.read_csv('data/products.csv')

    print("=" * 60)
    print("1. DATASET OVERVIEW & TIME HORIZONS")
    print("=" * 60)
    train['submitted_at'] = pd.to_datetime(train['submitted_at'])
    test['submitted_at'] = pd.to_datetime(test['submitted_at'])
    partners['onboarded_date'] = pd.to_datetime(partners['onboarded_date'])

    print(f"Train rows: {len(train)}, Unique claim_ids: {train['claim_id'].nunique()}")
    print(f"Train date span: {train['submitted_at'].min()} to {train['submitted_at'].max()}")
    print(f"Test rows: {len(test)}, Unique claim_ids: {test['claim_id'].nunique()}")
    print(f"Test date span: {test['submitted_at'].min()} to {test['submitted_at'].max()}")

    # Month by month claim counts in train
    train['year_month'] = train['submitted_at'].dt.to_period('M')
    test['year_month'] = test['submitted_at'].dt.to_period('M')
    print("\nTrain Claims by Month:")
    monthly = train.groupby('year_month').agg(
        total_claims=('claim_id', 'count'),
        unique_claims=('claim_id', 'nunique'),
        fraud_claims=('is_fraud', lambda x: (x == 1).sum()),
        undecided_claims=('is_fraud', lambda x: x.isna().sum()),
        non_fraud_claims=('is_fraud', lambda x: (x == 0).sum())
    )
    monthly['fraud_rate_%'] = (monthly['fraud_claims'] / (monthly['fraud_claims'] + monthly['non_fraud_claims'])) * 100
    print(monthly.to_string())

    print("\nTest Claims by Month:")
    print(test.groupby('year_month')['claim_id'].count().to_string())

    print("\n" + "=" * 60)
    print("2. DUPLICATE CLAIM_ID AUDIT")
    print("=" * 60)
    dup_counts = train['claim_id'].value_counts()
    dups = dup_counts[dup_counts > 1]
    print(f"Number of claim_ids appearing >1 time: {len(dups)}")
    print(f"Distribution of claim_id frequencies:\n{dup_counts.value_counts()}")

    # Check differences in duplicate rows
    dup_df = train[train['claim_id'].isin(dups.index)].sort_values(['claim_id', 'submitted_at'])
    
    # Are labels different across duplicates?
    label_diff = dup_df.groupby('claim_id')['is_fraud'].nunique(dropna=False)
    print(f"Duplicates with changing labels (including NaN): {(label_diff > 1).sum()} out of {len(dups)}")
    
    # Check if duplicate is a resubmission: time difference
    dup_time_diff = dup_df.groupby('claim_id')['submitted_at'].agg(lambda x: (x.max() - x.min()).total_seconds() / 3600)
    print(f"Time difference between first and last submission of duplicate claim (hours):")
    print(dup_time_diff.describe())

    # Are amounts identical?
    amount_diff = dup_df.groupby('claim_id')['claim_amount_inr'].nunique()
    print(f"Duplicates with changing claim amounts: {(amount_diff > 1).sum()} out of {len(dups)}")

    # What are the labels in duplicates?
    dup_label_summary = dup_df.groupby('claim_id')['is_fraud'].agg(
        has_fraud=lambda x: (x == 1).any(),
        all_zeros=lambda x: (x == 0).all(),
        has_nan=lambda x: x.isna().any()
    )
    print("Duplicate label profile:")
    print(dup_label_summary.value_counts().to_string())

    print("\n" + "=" * 60)
    print("3. LABELS & LEGACY ZOHO AUDIT")
    print("=" * 60)
    print("Train Source Distribution:")
    print(train.groupby(['source', 'is_fraud'], dropna=False).size().unstack(fill_value=0))

    # Check inspector note presence by source and label
    train['has_inspector_note'] = train['inspector_note'].notna() & (train['inspector_note'].str.strip() != '')
    train['partner_inspected_bool'] = train['partner_inspected'] == 'Y'
    
    print("\nInspector Note presence by source and is_fraud:")
    print(train.groupby(['source', 'is_fraud', 'has_inspector_note'], dropna=False).size().unstack(fill_value=0))

    print("\nPartner Inspected by source and is_fraud:")
    print(train.groupby(['source', 'is_fraud', 'partner_inspected_bool'], dropna=False).size().unstack(fill_value=0))

    print("\n" + "=" * 60)
    print("4. SERIAL NUMBER CLEANING & PATTERNS")
    print("=" * 60)
    # Simple cleaning function
    def clean_serial(s):
        if pd.isna(s):
            return ""
        s = str(s).strip().upper()
        s = s.replace("-", "").replace(" ", "").replace("_", "")
        # Common typos: O to 0, I to 1 in serial numeric parts
        return s

    train['clean_serial'] = train['product_serial'].apply(clean_serial)
    test['clean_serial'] = test['product_serial'].apply(clean_serial)

    print(f"Raw train serials nunique: {train['product_serial'].nunique()}, Clean serials nunique: {train['clean_serial'].nunique()}")
    print(f"Raw test serials nunique: {test['product_serial'].nunique()}, Clean serials nunique: {test['clean_serial'].nunique()}")

    # Serial reuse in train
    serial_claim_counts = train.groupby('clean_serial')['claim_id'].nunique()
    print("\nDistribution of claims per clean_serial in train:")
    print(serial_claim_counts.value_counts().head(10).to_string())

    # How does serial frequency correlate with fraud?
    train['serial_claim_count'] = train['clean_serial'].map(serial_claim_counts)
    print("\nFraud rate by serial_claim_count bracket in train:")
    train['serial_count_bin'] = pd.cut(train['serial_claim_count'], bins=[0, 1, 2, 3, 5, 100], right=True)
    print(train[train['is_fraud'].notna()].groupby('serial_count_bin', observed=False).agg(
        claims=('claim_id', 'count'),
        frauds=('is_fraud', 'sum'),
        fraud_rate=('is_fraud', 'mean')
    ).to_string())

    # Clean serial claimed across different partners
    serial_partner_counts = train.groupby('clean_serial')['partner_id'].nunique()
    train['serial_partner_count'] = train['clean_serial'].map(serial_partner_counts)
    print("\nFraud rate by serial_partner_count (same serial across multiple partners):")
    print(train[train['is_fraud'].notna()].groupby('serial_partner_count', observed=False).agg(
        claims=('claim_id', 'count'),
        frauds=('is_fraud', 'sum'),
        fraud_rate=('is_fraud', 'mean')
    ).to_string())

    print("\n" + "=" * 60)
    print("5. POLICY CHANGE (1 MAY 2026) & CLAIM AMOUNT AUDIT")
    print("=" * 60)
    train['post_may'] = train['submitted_at'] >= '2026-05-01'
    print("Claim count before vs after 1 May 2026:")
    print(train.groupby('post_may').agg(
        total_claims=('claim_id', 'count'),
        mean_amount=('claim_amount_inr', 'mean'),
        median_amount=('claim_amount_inr', 'median'),
        claims_under_2000=('claim_amount_inr', lambda x: (x < 2000).mean() * 100),
        fraud_count=('is_fraud', lambda x: (x == 1).sum()),
        decided_count=('is_fraud', lambda x: x.notna().sum())
    ))

    # Check claims near 2000 threshold
    print("\nAmount distribution bins before vs after May 1, 2026:")
    amount_bins = [0, 500, 1000, 1500, 1900, 2000, 2500, 5000, 10000, 50000]
    train['amt_bin'] = pd.cut(train['claim_amount_inr'], bins=amount_bins)
    amt_cross = pd.crosstab(train['amt_bin'], train['post_may'], normalize='columns') * 100
    amt_cross.columns = ['Pre-May %', 'Post-May %']
    print(amt_cross.to_string())

    # partner_inspected before vs after May 1
    print("\nPartner inspected % before vs after May 1, 2026 (by amount < 2000):")
    print(train.groupby(['post_may', train['claim_amount_inr'] < 2000])['partner_inspected_bool'].mean().unstack())

    print("\n" + "=" * 60)
    print("6. PARTNER ONBOARDING & FAIRNESS AUDIT")
    print("=" * 60)
    # Merge partner info
    train_p = train.merge(partners, on='partner_id', how='left')
    train_p['partner_tenure_days_at_claim'] = (train_p['submitted_at'] - train_p['onboarded_date']).dt.days
    train_p['is_new_partner'] = train_p['onboarded_date'] >= '2025-05-01' # past year from May 2026

    print(f"Total partners in partner master: {len(partners)}")
    print(f"Partners onboarded on or after 2025-05-01: {(partners['onboarded_date'] >= '2025-05-01').sum()}")
    print("\nPartner onboarding date distribution quartiles:")
    print(partners['onboarded_date'].describe())

    print("\nClaims & Fraud by partner vintage (Onboarded >= 2025-05-01 vs earlier):")
    print(train_p[train_p['is_fraud'].notna()].groupby('is_new_partner').agg(
        claims=('claim_id', 'count'),
        partners=('partner_id', 'nunique'),
        frauds=('is_fraud', 'sum'),
        fraud_rate=('is_fraud', 'mean')
    ))

    # Top fraud partners
    print("\nTop 10 partners by fraud count:")
    p_fraud = train_p[train_p['is_fraud'].notna()].groupby(['partner_id', 'city', 'partner_type', 'is_new_partner']).agg(
        total_claims=('claim_id', 'count'),
        fraud_claims=('is_fraud', 'sum'),
        fraud_rate=('is_fraud', 'mean'),
        total_fraud_amt=('claim_amount_inr', lambda x: x[train_p.loc[x.index, 'is_fraud'] == 1].sum())
    ).sort_values('fraud_claims', ascending=False).head(10)
    print(p_fraud.to_string())

    print("\n" + "=" * 60)
    print("7. LEAKAGE CHECK: INSPECTOR NOTE & PARTNER INSPECTED")
    print("=" * 60)
    print("Test set inspection note presence:")
    print(f"Test note present count: {(test['inspector_note'].notna() & (test['inspector_note'].str.strip() != '')).sum()} / {len(test)}")
    print(f"Test partner_inspected == 'Y': {(test['partner_inspected'] == 'Y').sum()} / {len(test)}")

    print("\nSample values of inspector_note in Train when is_fraud == 1:")
    print(train[train['is_fraud'] == 1]['inspector_note'].value_counts(dropna=False).head(10).to_string())

if __name__ == '__main__':
    run_audit()
