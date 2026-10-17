"""
Deep dive into Legacy Zoho labels, Warranty Expiry, Text descriptions,
and New vs Old Partner fraud distributions.
"""

import pandas as pd
import numpy as np

def deep_dive():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test_unlabelled.csv')
    partners = pd.read_csv('data/partners.csv')
    products = pd.read_csv('data/products.csv')

    train['submitted_at'] = pd.to_datetime(train['submitted_at'])
    test['submitted_at'] = pd.to_datetime(test['submitted_at'])
    partners['onboarded_date'] = pd.to_datetime(partners['onboarded_date'])

    # Merge product details
    train = train.merge(products, on='sku', how='left')
    test = test.merge(products, on='sku', how='left')
    
    # Merge partner details
    train = train.merge(partners, on='partner_id', how='left')
    test = test.merge(partners, on='partner_id', how='left')

    print("=" * 60)
    print("1. WARRANTY DURATION vs DAYS SINCE PURCHASE")
    print("=" * 60)
    train['warranty_days'] = train['warranty_months'] * 30.4375
    train['days_past_warranty'] = train['days_since_purchase'] - train['warranty_days']
    train['is_out_of_warranty'] = train['days_past_warranty'] > 0

    print("Claims submitted out of warranty in Train:")
    print(train.groupby('is_out_of_warranty').agg(
        claims=('claim_id', 'count'),
        fraud_count=('is_fraud', lambda x: (x == 1).sum()),
        fraud_rate=('is_fraud', lambda x: (x == 1).sum() / x.notna().sum() if x.notna().sum() > 0 else 0)
    ))

    print("\nOut of warranty in Test:")
    test['warranty_days'] = test['warranty_months'] * 30.4375
    test['days_past_warranty'] = test['days_since_purchase'] - test['warranty_days']
    test['is_out_of_warranty'] = test['days_past_warranty'] > 0
    print(test['is_out_of_warranty'].value_counts(normalize=True))

    print("\n" + "=" * 60)
    print("2. CLAIM AMOUNT vs PRODUCT LIST PRICE")
    print("=" * 60)
    train['amt_to_list_price_ratio'] = train['claim_amount_inr'] / train['list_price_inr']
    test['amt_to_list_price_ratio'] = test['claim_amount_inr'] / test['list_price_inr']

    print("Claim amount / List price ratio percentiles for is_fraud == 0 vs 1:")
    print("Non-fraud (0):")
    print(train[train['is_fraud'] == 0]['amt_to_list_price_ratio'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    print("Fraud (1):")
    print(train[train['is_fraud'] == 1]['amt_to_list_price_ratio'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\n" + "=" * 60)
    print("3. PHOTO ATTACHED & PRIOR CLAIMS")
    print("=" * 60)
    print("Photo attached vs fraud:")
    print(train.groupby('photo_attached').agg(
        claims=('claim_id', 'count'),
        fraud_count=('is_fraud', lambda x: (x == 1).sum()),
        fraud_rate=('is_fraud', lambda x: (x == 1).sum() / x.notna().sum())
    ))

    print("\nCustomer prior claims vs fraud:")
    print(train.groupby('customer_prior_claims').agg(
        claims=('claim_id', 'count'),
        fraud_count=('is_fraud', lambda x: (x == 1).sum()),
        fraud_rate=('is_fraud', lambda x: (x == 1).sum() / x.notna().sum())
    ))

    print("\n" + "=" * 60)
    print("4. LEGACY ZOHO vs CRM LABELS DEEP DIVE")
    print("=" * 60)
    # Check if there are legacy_zoho claims that were never resolved/inspected
    # In CRM, undecided cases have is_fraud == NaN.
    # What are the properties of CRM undecided cases?
    crm_nan = train[train['source'] == 'crm']
    print("CRM label status by photo_attached, partner_inspected, and has_note:")
    train['has_note'] = train['inspector_note'].notna() & (train['inspector_note'].str.strip() != '')
    print(train.groupby(['source', 'is_fraud'], dropna=False)[['photo_attached', 'partner_inspected', 'has_note']].value_counts())

    print("\n" + "=" * 60)
    print("5. PARTNER ANALYSIS: NEW PARTNERS DISTRIBUTION")
    print("=" * 60)
    train['is_new_partner'] = train['onboarded_date'] >= '2025-05-01'
    p_summary = train[train['is_fraud'].notna()].groupby(['partner_id', 'is_new_partner']).agg(
        total_claims=('claim_id', 'nunique'),
        fraud_claims=('is_fraud', lambda x: (x == 1).sum()),
        total_amt=('claim_amount_inr', 'sum'),
        fraud_amt=('claim_amount_inr', lambda x: x[train.loc[x.index, 'is_fraud'] == 1].sum())
    ).reset_index()

    new_p = p_summary[p_summary['is_new_partner'] == True]
    old_p = p_summary[p_summary['is_new_partner'] == False]
    print(f"Total active new partners in train: {len(new_p)}")
    print(f"New partners with 0 fraud: {(new_p['fraud_claims'] == 0).sum()} ({(new_p['fraud_claims'] == 0).mean()*100:.1f}%)")
    print(f"New partners with >=1 fraud: {(new_p['fraud_claims'] > 0).sum()} ({(new_p['fraud_claims'] > 0).mean()*100:.1f}%)")
    print(f"Total active old partners in train: {len(old_p)}")
    print(f"Old partners with 0 fraud: {(old_p['fraud_claims'] == 0).sum()} ({(old_p['fraud_claims'] == 0).mean()*100:.1f}%)")
    print(f"Old partners with >=1 fraud: {(old_p['fraud_claims'] > 0).sum()} ({(old_p['fraud_claims'] > 0).mean()*100:.1f}%)")

    # Concentration of fraud: Lorenz / Gini curve among partners
    fraud_by_partner = train[train['is_fraud'] == 1]['partner_id'].value_counts()
    print(f"\nTotal distinct partners with fraud in train: {len(fraud_by_partner)} out of {train['partner_id'].nunique()}")
    print(f"Top 5 fraud partners account for: {fraud_by_partner.head(5).sum()} / {train['is_fraud'].sum()} frauds ({fraud_by_partner.head(5).sum() / train['is_fraud'].sum()*100:.1f}%)")
    print(f"Top 10 fraud partners account for: {fraud_by_partner.head(10).sum()} / {train['is_fraud'].sum()} frauds ({fraud_by_partner.head(10).sum() / train['is_fraud'].sum()*100:.1f}%)")

if __name__ == '__main__':
    deep_dive()
