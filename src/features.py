"""
Feature engineering module for Kestrel Home Warranty Fraud.
Strictly pre-payout features with zero future lookahead leakage.
Incorporates:
- Cleaned serial reuse and cross-partner reuse flags
- Product list price ratios & log amounts
- Policy threshold proximity & bunching (< Rs 2,000)
- Days to warranty expiry & near-expiry flags
- Customer prior claims & photo attached flags
- Partner tenure, rolling 30D claim velocity, and under-2k share
- Post-May policy interaction terms
- Label cleaning indicators for legacy Zoho zeros
"""

import pandas as pd
import numpy as np

def clean_serial_str(s):
    if pd.isna(s):
        return ""
    return "".join([c for c in str(s).strip().upper() if c.isalnum()])

def prepare_data(train_path='data/train.csv', 
                 test_path='data/test_unlabelled.csv', 
                 partners_path='data/partners.csv', 
                 products_path='data/products.csv'):
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    partners = pd.read_csv(partners_path)
    products = pd.read_csv(products_path)

    # Convert timestamps
    train['submitted_at'] = pd.to_datetime(train['submitted_at'])
    test['submitted_at'] = pd.to_datetime(test['submitted_at'])
    partners['onboarded_date'] = pd.to_datetime(partners['onboarded_date'])

    # Deduplicate train: keep latest submission per claim_id (DEC-003)
    train_dedup = train.sort_values('submitted_at').groupby('claim_id', as_index=False).last()

    # Drop undecided claims from train (DEC-004)
    train_clean = train_dedup[train_dedup['is_fraud'].notna()].copy()
    train_clean['is_fraud'] = train_clean['is_fraud'].astype(int)

    train_clean['is_train'] = True
    test['is_train'] = False
    test['is_fraud'] = np.nan

    combined = pd.concat([train_clean, test], ignore_index=True)
    combined = combined.sort_values('submitted_at').reset_index(drop=True)

    # 1. Clean serial and cross-partner reuse (DEC-005 & Next_Instructions)
    combined['clean_serial'] = combined['product_serial'].apply(clean_serial_str)
    combined['serial_reuse_count'] = combined.groupby('clean_serial').cumcount()
    combined['serial_repeat_flag'] = (combined['serial_reuse_count'] > 0).astype(int)
    
    # Cumulative distinct partners per serial (vectorized)
    combined['is_new_partner_for_serial'] = (~combined.duplicated(subset=['clean_serial', 'partner_id'])).astype(int)
    combined['serial_distinct_partners'] = combined.groupby('clean_serial')['is_new_partner_for_serial'].cumsum()
    combined['serial_multi_partner_flag'] = (combined['serial_distinct_partners'] > 1).astype(int)

    # 2. Merge Product Master & Financial Ratios
    combined = combined.merge(products, on='sku', how='left')
    combined['list_price_inr'] = combined['list_price_inr'].fillna(combined['list_price_inr'].median())
    combined['warranty_months'] = combined['warranty_months'].fillna(12)
    
    combined['log_claim_amount'] = np.log1p(combined['claim_amount_inr'])
    combined['amt_to_price_ratio'] = combined['claim_amount_inr'] / combined['list_price_inr']

    # 3. Policy Threshold Exploitation (< Rs 2,000 bunching)
    combined['is_under_2000'] = (combined['claim_amount_inr'] < 2000).astype(int)
    combined['is_bunching_1800_2000'] = ((combined['claim_amount_inr'] >= 1800) & (combined['claim_amount_inr'] < 2000)).astype(int)
    combined['is_bunching_1500_2000'] = ((combined['claim_amount_inr'] >= 1500) & (combined['claim_amount_inr'] < 2000)).astype(int)
    combined['diff_from_2000'] = np.where(combined['claim_amount_inr'] < 2000, 2000 - combined['claim_amount_inr'], 0)

    # 4. Days to Warranty Expiry & Near-Expiry Rush
    combined['warranty_days'] = combined['warranty_months'] * 30.4375
    combined['days_to_warranty_expiry'] = combined['warranty_days'] - combined['days_since_purchase']
    combined['near_expiry_flag'] = ((combined['days_to_warranty_expiry'] >= 0) & (combined['days_to_warranty_expiry'] <= 45)).astype(int)
    combined['warranty_consumed_ratio'] = (combined['days_since_purchase'] / combined['warranty_days']).clip(0, 2)

    # 5. Customer Profile
    combined['photo_attached_flag'] = (combined['photo_attached'] == 'Y').astype(int)
    combined['customer_has_prior_claims'] = (combined['customer_prior_claims'] > 0).astype(int)
    combined['customer_high_prior_claims'] = (combined['customer_prior_claims'] >= 2).astype(int)

    # 6. Partner Master & Rolling 30D Velocity (strictly historical)
    combined = combined.merge(partners, on='partner_id', how='left')
    combined['partner_tenure_days'] = (combined['submitted_at'] - combined['onboarded_date']).dt.days.clip(lower=0)
    combined['is_new_partner'] = (combined['onboarded_date'] >= '2025-05-01').astype(int)
    combined['is_brand_new_partner'] = (combined['partner_tenure_days'] <= 45).astype(int)

    combined = combined.set_index('submitted_at').sort_index()
    p_roll_cnt = combined.groupby('partner_id')['claim_id'].rolling('30D', closed='left').count().reset_index()
    p_roll_u2k = combined.groupby('partner_id')['is_under_2000'].rolling('30D', closed='left').sum().reset_index()
    combined = combined.reset_index()

    combined['partner_claims_last_30d'] = p_roll_cnt['claim_id'].fillna(0)
    combined['partner_under_2k_last_30d'] = p_roll_u2k['is_under_2000'].fillna(0)
    combined['partner_under_2k_rate_30d'] = np.where(
        combined['partner_claims_last_30d'] > 0,
        combined['partner_under_2k_last_30d'] / combined['partner_claims_last_30d'],
        0.0
    )

    # 7. Post-May Policy Interactions
    combined['post_may'] = (combined['submitted_at'] >= '2026-05-01').astype(int)
    combined['post_may_x_under_2k'] = combined['post_may'] * combined['is_under_2000']
    combined['post_may_x_bunching_1800_2000'] = combined['post_may'] * combined['is_bunching_1800_2000']
    combined['post_may_x_new_partner'] = combined['post_may'] * combined['is_new_partner']
    combined['new_partner_small_claim'] = combined['is_new_partner'] * combined['is_under_2000']

    # Partner type dummies
    partner_type_dummies = pd.get_dummies(combined['partner_type'], prefix='partner_type', dtype=int)
    
    # Text tokens
    top_descs = ['blade jammed', 'motor not running', 'loud noise while running', 'power button not working',
                 'water leaking', 'remote not working', 'filter indicator stuck', 'burning smell']
    desc_dummies = pd.DataFrame()
    for d in top_descs:
        desc_dummies[f"desc_{d.replace(' ', '_')}"] = (combined['claim_description'] == d).astype(int)

    combined = pd.concat([combined, partner_type_dummies, desc_dummies], axis=1)

    train_df = combined[combined['is_train'] == True].copy().reset_index(drop=True)
    test_df = combined[combined['is_train'] == False].copy().reset_index(drop=True)

    # 8. Label Cleaning Flag for legacy zoho
    train_df['is_suspicious_legacy_zero'] = (
        (train_df['source'] == 'legacy_zoho') & 
        (train_df['is_fraud'] == 0) & 
        (train_df['partner_inspected'] == 'N')
    ).astype(int)

    # 9. Time Decay Weights: half-life = 90 days from May 2026
    ref_date = pd.to_datetime('2026-06-01')
    days_ago = (ref_date - train_df['submitted_at']).dt.days.clip(lower=0)
    train_df['time_decay_weight'] = 0.5 ** (days_ago / 90.0)

    print(f"Features prepared successfully: Train shape = {train_df.shape}, Test shape = {test_df.shape}")
    return train_df, test_df
