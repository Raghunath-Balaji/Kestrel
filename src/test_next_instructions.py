"""
Comprehensive implementation of all 7 recommendations in Next_Instructions.txt:
1. Feature enhancements (serial cross-partner, expiry proximity, log amount, interactions)
2. Label cleaning (down-weighting uninspected legacy zoho zeros)
3. Time decay weighting
4. Rolling time-based CV on training data to tune C and penalty
5. Probability calibration (Platt scaling)
6. Expected Rupee Value ranking: EV = p * amount - (1 - p) * 380
7. Stress tests: ablation (drop biggest fraud), small claims slice, and 1,000 bootstrap CIs.
"""

import sys
import os
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def clean_serial_str(s):
    if pd.isna(s):
        return ""
    return "".join([c for c in str(s).strip().upper() if c.isalnum()])

def build_advanced_features():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test_unlabelled.csv')
    partners = pd.read_csv('data/partners.csv')
    products = pd.read_csv('data/products.csv')

    train['submitted_at'] = pd.to_datetime(train['submitted_at'])
    test['submitted_at'] = pd.to_datetime(test['submitted_at'])
    partners['onboarded_date'] = pd.to_datetime(partners['onboarded_date'])

    # Deduplicate train (DEC-003)
    train_dedup = train.sort_values('submitted_at').groupby('claim_id', as_index=False).last()
    train_clean = train_dedup[train_dedup['is_fraud'].notna()].copy()
    train_clean['is_fraud'] = train_clean['is_fraud'].astype(int)

    train_clean['is_train'] = True
    test['is_train'] = False
    test['is_fraud'] = np.nan

    combined = pd.concat([train_clean, test], ignore_index=True)
    combined = combined.sort_values('submitted_at').reset_index(drop=True)

    # 1. Clean serial and cross-partner reuse (Item 3 in Next_Instructions.txt)
    combined['clean_serial'] = combined['product_serial'].apply(clean_serial_str)
    combined['serial_reuse_count'] = combined.groupby('clean_serial').cumcount()
    combined['serial_repeat_flag'] = (combined['serial_reuse_count'] > 0).astype(int)
    
    # Cumulative distinct partners per serial (vectorized)
    combined['is_new_partner_for_serial'] = (~combined.duplicated(subset=['clean_serial', 'partner_id'])).astype(int)
    combined['serial_distinct_partners'] = combined.groupby('clean_serial')['is_new_partner_for_serial'].cumsum()
    combined['serial_multi_partner_flag'] = (combined['serial_distinct_partners'] > 1).astype(int)

    # 2. Merge Product Master & Price Ratios (Item 3 in Next_Instructions.txt)
    combined = combined.merge(products, on='sku', how='left')
    combined['list_price_inr'] = combined['list_price_inr'].fillna(combined['list_price_inr'].median())
    combined['warranty_months'] = combined['warranty_months'].fillna(12)
    
    combined['log_claim_amount'] = np.log1p(combined['claim_amount_inr'])
    combined['amt_to_price_ratio'] = combined['claim_amount_inr'] / combined['list_price_inr']

    # 3. Policy Threshold Exploitation & Proximity (Item 3 in Next_Instructions.txt)
    combined['is_under_2000'] = (combined['claim_amount_inr'] < 2000).astype(int)
    combined['is_bunching_1800_2000'] = ((combined['claim_amount_inr'] >= 1800) & (combined['claim_amount_inr'] < 2000)).astype(int)
    combined['is_bunching_1500_2000'] = ((combined['claim_amount_inr'] >= 1500) & (combined['claim_amount_inr'] < 2000)).astype(int)
    combined['diff_from_2000'] = np.where(combined['claim_amount_inr'] < 2000, 2000 - combined['claim_amount_inr'], 0)

    # 4. Days to Warranty Expiry (Item 3 in Next_Instructions.txt)
    combined['warranty_days'] = combined['warranty_months'] * 30.4375
    combined['days_to_warranty_expiry'] = combined['warranty_days'] - combined['days_since_purchase']
    combined['near_expiry_flag'] = ((combined['days_to_warranty_expiry'] >= 0) & (combined['days_to_warranty_expiry'] <= 45)).astype(int)
    combined['warranty_consumed_ratio'] = (combined['days_since_purchase'] / combined['warranty_days']).clip(0, 2)

    # 5. Customer Profile
    combined['photo_attached_flag'] = (combined['photo_attached'] == 'Y').astype(int)
    combined['customer_has_prior_claims'] = (combined['customer_prior_claims'] > 0).astype(int)
    combined['customer_high_prior_claims'] = (combined['customer_prior_claims'] >= 2).astype(int)

    # 6. Partner Master & Rolling 30D Metrics (Item 3 in Next_Instructions.txt)
    combined = combined.merge(partners, on='partner_id', how='left')
    combined['partner_tenure_days'] = (combined['submitted_at'] - combined['onboarded_date']).dt.days.clip(lower=0)
    combined['is_new_partner'] = (combined['onboarded_date'] >= '2025-05-01').astype(int)
    combined['is_brand_new_partner'] = (combined['partner_tenure_days'] <= 45).astype(int)

    # Rolling 30D volume and under-2k share strictly past
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

    # 7. Post-May Policy Interactions (Item 3 in Next_Instructions.txt)
    combined['post_may'] = (combined['submitted_at'] >= '2026-05-01').astype(int)
    combined['post_may_x_under_2k'] = combined['post_may'] * combined['is_under_2000']
    combined['post_may_x_bunching_1800_2000'] = combined['post_may'] * combined['is_bunching_1800_2000']
    combined['post_may_x_new_partner'] = combined['post_may'] * combined['is_new_partner']
    combined['new_partner_small_claim'] = combined['is_new_partner'] * combined['is_under_2000']

    # Partner type dummies
    partner_type_dummies = pd.get_dummies(combined['partner_type'], prefix='partner_type', dtype=int)
    
    # Text tokens (TF-IDF / Frequency of descriptions)
    top_descs = ['blade jammed', 'motor not running', 'loud noise while running', 'power button not working',
                 'water leaking', 'remote not working', 'filter indicator stuck', 'burning smell']
    desc_dummies = pd.DataFrame()
    for d in top_descs:
        desc_dummies[f"desc_{d.replace(' ', '_')}"] = (combined['claim_description'] == d).astype(int)

    combined = pd.concat([combined, partner_type_dummies, desc_dummies], axis=1)

    train_df = combined[combined['is_train'] == True].copy().reset_index(drop=True)
    test_df = combined[combined['is_train'] == False].copy().reset_index(drop=True)

    # 8. Clean Labels: Identify noisy uninspected legacy Zoho zeros (Item 4 in Next_Instructions.txt)
    train_df['is_suspicious_legacy_zero'] = (
        (train_df['source'] == 'legacy_zoho') & 
        (train_df['is_fraud'] == 0) & 
        (train_df['partner_inspected'] == 'N')
    ).astype(int)

    # 9. Time Decay Weight (Item 5 in Next_Instructions.txt): Half-life = 90 days from May 2026
    ref_date = pd.to_datetime('2026-06-01')
    days_ago = (ref_date - train_df['submitted_at']).dt.days.clip(lower=0)
    # Exponential decay: weight = 0.5 ** (days / 90)
    train_df['time_decay_weight'] = 0.5 ** (days_ago / 90.0)

    print(f"Advanced features built. Train: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df

def test_pipeline():
    train_df, test_df = build_advanced_features()

    feature_cols = [
        'claim_amount_inr', 'log_claim_amount', 'list_price_inr', 'warranty_months', 'days_since_purchase',
        'amt_to_price_ratio', 'is_under_2000', 'is_bunching_1800_2000', 'is_bunching_1500_2000',
        'diff_from_2000', 'days_to_warranty_expiry', 'near_expiry_flag', 'warranty_consumed_ratio',
        'serial_reuse_count', 'serial_repeat_flag', 'serial_distinct_partners', 'serial_multi_partner_flag',
        'customer_prior_claims', 'customer_has_prior_claims', 'customer_high_prior_claims', 'photo_attached_flag',
        'partner_tenure_days', 'is_new_partner', 'is_brand_new_partner', 'new_partner_small_claim',
        'post_may', 'post_may_x_under_2k', 'post_may_x_bunching_1800_2000', 'post_may_x_new_partner',
        'partner_claims_last_30d', 'partner_under_2k_last_30d', 'partner_under_2k_rate_30d',
        'partner_type_authorised_service_centre', 'partner_type_franchise', 'partner_type_freelance_technician',
        'desc_blade_jammed', 'desc_motor_not_running', 'desc_loud_noise_while_running',
        'desc_power_button_not_working', 'desc_water_leaking', 'desc_remote_not_working',
        'desc_filter_indicator_stuck', 'desc_burning_smell'
    ]

    # Time split: Train on < 2026-06-01, Holdout on June 2026
    train_mask = train_df['submitted_at'] < '2026-06-01'
    val_mask = train_df['submitted_at'] >= '2026-06-01'

    df_tr = train_df[train_mask].copy().reset_index(drop=True)
    df_val = train_df[val_mask].copy().reset_index(drop=True)

    X_tr = df_tr[feature_cols].fillna(0)
    y_tr = df_tr['is_fraud'].values
    amts_tr = df_tr['claim_amount_inr'].values

    X_val = df_val[feature_cols].fillna(0)
    y_val = df_val['is_fraud'].values
    val_amts = df_val['claim_amount_inr'].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # -------------------------------------------------------------
    # Step 4 & 5: Label Cleaning + Time Decay Sample Weights
    # -------------------------------------------------------------
    label_weights = np.where(df_tr['is_suspicious_legacy_zero'] == 1, 0.25, 1.0)
    time_weights = df_tr['time_decay_weight'].values
    pos_mult = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    class_weights = np.where(y_tr == 1, pos_mult, 1.0)
    clean_sample_weight = class_weights * label_weights * time_weights

    # -------------------------------------------------------------
    # Step 6: Tune Regularization C via Rolling Time CV on Train
    # (Without looking at June 2026 validation!)
    # -------------------------------------------------------------
    print("\n--- Tuning Regularization C via Rolling Time CV on Train (< June 2026) ---")
    best_c = 0.1
    best_cv_pr = 0.0
    
    splits = [
        ('2025-12-01', '2026-02-01'),
        ('2026-02-01', '2026-04-01'),
        ('2026-04-01', '2026-06-01')
    ]
    for c_cand in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        cv_prs = []
        for v_start, v_end in splits:
            tr_idx = df_tr['submitted_at'] < v_start
            v_idx = (df_tr['submitted_at'] >= v_start) & (df_tr['submitted_at'] < v_end)
            if v_idx.sum() == 0 or df_tr.loc[v_idx, 'is_fraud'].sum() == 0:
                continue
            
            sc = StandardScaler()
            x_tr_cv = sc.fit_transform(X_tr[tr_idx])
            x_v_cv = sc.transform(X_tr[v_idx])
            w_cv = clean_sample_weight[tr_idx]

            m = LogisticRegression(max_iter=1000, random_state=42, C=c_cand)
            m.fit(x_tr_cv, y_tr[tr_idx], sample_weight=w_cv)
            preds = m.predict_proba(x_v_cv)[:, 1]
            cv_prs.append(average_precision_score(y_tr[v_idx], preds))
        mean_pr = np.mean(cv_prs)
        print(f"  C={c_cand:4.2f} -> Mean CV PR-AUC = {mean_pr:.4f}")
        if mean_pr > best_cv_pr:
            best_cv_pr = mean_pr
            best_c = c_cand
    print(f"Optimal C selected from historical time CV: {best_c}")

    # -------------------------------------------------------------
    # Step 7: Train Calibrated Probability Model (Platt Scaling)
    # -------------------------------------------------------------
    base_lr = LogisticRegression(max_iter=1000, random_state=42, C=best_c)
    cal_model = CalibratedClassifierCV(estimator=base_lr, method='sigmoid', cv=3)
    cal_model.fit(X_tr_s, y_tr, sample_weight=clean_sample_weight)
    cal_p = cal_model.predict_proba(X_val_s)[:, 1]

    brier = brier_score_loss(y_val, cal_p)
    print(f"\nCalibrated Model Brier Score on Validation: {brier:.4f}")
    print(f"Validation ROC-AUC = {roc_auc_score(y_val, cal_p):.4f}, PR-AUC = {average_precision_score(y_val, cal_p):.4f}")

    # -------------------------------------------------------------
    # Step 1: Decision Ranking by Expected Rupee Value (EV)
    # EV = p * claim_amount - (1 - p) * 380
    # -------------------------------------------------------------
    ev_scores = cal_p * val_amts - (1.0 - cal_p) * 380.0

    # Evaluate Top 40 using Expected Value Ranking
    ev_df = pd.DataFrame({
        'claim_id': df_val['claim_id'],
        'y': y_val,
        'p': cal_p,
        'ev': ev_scores,
        'amt': val_amts
    }).sort_values('ev', ascending=False).reset_index(drop=True)

    top40_ev = ev_df.iloc[:40]
    tp_ev = (top40_ev['y'] == 1).sum()
    fp_ev = 40 - tp_ev
    stopped_ev = top40_ev[top40_ev['y'] == 1]['amt'].sum()
    net_ev_380 = stopped_ev - fp_ev * 380.0
    net_ev_640 = stopped_ev - fp_ev * 640.0

    print("\n" + "=" * 80)
    print("RESULTS: EXPECTED RUPEE VALUE RANKING (EV = p*Amount - (1-p)*380)")
    print("=" * 80)
    print(f"Top 40 Reviewed: TP = {tp_ev}/22 ({tp_ev/40:.1%})")
    print(f"Fraud Rupees Stopped: Rs {stopped_ev:,.0f}")
    print(f"Net Rupee Value (Goodwill Rs 380): Rs {net_ev_380:,.0f} (+Rs {net_ev_380/40:,.0f}/claim reviewed)")
    print(f"Net Rupee Value (Strict Rs 640):   Rs {net_ev_640:,.0f} (+Rs {net_ev_640/40:,.0f}/claim reviewed)")

    # -------------------------------------------------------------
    # Stress Test 1: Ablation Test (Drop biggest fraud)
    # -------------------------------------------------------------
    print("\n--- Stress Test 1: Ablation Test (Removing Single Biggest Caught Fraud) ---")
    caught = top40_ev[top40_ev['y'] == 1].sort_values('amt', ascending=False)
    biggest_claim = caught.iloc[0]
    print(f"Biggest caught fraud: {biggest_claim['claim_id']} = Rs {biggest_claim['amt']:,.0f}")
    
    ablation_caught = caught.iloc[1:]
    ablation_stopped = ablation_caught['amt'].sum()
    ablation_tp = len(ablation_caught)
    ablation_fp = 40 - ablation_tp
    ablation_net = ablation_stopped - ablation_fp * 380.0
    print(f"Ablation Result: TP={ablation_tp}/40, Stopped=Rs {ablation_stopped:,.0f}, Net(380)=Rs {ablation_net:,.0f} (Rs {ablation_net/40:,.0f}/claim)")

    # -------------------------------------------------------------
    # Stress Test 2: Small Claims Slice (< Rs 2,000 only)
    # -------------------------------------------------------------
    print("\n--- Stress Test 2: Small Claims Slice (< Rs 2,000 only) ---")
    small_df = ev_df[ev_df['amt'] < 2000].sort_values('ev', ascending=False).reset_index(drop=True)
    top40_small = small_df.iloc[:40]
    tp_sm = (top40_small['y'] == 1).sum()
    fp_sm = 40 - tp_sm
    stopped_sm = top40_small[top40_small['y'] == 1]['amt'].sum()
    net_sm = stopped_sm - fp_sm * 380.0
    print(f"Small Claims Top 40: TP={tp_sm}/21 ({tp_sm/40:.1%}), Stopped=Rs {stopped_sm:,.0f}, Net(380)=Rs {net_sm:,.0f} (Rs {net_sm/40:,.0f}/claim)")

    # -------------------------------------------------------------
    # Stress Test 3: 1,000 Bootstrap Iterations
    # -------------------------------------------------------------
    print("\n--- Stress Test 3: 1,000 Bootstrap Confidence Intervals ---")
    np.random.seed(42)
    boot_net = []
    boot_prec = []
    boot_stopped = []

    for _ in range(1000):
        idx = np.random.choice(len(y_val), size=len(y_val), replace=True)
        yt = y_val[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        ya = val_amts[idx]
        ev_b = ev_scores[idx]
        
        ord_b = np.argsort(ev_b)[::-1][:40]
        tp_b = (yt[ord_b] == 1).sum()
        fp_b = 40 - tp_b
        st_b = ya[ord_b][yt[ord_b] == 1].sum()
        net_b = (st_b - fp_b * 380.0) / 40.0
        
        boot_prec.append(tp_b / 40.0)
        boot_stopped.append(st_b)
        boot_net.append(net_b)

    ci_net = (np.percentile(boot_net, 2.5), np.percentile(boot_net, 97.5))
    ci_prec = (np.percentile(boot_prec, 2.5), np.percentile(boot_prec, 97.5))
    ci_st = (np.percentile(boot_stopped, 2.5), np.percentile(boot_stopped, 97.5))

    print(f"Precision@40 95% CI: [{ci_prec[0]:.1%}, {ci_prec[1]:.1%}] (Median: {np.median(boot_prec):.1%})")
    print(f"Fraud Stopped 95% CI: [Rs {ci_st[0]:,.0f}, Rs {ci_st[1]:,.0f}] (Median: Rs {np.median(boot_stopped):,.0f})")
    print(f"Net Rs/Claim 95% CI:  [Rs {ci_net[0]:,.0f}, Rs {ci_net[1]:,.0f}] (Median: Rs {np.median(boot_net):,.0f})")

    return {
        'model': cal_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'best_c': best_c,
        'top40_ev': top40_ev,
        'net_ev_380': net_ev_380,
        'stopped_ev': stopped_ev
    }

if __name__ == '__main__':
    test_pipeline()
