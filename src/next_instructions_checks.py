"""
Executes the specific validation stress-tests requested in Next_Instructions.txt:
1. Ablation test: Remove single biggest caught fraud (Rs 17,866) and recompute net value.
2. Slice test: Performance on post-May small claims (< Rs 2,000) only.
3. Bootstrap comparison of Net Rupees per claim reviewed for both Unweighted and Cost-Weighted LR.
"""

import sys
import os
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from features import prepare_data

def run_stress_tests():
    train_df, test_df = prepare_data()

    for df in [train_df, test_df]:
        df['is_brand_new_partner'] = (df['partner_tenure_days'] <= 45).astype(int)
        df['new_partner_small_claim'] = df['is_new_partner'] * df['is_under_2000']
        df['brand_new_small_claim'] = df['is_brand_new_partner'] * df['is_under_2000']
        df['bunching_small_claim'] = df['is_bunching_1800_2000'] * df['is_new_partner']

    train_mask = train_df['submitted_at'] < '2026-06-01'
    val_mask = train_df['submitted_at'] >= '2026-06-01'

    df_tr = train_df[train_mask].copy().reset_index(drop=True)
    df_val = train_df[val_mask].copy().reset_index(drop=True)

    feature_cols = [
        'claim_amount_inr', 'list_price_inr', 'warranty_months', 'days_since_purchase',
        'amt_to_price_ratio', 'is_under_2000', 'is_bunching_1800_2000', 'is_bunching_1500_2000',
        'diff_from_2000', 'warranty_consumed_ratio', 'is_early_claim', 'is_late_claim',
        'serial_prior_claim_count', 'serial_repeat_flag', 'customer_prior_claims',
        'customer_has_prior_claims', 'customer_high_prior_claims', 'photo_attached_flag',
        'partner_tenure_days', 'is_new_partner', 'is_brand_new_partner', 'new_partner_small_claim',
        'brand_new_small_claim', 'bunching_small_claim',
        'partner_claims_last_30d', 'partner_under_2k_last_30d', 'partner_under_2k_rate_30d',
        'partner_type_authorised_service_centre', 'partner_type_franchise', 'partner_type_freelance_technician'
    ]

    X_tr = df_tr[feature_cols].fillna(0)
    y_tr = df_tr['is_fraud'].values
    amounts_tr = df_tr['claim_amount_inr'].values

    X_val = df_val[feature_cols].fillna(0)
    y_val = df_val['is_fraud'].values
    val_amts = df_val['claim_amount_inr'].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # Model 1: Unweighted LR
    lr_plain = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, C=0.1)
    lr_plain.fit(X_tr_s, y_tr)
    scores_plain = lr_plain.predict_proba(X_val_s)[:, 1]

    # Model 2: Cost-Weighted LR
    pos_mult = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    mean_amt_fraud = amounts_tr[y_tr == 1].mean()
    w_amount = np.where(y_tr == 1, pos_mult * (amounts_tr / mean_amt_fraud), 1.0)

    lr_weighted = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    lr_weighted.fit(X_tr_s, y_tr, sample_weight=w_amount)
    scores_weighted = lr_weighted.predict_proba(X_val_s)[:, 1]

    print("=" * 80)
    print("CHECK 1: SENSITIVITY TO SINGLE BIGGEST CAUGHT FRAUD (ABLATION TEST)")
    print("=" * 80)
    eval_w = pd.DataFrame({
        'claim_id': df_val['claim_id'],
        'y': y_val,
        'score': scores_weighted,
        'amt': val_amts
    }).sort_values('score', ascending=False).reset_index(drop=True)

    top40_w = eval_w.iloc[:40].copy()
    caught_frauds = top40_w[top40_w['y'] == 1]
    biggest_fraud = caught_frauds.sort_values('amt', ascending=False).iloc[0]
    print(f"Top 40 caught {len(caught_frauds)} frauds totalling Rs {caught_frauds['amt'].sum():,.0f}.")
    print(f"Single biggest caught fraud: Claim {biggest_fraud['claim_id']} = Rs {biggest_fraud['amt']:,.0f}")

    # Recompute without the biggest fraud
    ablation_frauds = caught_frauds[caught_frauds['claim_id'] != biggest_fraud['claim_id']]
    ablation_tp = len(ablation_frauds)
    ablation_fp = 40 - ablation_tp # assuming 1 spot replaced by a non-fraud
    ablation_stopped = ablation_frauds['amt'].sum()
    ablation_net_380 = ablation_stopped - ablation_fp * 380.0
    ablation_net_per_claim = ablation_net_380 / 40.0

    print(f"\nAblation Result (Removing single Rs {biggest_fraud['amt']:,.0f} claim):")
    print(f"  TP: {ablation_tp}/40")
    print(f"  Fraud Stopped: Rs {ablation_stopped:,.0f} (vs Rs {caught_frauds['amt'].sum():,.0f} full)")
    print(f"  Net Value (Rs 380): Rs {ablation_net_380:,.0f} (Net/claim: Rs {ablation_net_per_claim:,.0f})")
    print(f"  Comparison with Plain LR: Plain LR Net = Rs 8,781 (Rs 220/claim)")

    print("\n" + "=" * 80)
    print("CHECK 2: PERFORMANCE ON SMALL-CLAIMS SLICE (< Rs 2,000 ONLY)")
    print("=" * 80)
    small_mask = df_val['claim_amount_inr'] < 2000
    df_val_small = df_val[small_mask].copy().reset_index(drop=True)
    y_small = df_val_small['is_fraud'].values
    amts_small = df_val_small['claim_amount_inr'].values
    scores_plain_small = scores_plain[small_mask]
    scores_weighted_small = scores_weighted[small_mask]

    print(f"Total small claims in June 2026: {len(df_val_small)} (Frauds = {y_small.sum()})")

    # Evaluate top 40 small claims for both models
    def eval_small(scores, name):
        ev = pd.DataFrame({'y': y_small, 's': scores, 'amt': amts_small}).sort_values('s', ascending=False).reset_index(drop=True)
        top_k = min(40, len(ev))
        t40 = ev.iloc[:top_k]
        tp = (t40['y'] == 1).sum()
        fp = top_k - tp
        stopped = t40[t40['y'] == 1]['amt'].sum()
        net = stopped - fp * 380.0
        print(f"  {name:30s} | TP: {tp:2d}/{int(y_small.sum())} ({tp/top_k:.1%}) | Stopped: Rs {stopped:6,.0f} | Net(380): Rs {net:6,.0f} (Rs {net/top_k:,.0f}/claim)")

    eval_small(scores_plain_small, "Plain LR (Small Claims)")
    eval_small(scores_weighted_small, "Cost-Weighted LR (Small Claims)")

    print("\n" + "=" * 80)
    print("CHECK 3: 1,000 BOOTSTRAP OVERLAP TEST (NET RUPEES PER CLAIM REVIEWED)")
    print("=" * 80)
    np.random.seed(42)
    n_boot = 1000
    boot_net_plain = []
    boot_net_weighted = []

    for _ in range(n_boot):
        idx = np.random.choice(len(y_val), size=len(y_val), replace=True)
        yt = y_val[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        ya = val_amts[idx]
        
        # Plain LR
        ys_p = scores_plain[idx]
        ord_p = np.argsort(ys_p)[::-1][:40]
        tp_p = (yt[ord_p] == 1).sum()
        fp_p = 40 - tp_p
        net_p = (ya[ord_p][yt[ord_p] == 1].sum() - fp_p * 380.0) / 40.0
        boot_net_plain.append(net_p)

        # Cost-Weighted LR
        ys_w = scores_weighted[idx]
        ord_w = np.argsort(ys_w)[::-1][:40]
        tp_w = (yt[ord_w] == 1).sum()
        fp_w = 40 - tp_w
        net_w = (ya[ord_w][yt[ord_w] == 1].sum() - fp_w * 380.0) / 40.0
        boot_net_weighted.append(net_w)

    ci_plain = (np.percentile(boot_net_plain, 2.5), np.percentile(boot_net_plain, 97.5))
    ci_weighted = (np.percentile(boot_net_weighted, 2.5), np.percentile(boot_net_weighted, 97.5))

    print(f"Plain LR Net Rs/Claim 95% CI:         [Rs {ci_plain[0]:,.0f}, Rs {ci_plain[1]:,.0f}] (Median: Rs {np.median(boot_net_plain):,.0f})")
    print(f"Cost-Weighted LR Net Rs/Claim 95% CI: [Rs {ci_weighted[0]:,.0f}, Rs {ci_weighted[1]:,.0f}] (Median: Rs {np.median(boot_net_weighted):,.0f})")

    # Overlap quantification
    overlap_low = max(ci_plain[0], ci_weighted[0])
    overlap_high = min(ci_plain[1], ci_weighted[1])
    overlap_range = max(0, overlap_high - overlap_low)
    print(f"\nOverlap range: [Rs {overlap_low:,.0f}, Rs {overlap_high:,.0f}] ({overlap_range:,.0f} Rs width)")
    pct_weighted_beats_plain = (np.array(boot_net_weighted) > np.array(boot_net_plain)).mean() * 100
    print(f"Percentage of bootstrap samples where Cost-Weighted beats Plain LR: {pct_weighted_beats_plain:.1f}%")

if __name__ == '__main__':
    run_stress_tests()
