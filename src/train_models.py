"""
Model training, benchmarking, economic evaluation, and bootstrap analysis.
Strictly time-based holdout validation (May-June 2026).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import xgboost as xgb
import joblib
import os

from features import prepare_data

def evaluate_desk_policy(y_true, y_scores, claim_amounts, top_k=80, goodwill_cost=380.0, contact_cost=260.0):
    """
    Simulates desk inspecting top_k claims (e.g. 40/month over 2 months = 80 claims).
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'score': y_scores,
        'amount': claim_amounts
    }).sort_values('score', ascending=False).reset_index(drop=True)

    top_claims = df.iloc[:top_k]
    tp = (top_claims['y_true'] == 1).sum()
    fp = (top_claims['y_true'] == 0).sum()
    precision = tp / top_k
    recall = tp / y_true.sum() if y_true.sum() > 0 else 0.0

    fraud_rupees_stopped = top_claims[top_claims['y_true'] == 1]['amount'].sum()
    goodwill_loss = fp * goodwill_cost
    service_contact_loss = fp * contact_cost # Customer contacting desk for delayed repair
    net_value = fraud_rupees_stopped - goodwill_loss - service_contact_loss
    net_value_per_claim_checked = net_value / top_k

    return {
        'top_k': top_k,
        'true_positives': tp,
        'false_positives': fp,
        'precision_at_k': precision,
        'recall_at_k': recall,
        'fraud_rupees_stopped': fraud_rupees_stopped,
        'goodwill_loss': goodwill_loss,
        'contact_loss': service_contact_loss,
        'net_rupee_value': net_value,
        'net_rupee_per_claim_checked': net_value_per_claim_checked
    }

def bootstrap_confidence_intervals(y_true, y_scores, claim_amounts, top_k=80, n_bootstraps=1000, seed=42):
    np.random.seed(seed)
    n = len(y_true)
    net_values = []
    precisions = []
    pr_aucs = []
    roc_aucs = []

    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        ys = y_scores[idx]
        ca = claim_amounts[idx]

        pr_aucs.append(average_precision_score(yt, ys))
        roc_aucs.append(roc_auc_score(yt, ys))

        res = evaluate_desk_policy(yt, ys, ca, top_k=top_k)
        net_values.append(res['net_rupee_value'])
        precisions.append(res['precision_at_k'])

    return {
        'pr_auc_ci': (np.percentile(pr_aucs, 2.5), np.percentile(pr_aucs, 97.5)),
        'roc_auc_ci': (np.percentile(roc_aucs, 2.5), np.percentile(roc_aucs, 97.5)),
        'precision_ci': (np.percentile(precisions, 2.5), np.percentile(precisions, 97.5)),
        'net_value_ci': (np.percentile(net_values, 2.5), np.percentile(net_values, 97.5)),
    }

def run_benchmarks():
    print("=" * 70)
    print("1. PREPARING FEATURES & TIME-BASED HOLDOUT SPLIT")
    print("=" * 70)
    train_df, test_df = prepare_data()

    # Time split: Train on < 2026-05-01, Validate on >= 2026-05-01
    val_mask = train_df['submitted_at'] >= '2026-05-01'
    df_tr = train_df[~val_mask].copy().reset_index(drop=True)
    df_val = train_df[val_mask].copy().reset_index(drop=True)

    feature_cols = [
        'claim_amount_inr', 'list_price_inr', 'warranty_months', 'days_since_purchase',
        'amt_to_price_ratio', 'is_under_2000', 'is_bunching_1800_2000', 'is_bunching_1500_2000',
        'diff_from_2000', 'warranty_consumed_ratio', 'is_early_claim', 'is_late_claim',
        'serial_prior_claim_count', 'serial_repeat_flag', 'customer_prior_claims',
        'customer_has_prior_claims', 'customer_high_prior_claims', 'photo_attached_flag',
        'partner_tenure_days', 'is_new_partner', 'partner_claims_last_30d',
        'partner_under_2k_last_30d', 'partner_under_2k_rate_30d', 'partner_hist_fraud_score',
        'partner_type_authorised_service_centre', 'partner_type_franchise', 'partner_type_freelance_technician',
        'desc_blade_jammed', 'desc_motor_not_running', 'desc_loud_noise_while_running',
        'desc_power_button_not_working', 'desc_water_leaking', 'desc_remote_not_working',
        'desc_filter_indicator_stuck', 'desc_burning_smell'
    ]

    X_train = df_tr[feature_cols].copy()
    y_train = df_tr['is_fraud'].values
    
    X_val = df_val[feature_cols].copy()
    y_val = df_val['is_fraud'].values
    val_amounts = df_val['claim_amount_inr'].values

    print(f"Historical Train samples: {len(X_train)}, Frauds: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"Validation (May-Jun 2026) samples: {len(X_val)}, Frauds: {y_val.sum()} ({y_val.mean()*100:.2f}%)")
    print(f"Total features: {len(feature_cols)}")

    # 1. Baseline: All Not-Fraud
    acc_all_zero = accuracy_score(y_val, np.zeros_like(y_val))
    print(f"\nBaseline 0: All 'Not Fraud' Model Accuracy: {acc_all_zero*100:.2f}% (Fraud Caught: 0, Net Value: Rs 0)")

    # 2. Baseline: Random Review
    np.random.seed(42)
    random_scores = np.random.rand(len(y_val))
    rand_res = evaluate_desk_policy(y_val, random_scores, val_amounts, top_k=80)
    print(f"Baseline 1: Random Review top 80: TP={rand_res['true_positives']}, Precision={rand_res['precision_at_k']*100:.1f}%, Fraud Stopped=Rs {rand_res['fraud_rupees_stopped']:,.0f}, Net Rupee Value=Rs {rand_res['net_rupee_value']:,.0f}")

    # 3. Baseline: Simple Rule (Claims in 1800-2000 from partners with high under-2k rate or repeat serial)
    rule_score = (
        (df_val['is_bunching_1800_2000'] * 3.0) +
        (df_val['serial_repeat_flag'] * 2.0) +
        (df_val['partner_under_2k_rate_30d'] > 0.8).astype(int) * 2.0 +
        (df_val['customer_high_prior_claims'] * 1.5)
    ).values
    rule_res = evaluate_desk_policy(y_val, rule_score, val_amounts, top_k=80)
    print(f"Baseline 2: Business Rule top 80: TP={rule_res['true_positives']}, Precision={rule_res['precision_at_k']*100:.1f}%, Fraud Stopped=Rs {rule_res['fraud_rupees_stopped']:,.0f}, Net Rupee Value=Rs {rule_res['net_rupee_value']:,.0f}")

    # 4. Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(0))
    X_val_scaled = scaler.transform(X_val.fillna(0))
    
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, C=0.1)
    lr.fit(X_train_scaled, y_train)
    lr_scores = lr.predict_proba(X_val_scaled)[:, 1]
    lr_res = evaluate_desk_policy(y_val, lr_scores, val_amounts, top_k=80)
    print(f"\nModel 1: Logistic Regression: ROC-AUC={roc_auc_score(y_val, lr_scores):.4f}, PR-AUC={average_precision_score(y_val, lr_scores):.4f}")
    print(f"         Desk Top 80: TP={lr_res['true_positives']}/{y_val.sum()}, Prec={lr_res['precision_at_k']*100:.1f}%, Rec={lr_res['recall_at_k']*100:.1f}%, Stopped=Rs {lr_res['fraud_rupees_stopped']:,.0f}, Net=Rs {lr_res['net_rupee_value']:,.0f} (Rs {lr_res['net_rupee_per_claim_checked']:,.0f}/claim)")

    # 5. Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train.fillna(0), y_train)
    rf_scores = rf.predict_proba(X_val.fillna(0))[:, 1]
    rf_res = evaluate_desk_policy(y_val, rf_scores, val_amounts, top_k=80)
    print(f"\nModel 2: Random Forest: ROC-AUC={roc_auc_score(y_val, rf_scores):.4f}, PR-AUC={average_precision_score(y_val, rf_scores):.4f}")
    print(f"         Desk Top 80: TP={rf_res['true_positives']}/{y_val.sum()}, Prec={rf_res['precision_at_k']*100:.1f}%, Rec={rf_res['recall_at_k']*100:.1f}%, Stopped=Rs {rf_res['fraud_rupees_stopped']:,.0f}, Net=Rs {rf_res['net_rupee_value']:,.0f} (Rs {rf_res['net_rupee_per_claim_checked']:,.0f}/claim)")

    # 6. XGBoost / Gradient Boosting
    # Scale positive weight to handle imbalance
    scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()
    model_xgb = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric='logloss'
    )
    model_xgb.fit(X_train, y_train)
    xgb_scores = model_xgb.predict_proba(X_val)[:, 1]
    xgb_res = evaluate_desk_policy(y_val, xgb_scores, val_amounts, top_k=80)
    print(f"\nModel 3: XGBoost: ROC-AUC={roc_auc_score(y_val, xgb_scores):.4f}, PR-AUC={average_precision_score(y_val, xgb_scores):.4f}")
    print(f"         Desk Top 80: TP={xgb_res['true_positives']}/{y_val.sum()}, Prec={xgb_res['precision_at_k']*100:.1f}%, Rec={xgb_res['recall_at_k']*100:.1f}%, Stopped=Rs {xgb_res['fraud_rupees_stopped']:,.0f}, Net=Rs {xgb_res['net_rupee_value']:,.0f} (Rs {xgb_res['net_rupee_per_claim_checked']:,.0f}/claim)")

    # Bootstrap intervals for best model
    print("\nComputing bootstrap intervals (1,000 iterations)...")
    boot = bootstrap_confidence_intervals(y_val, xgb_scores, val_amounts, top_k=80, n_bootstraps=1000)
    print(f"XGBoost PR-AUC 95% CI: [{boot['pr_auc_ci'][0]:.4f}, {boot['pr_auc_ci'][1]:.4f}]")
    print(f"XGBoost ROC-AUC 95% CI: [{boot['roc_auc_ci'][0]:.4f}, {boot['roc_auc_ci'][1]:.4f}]")
    print(f"XGBoost Precision@80 95% CI: [{boot['precision_ci'][0]*100:.1f}%, {boot['precision_ci'][1]*100:.1f}%]")
    print(f"XGBoost Net Value 95% CI: [Rs {boot['net_value_ci'][0]:,.0f}, Rs {boot['net_value_ci'][1]:,.0f}]")

    # Partner fairness check: Are we just flagging all new partners?
    df_val['score'] = xgb_scores
    top80_val = df_val.sort_values('score', ascending=False).head(80)
    print("\n" + "=" * 70)
    print("PARTNER FAIRNESS CHECK ON TOP-80 FLAGGED CLAIMS")
    print("=" * 70)
    print(f"Total claims in Validation: New partners={df_val['is_new_partner'].sum()}, Old partners={(df_val['is_new_partner'] == 0).sum()}")
    print(f"Flagged in Top 80: New partners={top80_val['is_new_partner'].sum()} ({top80_val['is_new_partner'].mean()*100:.1f}%), Old partners={(top80_val['is_new_partner'] == 0).sum()} ({(1-top80_val['is_new_partner'].mean())*100:.1f}%)")
    print("\nDistinct partners flagged in top 80:")
    print(top80_val.groupby(['partner_id', 'is_new_partner']).agg(
        claims_flagged=('claim_id', 'count'),
        actual_frauds=('is_fraud', 'sum')
    ).sort_values('claims_flagged', ascending=False).head(10).to_string())

    # Save model and artifacts
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(model_xgb, 'artifacts/xgb_fraud_model.joblib')
    joblib.dump(feature_cols, 'artifacts/feature_cols.joblib')
    print("\nModel saved to artifacts/xgb_fraud_model.joblib")

    return {
        'val_df': df_val,
        'feature_cols': feature_cols,
        'model': model_xgb,
        'xgb_res': xgb_res,
        'lr_res': lr_res,
        'rule_res': rule_res,
        'rand_res': rand_res,
        'boot': boot
    }

if __name__ == '__main__':
    run_benchmarks()
