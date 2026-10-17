"""
Full Model Benchmark, Financial Simulation, Bootstrap CIs, and Evaluation Report Charts.
Validation set: June 2026 holdout (post-May policy regime, 720 claims, 22 frauds).
Incorporates all Next_Instructions recommendations:
- Advanced serial & cross-partner reuse features
- Sub-2k policy bunching & interaction terms
- Log amount & price ratio
- Days to warranty expiry
- Label cleaning & time-decay sample weighting
- Linear vs Softened Cost-Weighting vs Expected Rupee Value ranking
"""

import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve, precision_recall_curve
import joblib

from features import prepare_data

def run_evaluation_pipeline():
    os.makedirs('reports', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)

    train_df, test_df = prepare_data()

    # Train / Validation Split:
    # Train on April 2025 through May 2026 (includes historical baselines and initial May policy shift)
    # Validate on June 2026 (the latest labelled month, directly preceding July 2026 test)
    train_mask = train_df['submitted_at'] < '2026-06-01'
    val_mask = train_df['submitted_at'] >= '2026-06-01'

    df_tr = train_df[train_mask].copy().reset_index(drop=True)
    df_val = train_df[val_mask].copy().reset_index(drop=True)

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

    X_tr = df_tr[feature_cols].fillna(0)
    y_tr = df_tr['is_fraud'].values
    amounts_tr = df_tr['claim_amount_inr'].values

    X_val = df_val[feature_cols].fillna(0)
    y_val = df_val['is_fraud'].values
    val_amts = df_val['claim_amount_inr'].values

    total_val_claims = len(y_val)
    total_val_frauds = int(y_val.sum())

    print(f"Validation Set (June 2026): {total_val_claims} claims, {total_val_frauds} frauds (Base rate = {total_val_frauds/total_val_claims*100:.2f}%)")

    # Fit Scaler
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    # Base sample weights with time decay & label cleaning
    pos_mult = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    time_w = df_tr['time_decay_weight'].values
    label_w = np.where(df_tr['is_suspicious_legacy_zero'] == 1, 0.5, 1.0)
    w_base = np.where(y_tr == 1, pos_mult, 1.0) * time_w * label_w

    # Cost-weightings
    mean_amt_fraud = amounts_tr[y_tr == 1].mean()
    w_linear = np.where(y_tr == 1, pos_mult * (amounts_tr / mean_amt_fraud), 1.0) * time_w * label_w
    w_soft = np.where(y_tr == 1, pos_mult * ((amounts_tr ** 0.5) / (amounts_tr[y_tr == 1] ** 0.5).mean()), 1.0) * time_w * label_w

    # 1. Baseline 0: All Non-Fraud
    acc_all0 = accuracy_score(y_val, np.zeros_like(y_val))

    # 2. Baseline 1: Random Review (40 claims)
    np.random.seed(42)
    rand_scores = np.random.rand(total_val_claims)

    # 3. Baseline 2: Heuristic Rule
    rule_scores = (
        (df_val['is_bunching_1800_2000'] * 3.0) +
        ((df_val['partner_tenure_days'] <= 180).astype(int) * 2.0) +
        (df_val['serial_repeat_flag'] * 2.0) +
        (df_val['customer_high_prior_claims'] * 1.5)
    ).values

    # 4. Model 1: Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=5, class_weight='balanced', random_state=42)
    rf.fit(X_tr, y_tr, sample_weight=w_base)
    rf_scores = rf.predict_proba(X_val)[:, 1]

    # 5. Model 2: XGBoost
    xgb_mod = xgb.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.04, subsample=0.85, 
        colsample_bytree=0.85, scale_pos_weight=5.0, random_state=42
    )
    xgb_mod.fit(X_tr, y_tr)
    xgb_scores = xgb_mod.predict_proba(X_val)[:, 1]

    # 6. Model 3: Unweighted Regularized Logistic Regression (C=0.05 from CV)
    lr_plain = LogisticRegression(max_iter=1000, random_state=42, C=0.05)
    lr_plain.fit(X_tr_s, y_tr, sample_weight=w_base)
    lr_plain_scores = lr_plain.predict_proba(X_val_s)[:, 1]

    # 7. Model 4: Linear Cost-Weighted LR (alpha = 1.0)
    lr_linear = LogisticRegression(max_iter=1000, random_state=42, C=0.05)
    lr_linear.fit(X_tr_s, y_tr, sample_weight=w_linear)
    lr_linear_scores = lr_linear.predict_proba(X_val_s)[:, 1]

    # 8. Model 5: Softened Cost-Weighted LR (alpha = 0.50) [Recommended Model]
    lr_soft = LogisticRegression(max_iter=1000, random_state=42, C=0.05)
    lr_soft.fit(X_tr_s, y_tr, sample_weight=w_soft)
    lr_soft_scores = lr_soft.predict_proba(X_val_s)[:, 1]

    # 9. Model 6: Expected Rupee Value Ranking (EV = p * claim_amount - (1 - p) * 380)
    ev_scores = lr_plain_scores * val_amts - (1.0 - lr_plain_scores) * 380.0

    # 10. Model 7: Ensemble Blend (50% Softened LR + 50% XGB)
    blend_scores = 0.5 * lr_soft_scores + 0.5 * xgb_scores

    # Metrics evaluation helper
    def get_eval(scores, name):
        roc = roc_auc_score(y_val, scores) if len(np.unique(scores)) > 1 else 0.5
        pr = average_precision_score(y_val, scores) if len(np.unique(scores)) > 1 else y_val.mean()
        
        # Rank top 40 claims (1 month review capacity)
        eval_df = pd.DataFrame({'y': y_val, 's': scores, 'amt': val_amts}).sort_values('s', ascending=False).reset_index(drop=True)
        top40 = eval_df.iloc[:40]
        tp = (top40['y'] == 1).sum()
        fp = 40 - tp
        prec = tp / 40.0
        rec = tp / total_val_frauds
        amt_stopped = top40[top40['y'] == 1]['amt'].sum()
        net_380 = amt_stopped - fp * 380.0
        net_640 = amt_stopped - fp * 640.0
        
        # Binary accuracy using top 40 threshold
        top40_order = np.argsort(scores)[::-1][:40]
        pred_bin = np.zeros(total_val_claims)
        pred_bin[top40_order] = 1
        acc = accuracy_score(y_val, pred_bin)

        return {
            'Model': name,
            'Accuracy': acc,
            'ROC_AUC': roc,
            'PR_AUC': pr,
            'TP@40': tp,
            'Precision@40': prec,
            'Recall@40': rec,
            'Fraud_Stopped_INR': amt_stopped,
            'Net_Value_380_INR': net_380,
            'Net_Per_Claim_380': net_380 / 40.0,
            'Net_Value_640_INR': net_640,
            'Net_Per_Claim_640': net_640 / 40.0
        }

    results = [
        {
            'Model': 'Baseline 0: All Non-Fraud',
            'Accuracy': acc_all0,
            'ROC_AUC': 0.5,
            'PR_AUC': total_val_frauds / total_val_claims,
            'TP@40': 0,
            'Precision@40': 0.0,
            'Recall@40': 0.0,
            'Fraud_Stopped_INR': 0.0,
            'Net_Value_380_INR': 0.0,
            'Net_Per_Claim_380': 0.0,
            'Net_Value_640_INR': 0.0,
            'Net_Per_Claim_640': 0.0
        },
        get_eval(rand_scores, 'Baseline 1: Random Review (40)'),
        get_eval(rule_scores, 'Baseline 2: Business Rule Heuristic'),
        get_eval(rf_scores, 'Model 1: Random Forest'),
        get_eval(xgb_scores, 'Model 2: XGBoost Classifier'),
        get_eval(lr_plain_scores, 'Model 3: Unweighted Logistic Regression'),
        get_eval(lr_linear_scores, 'Model 4: Cost-Weighted LR (Linear)'),
        get_eval(lr_soft_scores, 'Model 5: Softened Cost-Weighted LR (alpha=0.5)'),
        get_eval(ev_scores, 'Model 6: Expected Value Ranking (EV)'),
        get_eval(blend_scores, 'Model 7: Ensemble Blend (Softened LR + XGB)')
    ]

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("BENCHMARK COMPARISON ON JUNE 2026 HOLDOUT (40 CLAIMS / MONTH DESK CAPACITY)")
    print("=" * 90)
    print(res_df.to_string(index=False))

    # Bootstrap 95% Confidence Intervals for Softened Cost-Weighted Logistic Regression
    print("\nComputing 1,000 Bootstrap Confidence Intervals for Softened Cost-Weighted LR...")
    np.random.seed(42)
    n_boot = 1000
    boot_prec = []
    boot_net = []
    boot_stopped = []
    boot_roc = []
    boot_pr = []

    for _ in range(n_boot):
        idx = np.random.choice(total_val_claims, size=total_val_claims, replace=True)
        yt = y_val[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        ys = lr_soft_scores[idx]
        ya = val_amts[idx]
        
        boot_roc.append(roc_auc_score(yt, ys))
        boot_pr.append(average_precision_score(yt, ys))
        
        order = np.argsort(ys)[::-1][:40]
        tp_b = (yt[order] == 1).sum()
        fp_b = 40 - tp_b
        stopped_b = ya[order][yt[order] == 1].sum()
        net_b = stopped_b - fp_b * 380.0
        
        boot_prec.append(tp_b / 40.0)
        boot_stopped.append(stopped_b)
        boot_net.append(net_b)

    ci_dict = {
        'ROC_AUC_95%_CI': (np.percentile(boot_roc, 2.5), np.percentile(boot_roc, 97.5)),
        'PR_AUC_95%_CI': (np.percentile(boot_pr, 2.5), np.percentile(boot_pr, 97.5)),
        'Precision@40_95%_CI': (np.percentile(boot_prec, 2.5), np.percentile(boot_prec, 97.5)),
        'Fraud_Stopped_95%_CI': (np.percentile(boot_stopped, 2.5), np.percentile(boot_stopped, 97.5)),
        'Net_Rupees_95%_CI': (np.percentile(boot_net, 2.5), np.percentile(boot_net, 97.5)),
    }
    for k, v in ci_dict.items():
        print(f"  {k}: [{v[0]:,.2f}, {v[1]:,.2f}]")

    # Generate Report Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Chart 1: ROC & PR Curves
    fpr_lr, tpr_lr, _ = roc_curve(y_val, lr_soft_scores)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_val, xgb_scores)
    axes[0].plot(fpr_lr, tpr_lr, color='#1b9e77', linewidth=2.5, label=f'Softened LR (AUC = {roc_auc_score(y_val, lr_soft_scores):.3f})')
    axes[0].plot(fpr_xgb, tpr_xgb, color='#d95f02', linewidth=2, label=f'XGBoost (AUC = {roc_auc_score(y_val, xgb_scores):.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance')
    axes[0].set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate', fontsize=10)
    axes[0].set_ylabel('True Positive Rate', fontsize=10)
    axes[0].legend(loc='lower right')

    # Chart 2: Net Rupee Value per Claim Reviewed (The Finance KPI)
    models = ['Random', 'Rule', 'RF', 'XGB', 'Plain LR', 'Linear LR', 'Softened LR', 'EV Rank', 'Blend']
    model_names_df = [
        'Baseline 1: Random Review (40)', 'Baseline 2: Business Rule Heuristic', 'Model 1: Random Forest',
        'Model 2: XGBoost Classifier', 'Model 3: Unweighted Logistic Regression', 'Model 4: Cost-Weighted LR (Linear)',
        'Model 5: Softened Cost-Weighted LR (alpha=0.5)', 'Model 6: Expected Value Ranking (EV)',
        'Model 7: Ensemble Blend (Softened LR + XGB)'
    ]
    net_per_claim = [res_df.loc[res_df['Model'] == m, 'Net_Per_Claim_380'].values[0] for m in model_names_df]
    colors = ['#d95f02' if v < 0 else '#1b9e77' for v in net_per_claim]
    bars = axes[1].barh(models, net_per_claim, color=colors, height=0.6)
    axes[1].axvline(0, color='black', linewidth=1)
    axes[1].set_title('Net Rupee Value Generated Per Claim Reviewed', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Net INR per Claim Checked (Goodwill Cost = Rs 380)', fontsize=10)
    for bar in bars:
        w = bar.get_width()
        x_pos = w + (10 if w >= 0 else -40)
        axes[1].text(x_pos, bar.get_y() + bar.get_height()/2.0, f"Rs {w:,.0f}", va='center', fontsize=9, fontweight='bold')

    # Chart 3: Fraud Stopped vs Goodwill Cost (Rupee Breakdown in Top 40)
    x = np.arange(len(models))
    stopped = [res_df.loc[res_df['Model'] == m, 'Fraud_Stopped_INR'].values[0] for m in model_names_df]
    goodwill = [(40 - res_df.loc[res_df['Model'] == m, 'TP@40'].values[0]) * 380 for m in model_names_df]

    width = 0.35
    axes[2].bar(x - width/2, stopped, width, label='Fraud Stopped (INR)', color='#2b5c8f')
    axes[2].bar(x + width/2, goodwill, width, label='Goodwill Cost (INR)', color='#e7298a')
    axes[2].set_title('Top 40 Financial Breakdown: Stopped vs Goodwill Loss', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=35, ha='right')
    axes[2].set_ylabel('INR', fontsize=10)
    axes[2].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('reports/model_evaluation_charts.png', dpi=150)
    print("Evaluation charts saved to reports/model_evaluation_charts.png")

    # Retrain Best Models on ALL Labelled Train Data (April 2025 - June 2026) for Test Inference
    print("\n" + "=" * 90)
    print("RETRAINING FINAL MODELS ON ALL 11,146 LABELLED RECORDS")
    print("=" * 90)
    X_full = train_df[feature_cols].fillna(0)
    y_full = train_df['is_fraud'].values
    amounts_full = train_df['claim_amount_inr'].values

    scaler_final = StandardScaler()
    X_full_s = scaler_final.fit_transform(X_full)

    # Full time weights & label weights
    pos_mult_full = (len(y_full) - y_full.sum()) / y_full.sum()
    time_w_full = train_df['time_decay_weight'].values
    label_w_full = np.where(train_df['is_suspicious_legacy_zero'] == 1, 0.5, 1.0)
    w_soft_full = np.where(y_full == 1, pos_mult_full * ((amounts_full ** 0.5) / (amounts_full[y_full == 1] ** 0.5).mean()), 1.0) * time_w_full * label_w_full
    w_base_full = np.where(y_full == 1, pos_mult_full, 1.0) * time_w_full * label_w_full

    lr_soft_final = LogisticRegression(max_iter=1000, random_state=42, C=0.05)
    lr_soft_final.fit(X_full_s, y_full, sample_weight=w_soft_full)

    lr_unweighted_final = LogisticRegression(max_iter=1000, random_state=42, C=0.05)
    lr_unweighted_final.fit(X_full_s, y_full, sample_weight=w_base_full)

    xgb_final = xgb.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.04, subsample=0.85, 
        colsample_bytree=0.85, scale_pos_weight=5.0, random_state=42
    )
    xgb_final.fit(X_full, y_full)

    # Save artifacts for production service and inference
    joblib.dump(lr_soft_final, 'artifacts/final_lr_weighted_model.joblib')
    joblib.dump(lr_unweighted_final, 'artifacts/final_lr_unweighted_model.joblib')
    joblib.dump(xgb_final, 'artifacts/final_xgb_model.joblib')
    joblib.dump(scaler_final, 'artifacts/final_scaler.joblib')
    joblib.dump(feature_cols, 'artifacts/feature_cols.joblib')
    res_df.to_csv('reports/benchmark_summary.csv', index=False)
    print("All production model artifacts and benchmark summary saved in artifacts/ and reports/")

    return res_df, ci_dict

if __name__ == '__main__':
    run_evaluation_pipeline()
