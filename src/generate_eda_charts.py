"""
Generate Phase 1 EDA charts for report:
1. Policy shift spike under Rs 2,000
2. Partner fraud concentration
3. Prior claims vs fraud rate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('reports', exist_ok=True)

train = pd.read_csv('data/train.csv')
partners = pd.read_csv('data/partners.csv')
products = pd.read_csv('data/products.csv')

train['submitted_at'] = pd.to_datetime(train['submitted_at'])
train['post_may'] = train['submitted_at'] >= '2026-05-01'

# Deduplicate train: keep latest submission
train_dedup = train.sort_values('submitted_at').groupby('claim_id', as_index=False).last()

# Filter decided claims
decided = train_dedup[train_dedup['is_fraud'].notna()].copy()

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Amount distribution near Rs 2000 before vs after May 1, 2026
pre_may_amts = train_dedup[~train_dedup['post_may']]['claim_amount_inr']
post_may_amts = train_dedup[train_dedup['post_may']]['claim_amount_inr']

bins = np.linspace(0, 4000, 41)
axes[0].hist(pre_may_amts, bins=bins, density=True, alpha=0.5, label='Pre-May 2026', color='#2b5c8f')
axes[0].hist(post_may_amts, bins=bins, density=True, alpha=0.5, label='Post-May 2026 (Auto-Approve < 2k)', color='#d95f02')
axes[0].axvline(2000, color='red', linestyle='--', linewidth=2, label='Rs 2,000 Auto-Approve Threshold')
axes[0].set_title('Claim Amount Distribution Shift (Policy Effect)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Claim Amount (INR)', fontsize=10)
axes[0].set_ylabel('Density', fontsize=10)
axes[0].legend(loc='upper right')
axes[0].set_xlim(0, 4000)

# Plot 2: Partner Fraud Concentration (Pareto chart)
fraud_counts = decided[decided['is_fraud'] == 1].groupby('partner_id').size().sort_values(ascending=False)
cum_fraud = np.cumsum(fraud_counts.values) / fraud_counts.sum() * 100
partner_ranks = np.arange(1, len(fraud_counts) + 1)

axes[1].plot(partner_ranks, cum_fraud, marker='o', color='#7570b3', linewidth=2, markersize=4)
axes[1].axhline(63.4, color='red', linestyle=':', label='Top 10 Partners = 63.4% of Fraud')
axes[1].axvline(10, color='red', linestyle=':')
axes[1].set_title('Cumulative Fraud Concentration by Partner', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Rank of Partner (by Fraud Count)', fontsize=10)
axes[1].set_ylabel('% of All Detected Fraud', fontsize=10)
axes[1].legend(loc='lower right')
axes[1].set_xlim(1, 55)
axes[1].set_ylim(0, 105)

# Plot 3: Prior claims vs Fraud Rate
prior_stats = decided.groupby('customer_prior_claims').agg(
    total=('claim_id', 'count'),
    frauds=('is_fraud', 'sum')
).reset_index()
prior_stats['fraud_rate_%'] = (prior_stats['frauds'] / prior_stats['total']) * 100
prior_stats = prior_stats[prior_stats['customer_prior_claims'] <= 4]

bars = axes[2].bar(prior_stats['customer_prior_claims'], prior_stats['fraud_rate_%'], color='#1b9e77', alpha=0.85, width=0.55)
axes[2].set_title('Fraud Rate by Customer Prior Claims', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Customer Prior Claims', fontsize=10)
axes[2].set_ylabel('Empirical Fraud Rate (%)', fontsize=10)
for bar in bars:
    yval = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/eda_summary_charts.png', dpi=150)
print("Charts saved successfully to reports/eda_summary_charts.png")
