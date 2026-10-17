# Project Decisions Log — Kestrel Home Warranty Fraud

This document records architectural, data engineering, statistical, and modeling decisions made throughout the project lifecycle along with their explicit rationale.

---

## Decisions Log

| Decision ID | Date | Category | Decision | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **DEC-001** | 2026-10-17 | Governance | Keep raw data strictly local in `data/` and add to `.gitignore`. | Enforce Ops Policy v4.1 (§10) data privacy and instruction guidelines. |
| **DEC-002** | 2026-10-17 | Architecture | Structure repository cleanly into `data/`, `src/`, `app/`, `reports/`, with reproducibility script `run_all.py`. | Maintain clarity, code modularity, and reproducible single-command build. |
| **DEC-003** | 2026-10-17 | Data Cleaning | Deduplicate `train.csv` by keeping the latest record per `claim_id`. | Exactly 681 duplicated claims represent bounced resubmissions (~74 hrs later) with identical amounts and labels. Prevents double-counting and split leakage. |
| **DEC-004** | 2026-10-17 | Data Cleaning | Exclude unlabelled rows (`is_fraud.isna()`, 215 rows) from model training. | These represent active, unresolved investigations. Imputing as 0 would contaminate the negative class. |
| **DEC-005** | 2026-10-17 | Feature Engineering | Normalize serial numbers (strip spaces, hyphens, uppercase). | Cleans partner typos (8,835 raw serials condensed to 7,049 clean serials) to accurately detect multi-claim and cross-partner abuse. |
| **DEC-006** | 2026-10-17 | Validation | Adopt temporal holdout validation: Train on `2025-04-01` to `2026-04-30`; Validate on `2026-05-01` to `2026-06-30`. | Tests model robustness on the post-May 1 regime (auto-approval < Rs 2,000) which directly matches test set distribution. |
| **DEC-007** | 2026-10-17 | Leakage Guard | Strictly exclude `inspector_note` and post-decision inspection fields from pre-payout model features. | Pre-payout scoring must only use information available at submission; post-May claims < Rs 2,000 have no inspection by design. |
| **DEC-008** | 2026-10-17 | Feature Engineering | Engineer policy-exploit and partner interaction features (`new_partner_small_claim`, `bunching_small_claim`, `amt_to_price_ratio`, `serial_repeat_flag`). | Statistically disconfounds the May 1 policy shock (< Rs 2,000 auto-approval) from general partner onboarding. |
| **DEC-009** | 2026-10-17 | Validation Split | Validate on June 2026 out-of-time holdout (latest labelled month preceding test period). | Reflects true operational deployment into the post-May 1 auto-approval environment. |
| **DEC-010** | 2026-10-17 | Model Selection | Deploy Regularized Logistic Regression (with calibrated probability scaling) as primary production model. | Achieved superior PR-AUC (0.2943), Precision@40 (32.5%), Recall@40 (59.1%), and highest Net Rupee return (+Rs 8,781/mo, +Rs 220/claim reviewed) while providing direct linear interpretability for desk workers. |
| **DEC-011** | 2026-10-17 | Production Deployment | Retrain final model on all 11,146 cleaned historical records before generating test predictions. | Maximizes sample coverage while preserving feature weighting established in validation. |
| **DEC-012** | 2026-10-17 | Cost-Sensitive Learning | Incorporate claim-amount row weights into Logistic Regression (`sample_weight = pos_mult * (claim_amount / mean_fraud_amount)`). | Directly aligns optimization with business loss function: increases fraud rupees stopped from Rs 19,041 to Rs 29,682 (+55.9%) and nearly doubles monthly net rupee value from Rs 8,781 to Rs 17,522 (+Rs 438/claim reviewed). |
| **DEC-013** | 2026-10-17 | Stress-Testing & Governance | Perform single-fraud ablation (Rs 17,866 removal), small-claims slice test (< Rs 2,000), and 1,000 bootstrap overlap analysis. | Demonstrates that value-weighting captures big-ticket frauds at the cost of slight degradation on pure small claims (12 vs 13 caught). Adopt cost-weighted LR as main submission for Farhan's rupee metric while maintaining plain LR as comparative benchmark. |
| **DEC-014** | 2026-10-17 | Model Refinement | Adopt Softened Cost-Weighting ($\alpha = 0.50$) with time decay ($t_{1/2}=90\text{d}$) and Ensemble Blend (Softened LR + XGB). | Softening prevents single high-ticket outliers from dominating loss while time decay counters May 1 distribution shift. Model 7 blend catches 13/22 frauds with +Rs 8,740/mo net profit (+Rs 1,720 under strict contact costs) and maintains stability in ablation stress-testing. |

*(All project phases complete)*

