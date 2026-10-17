# AI Usage Log — Kestrel Home Warranty Fraud

This document transparently tracks all AI tooling, prompts, cost estimates, iterations, and discarded approaches during the project lifecycle.

---

## Log

| Timestamp | Tool / Model | Task / Prompt Intent | Output Summary | Kept or Discarded? | Reason / Learning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-10-16 14:15 IST | Claude Sonnet 5.5 | Domain research on consumer electronics warranty fraud patterns & policy exploits | Synthesized warranty fraud vectors: serial reuse, late-warranty surge, threshold bunching | Kept | Formed the core hypothesis framework for data audit and feature design |
| 2026-10-16 15:30 IST | Gemini 3.1 Pro | Initial repository & operational policy schema inspection | Extracted key constraints: 40 claims/mo desk cap, Rs 380 goodwill, Rs 2,000 auto-approval | Kept | Essential baseline understanding of Ops Policy v4.1 |
| 2026-10-16 16:45 IST | Gemini Flash | Phase 1 EDA script: audit duplicates, messy serials, and unlabelled rows | Python scripts identifying 681 bounced duplicates and 215 unlabelled CRM rows | Kept | Established clean deduplication protocol (DEC-003) and label exclusion (DEC-004) |
| 2026-10-16 18:20 IST | Claude Sonnet 5.5 | Text feature extraction: 200 TF-IDF n-grams from `claim_description` | Generated sparse TF-IDF matrix for free-text description tokens | **Discarded** | Overfitted severely due to sparse fraud labels (3% base rate); reduced to 8 high-frequency binary keywords |
| 2026-10-16 19:40 IST | Gemini 3.1 Pro | Partner fraud target encoding across historical window | Computed out-of-fold historical fraud rate per partner ID | **Discarded** | Failed to capture May 1 policy shift: clean historical partners suddenly began submitting bunched sub-2k claims |
| 2026-10-16 21:15 IST | Gemini Flash | Temporal policy shock analysis (1 May 2026 transition) | Quantified 520% explosion in Rs 1,900–1,999 claims and disproved "new partners" theory | Kept | Saved evidence to `reports/eda_summary_charts.png`; showed 82.6% of new partners are honest |
| 2026-10-17 09:30 IST | Gemini 3.1 Pro | Standard random k-fold cross-validation scheme | Scikit-learn StratifiedKFold implementation across train dataset | **Discarded** | Temporal look-ahead leakage across the May 1 policy shock; replaced with strictly forward June holdout |
| 2026-10-17 11:00 IST | Gemini Flash | Pre-payout feature engineering: serial repeat & cross-partner reuse | Built vectorised cumulative clean serial metrics and 30-day backward rolling partner rates | Kept | Zero-leakage features isolating threshold gaming without penalising new partner tenure |
| 2026-10-17 12:45 IST | Gemini 3.1 Pro | Polynomial & interaction expansion (degree 3) | Generated ~180 higher-order interaction combinations | **Discarded** | Multicollinearity caused unstable coefficients in regularized logistic regression; kept only targeted policy interactions |
| 2026-10-17 14:15 IST | Claude Sonnet 5.5 | Model benchmark evaluation & bootstrap confidence interval simulation | Evaluated Baselines, RF, XGB, and Unweighted LR on June holdout with 1,000 bootstrap CIs | Kept | Selected regularized LR as primary interpretable baseline (Precision@40=32.5%, Net=+Rs 8,781/mo) |
| 2026-10-17 16:00 IST | Gemini 3.1 Pro | Linear amount-weighted loss function ($\text{weight} \propto \text{claim\_amount}$) | Scaled sample weights linearly by claim amount to target high-dollar claims | **Discarded / Revised** | Ablation test showed 60% of net value hung on a single Rs 17,866 claim; revised to softened square-root weights |
| 2026-10-17 17:30 IST | Gemini Flash | Bug fix: Review desk top-40 binary accuracy indexing | Identified pandas RangeIndex alignment issue in top-40 threshold calculation | Kept | Fixed identical 91.94% display artifact; true accuracy correctly scales to 95.00% |
| 2026-10-17 18:45 IST | Gemini 3.1 Pro | Next_Instructions stress-testing: softened weights ($\alpha=0.50$) & time-decay | Evaluated softened weights, 90-day time decay, and Model 7 (Softened LR + XGB blend) | Kept | Successfully eliminated ablation collapse; Model 7 blend achieved 0.919 ROC-AUC and 13/22 caught frauds |
| 2026-10-17 19:30 IST | Claude Sonnet 5.5 | Executive memo drafting & video script creation | Authored 1-page non-technical `memo.md` and 3-minute video presentation script in `RECORDING_NOTES.md` | Kept | Addressed Ritu's 97% accuracy trap and included exact required ablation quote |
| 2026-10-17 20:00 IST | Gemini Flash | FastAPI service, interactive desk dashboard, and pipeline reproducibility | Built `app/main.py`, single-screen UI `index.html`, and end-to-end `run_all.py` | Kept | Verified 100% offline local execution with zero external paid APIs; runs end-to-end in ~14s |

---

## Cost & API Summary
- Models Used: Gemini 3.1 Pro, Gemini Flash, Claude Sonnet 5.5
- Total External Paid API Cost: $0.00 (Run via existing subscription access; all model serving and inference strictly offline)
- Paid External APIs at Inference / Serving: 0 (Strict offline rule enforced)
