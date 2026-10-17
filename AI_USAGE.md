# AI Usage Log — Kestrel Home Warranty Fraud

This document transparently tracks all AI tooling, prompts, cost estimates, iterations, and discarded approaches during the project.

---

## Log

| Timestamp | Tool / Model | Task / Prompt Intent | Output Summary | Kept or Discarded? | Reason / Learning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-10-17 13:00 IST | Gemini 3.8 Flash (High) | Initial repository & data inspection | Extracted key constraints, schema, and baseline counts | Kept | Identified critical constraints (40/mo desk cap, Rs 380 goodwill, Rs 2000 May policy change) |
| 2026-10-17 13:25 IST | Gemini 3.8 Flash (High) | Repository setup & structure initialization | Initialized directories, .gitignore, assumptions, decisions log | Kept | Ensures clean privacy compliance and reproducibility |
| 2026-10-17 13:30 IST | Gemini 3.8 Flash (High) | Phase 1 EDA & Data Integrity Audit | Ran automated audit scripts on duplicates, labels, serials, and policy shift | Kept | Uncovered 5x surge in Rs 1900-2000 claims, partner concentration (10 partners = 63.4% fraud), and validated deduplication rules |
| 2026-10-17 13:35 IST | Gemini 3.8 Flash (High) | Phase 1 Visualization Generation | Created summary visualization in reports/eda_summary_charts.png | Kept | Clear evidence of policy shift and partner concentration for reporting |
| 2026-10-17 14:05 IST | Gemini 3.8 Flash (High) | Phase 2 Feature Engineering | Created policy exploit, partner velocity, and serial reuse features | Kept | Critical to isolating small-claim gaming from partner onboarding |
| 2026-10-17 16:15 IST | Gemini 3.8 Flash (High) | Phase 2 Benchmarking & Bootstrap | Evaluated 6 model candidates on June 2026 holdout with 1,000 bootstrap CIs | Kept | Selected Regularized Logistic Regression: Precision@40=32.5%, Net Value=+Rs 8,781/mo, PR-AUC=0.294 |
| 2026-10-17 18:16 IST | Gemini 3.8 Flash (High) | User Request: Cost-Sensitive Row Weighting | Evaluated sample weights scaled by claim amount on Logistic Regression | Kept | Massive success: Stopped fraud jumped from Rs 19,041 to Rs 29,682 (+55.9%) and Net Rupees nearly doubled to Rs 17,522/mo (+Rs 438/claim) |
| 2026-10-17 18:25 IST | Gemini 3.8 Flash (High) | Bug Fix: Top-40 Binary Accuracy Indexing | Identified RangeIndex slice bug in pred_bin assignment; replaced with top40_order index mapping | Kept | Fixed identical 91.94% artifact; true top-40 accuracy correctly scales from 91.67% to 95.00% across models |
| 2026-10-17 18:35 IST | Gemini 3.8 Flash (High) | Phase 3 Pre-Inference Commitment | Authored PREDICTION.md before running test set inference | Kept | Fully compliant with instructions pre-commitment requirement |
| 2026-10-17 18:38 IST | Gemini 3.8 Flash (High) | Phase 3 Predictions Generation | Scored 2,252 test claims in predictions.csv matching sample_submission.csv | Kept | Strict validation against sample submission shape, headers, and sequence |
| 2026-10-17 18:42 IST | Gemini 3.8 Flash (High) | Phase 3 Service & Web UI Development | Built app/main.py (POST /score) and app/static/index.html | Kept | Zero external network calls; clean single-screen UI for desk agents |
| 2026-10-17 18:46 IST | Gemini 3.8 Flash (High) | Phase 3 Executive Memo & Video Script | Authored 1-page memo.md and RECORDING_NOTES.md (<=3 min script) | Kept | Clean non-technical messaging for Ritu and video script |
| 2026-10-17 18:48 IST | Gemini 3.8 Flash (High) | Phase 3 Reproducibility Pipeline | Pinned requirements.txt, built run_all.py, and authored README.md | Kept | 1-command end-to-end reproducibility verified in 8.5s |
| 2026-10-17 18:52 IST | Gemini 3.8 Flash (High) | Next_Instructions.txt Stress-Tests | Executed single-fraud ablation (Rs 17,866), small-claims slice test, and bootstrap overlap | Kept | Critical finding: value-weighting trades quantity for size; incorporated requested quote into memo.md |

---

## Cost & API Summary
- Models Used: Gemini 3.8 Flash (High)
- Paid External APIs at Inference: 0 (Strict offline rule enforced)
