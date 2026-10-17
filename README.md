# Kestrel Home Warranty Fraud Detection & Review Desk System

A production-grade, pre-payout warranty claim risk scoring service and economic decision engine for **Kestrel Home Appliances (Pune)**.

Designed to operate under the hard constraint of **reviewing at most 40 claims per month**, this system prioritizes claims by financial loss exposure, preventing **Rs 29,682/month in fraudulent payouts** and netting **+Rs 17,522/month (+Rs 438 per claim reviewed)** after factoring in customer goodwill costs.

---

## Quickstart: From Clean Machine to Running Service

Follow these exact commands in PowerShell or bash:

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/your-username/kestrel.git
cd kestrel

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1    # On Windows PowerShell
# source venv/bin/activate     # On Linux / macOS

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Start the Local Service
```bash
# Launch FastAPI server (runs 100% locally with zero paid APIs or internet access)
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

* **Interactive Web Dashboard:** Open `http://127.0.0.1:8000` in any browser.
* **Interactive API Documentation:** Open `http://127.0.0.1:8000/docs` (Swagger UI).

### 3. Re-generate All Models, Reports & Predictions (1-Command Reproducibility)
```bash
python run_all.py
```
This rebuilds all features, trains candidate and production models, regenerates `reports/model_evaluation_charts.png`, and outputs `predictions.csv`.

---

## Repository Structure

```text
kestrel/
├── app/
│   ├── main.py                     # FastAPI backend (POST /score, GET /health)
│   └── static/
│       └── index.html              # Clean single-screen UI for investigation desk
├── artifacts/                      # Saved production models, scalers, and feature columns
├── data/                           # Raw data directory (git-ignored for customer privacy)
│   ├── train.csv
│   ├── test_unlabelled.csv
│   ├── partners.csv
│   ├── products.csv
│   └── sample_submission.csv
├── reports/                        # Evaluation charts and benchmark summaries
│   ├── benchmark_summary.csv
│   ├── eda_summary_charts.png
│   └── model_evaluation_charts.png
├── src/                            # Pipeline source code
│   ├── eda_audit.py                # Data traps audit (duplicates, legacy zeros, serial cleaning)
│   ├── features.py                 # Leakage-free pre-payout feature engineering
│   ├── train_models.py             # Model training & out-of-time benchmarking
│   ├── evaluate_and_report.py      # Economic simulations, bootstrap CIs & chart generator
│   └── predict_test.py             # Test inference strictly validated against sample format
├── .gitignore                      # Privacy protection (excludes raw customer CSVs)
├── AI_USAGE.md                     # Honest disclosure log of AI prompts, models, and costs
├── ASSUMPTIONS.md                  # Running operational, financial, and feature assumptions
├── DECISIONS.md                    # Architectural and modeling decisions (DEC-001 to DEC-012)
├── memo.md                         # 1-page non-technical executive memo for Ritu Deshpande
├── PREDICTION.md                   # Pre-inference expected test performance commitment
├── predictions.csv                 # 2,252 scored test claims matching sample_submission.csv
├── README.md                       # Setup and service documentation
├── RECORDING_NOTES.md              # 3-minute video presentation walkthrough script
├── requirements.txt                # Pinned dependency versions
└── run_all.py                      # Master end-to-end reproducibility script
```

---

## Benchmark & Financial Return (June 2026 Holdout, Top 40 Claims/Month)

| Model Strategy | Caught Fraud (TP@40) | Precision@40 | Total Fraud Stopped | Net Rupee Value (Goodwill Rs 380) | **Net Value Per Claim Checked** | Net Value (Goodwill + Contact Rs 640) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 0: All Non-Fraud** | 0 / 22 | 0.0% | Rs 0 | Rs 0 | Rs 0 / claim | Rs 0 |
| **Baseline 1: Random Review** | 1 / 22 | 2.5% | Rs 1,342 | -Rs 13,478 | -Rs 337 / claim | -Rs 23,618 |
| **Baseline 2: Business Rule** | 4 / 22 | 10.0% | Rs 7,066 | -Rs 6,614 | -Rs 165 / claim | -Rs 15,974 |
| **Model 1: Random Forest** | 7 / 22 | 17.5% | Rs 11,649 | -Rs 891 | -Rs 22 / claim | -Rs 9,471 |
| **Model 2: XGBoost Classifier** | 8 / 22 | 20.0% | Rs 12,867 | +Rs 707 | +Rs 18 / claim | -Rs 7,613 |
| **Model 3: Unweighted Logistic Reg** | 13 / 22 | 32.5% | Rs 19,041 | +Rs 8,781 | +Rs 220 / claim | +Rs 1,761 |
| **Model 4: Cost-Weighted Logistic Reg** | **8 / 22** | **20.0%** | **Rs 29,682** *(+55.9%)* | **+Rs 17,522** *(+99.5%)* | **+Rs 438 / claim** | **+Rs 9,202** *(5.2x higher)* |

---

## Key Business Insights & Empirical Findings

1. **The "97% Accuracy" KPI is a Trap:**
   * In a population with 3% fraud, predicting "Not Fraud" for every claim yields **96.9% accuracy** while saving Rs 0.
   * Under a 40-claims/month review capacity with ~22 actual monthly frauds, at least 18 genuine claims will always be reviewed, establishing a mathematical ceiling on review desk accuracy of **97.5%**.
   * Our Cost-Weighted Model delivers **93.6% accuracy** while stopping **Rs 29,682 in fraudulent payouts every month**.

2. **The "New Partners" Theory Disproven:**
   * **82.6% of new partners have zero fraudulent claims**, matching the 86.6% zero-rate of legacy partners.
   * 7 of the top 10 worst fraud offenders are **old, established partners** billing large-ticket repairs (e.g., SP3095: Rs 103,769; SP3207: Rs 64,119).
   * The surge in small fraud claims was triggered by the **1 May 2026 auto-approval policy change** (claims < Rs 2,000 auto-approved without inspection), causing claims in the Rs 1,900–1,999 bracket to surge by **520%**.
   * The model specifically isolates the small cluster of gaming outlets without penalizing legitimate tier-2/3 expansion partners.

---

## API Documentation: `POST /score`

### Sample Request
```json
POST /score
Content-Type: application/json

{
  "claim_id": "WC711887",
  "submitted_at": "2026-07-15 10:30",
  "partner_id": "SP3160",
  "sku": "KH-AF-03",
  "product_serial": "KH354589154",
  "days_since_purchase": 120,
  "claim_amount_inr": 1985.0,
  "photo_attached": "Y",
  "customer_prior_claims": 1,
  "claim_description": "blade jammed"
}
```

### Sample Response
```json
{
  "claim_id": "WC711887",
  "fraud_score": 0.5901,
  "recommendation": "Standard Verification Queue (Secondary Review)",
  "risk_tier": "MODERATE RISK",
  "reasons": [
    "Policy Exploit Warning: Claim amount of Rs 1,985 is bunched right below the Rs 2,000 auto-approval threshold.",
    "Customer has 1 prior warranty claim on record."
  ],
  "claim_amount_inr": 1985.0,
  "net_exposure_inr": 0.0
}
```
