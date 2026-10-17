"""
FastAPI application for Kestrel Home Warranty Fraud Scoring Service.
Runs 100% locally with zero external network dependencies or paid APIs.
Provides POST /score and serves an interactive web dashboard at GET /.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
import os
import re

app = FastAPI(
    title="Kestrel Warranty Fraud Scoring Service",
    description="Pre-payout fraud scoring API and explanation engine for Kestrel Home warranty review desk.",
    version="1.0.0"
)

# Load production artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data")

model = joblib.load(os.path.join(ARTIFACTS_DIR, "final_lr_weighted_model.joblib"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "final_scaler.joblib"))
feature_cols = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_cols.joblib"))

# Load reference tables
products_df = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
products_map = products_df.set_index("sku").to_dict(orient="index")

partners_df = pd.read_csv(os.path.join(DATA_DIR, "partners.csv"))
partners_map = partners_df.set_index("partner_id").to_dict(orient="index")

def clean_serial(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(s).strip().upper())

# Build historical lookups for serial reuse and partner velocity
train_csv = os.path.join(DATA_DIR, "train.csv")
test_csv = os.path.join(DATA_DIR, "test_unlabelled.csv")
serial_history = {}
partner_history = {}

if os.path.exists(train_csv):
    try:
        tr = pd.read_csv(train_csv)
        tr['clean_serial'] = tr['product_serial'].apply(clean_serial)
        for s, grp in tr.groupby('clean_serial'):
            if s:
                serial_history[s] = {
                    'count': len(grp),
                    'partners': grp['partner_id'].nunique()
                }
        tr['submitted_at'] = pd.to_datetime(tr['submitted_at'])
        tr['is_u2k'] = (tr['claim_amount_inr'] < 2000).astype(int)
        for pid, grp in tr.groupby('partner_id'):
            recent = grp[grp['submitted_at'] >= '2026-05-01']
            tot_cnt = len(recent)
            u2k_cnt = recent['is_u2k'].sum() if tot_cnt > 0 else 0
            partner_history[pid] = {
                'recent_cnt': tot_cnt,
                'recent_u2k': u2k_cnt,
                'recent_rate': u2k_cnt / tot_cnt if tot_cnt > 0 else 0.0
            }
    except Exception as e:
        print(f"Warning: Could not build historical cache: {e}")

class ClaimRequest(BaseModel):
    claim_id: str = Field(..., example="WC711887")
    submitted_at: Optional[str] = Field("2026-07-15 10:30", example="2026-07-15 10:30")
    partner_id: str = Field(..., example="SP3160")
    sku: str = Field(..., example="KH-AF-03")
    product_serial: str = Field(..., example="KH354589154")
    days_since_purchase: int = Field(..., example=120)
    claim_amount_inr: float = Field(..., example=1985.0)
    photo_attached: Optional[str] = Field("Y", example="Y")
    customer_prior_claims: Optional[int] = Field(0, example=1)
    claim_description: Optional[str] = Field("blade jammed", example="blade jammed")

class ScoreResponse(BaseModel):
    claim_id: str
    fraud_score: float
    recommendation: str
    risk_tier: str
    reasons: List[str]
    claim_amount_inr: float
    net_exposure_inr: float

def generate_plain_reasons(req: ClaimRequest, features: dict, score: float) -> List[str]:
    reasons = []
    
    # 1. Policy threshold exploitation (< Rs 2,000 bunching)
    amt = req.claim_amount_inr
    if 1800 <= amt < 2000:
        reasons.append(
            f"Policy Exploit Warning: Claim amount of Rs {amt:,.0f} is bunched right below the Rs 2,000 auto-approval threshold."
        )
    elif amt < 2000 and features.get("new_partner_small_claim", 0) == 1:
        reasons.append(
            f"Sub-Rs 2,000 claim submitted by recently onboarded partner {req.partner_id}."
        )

    # 2. Serial Reuse & Cross-Partner Collusion
    if features.get("serial_multi_partner_flag", 0) == 1:
        reasons.append(
            f"Cross-Partner Serial Reuse: Product serial {req.product_serial} has been claimed across {features.get('serial_distinct_partners', 2)} different partner outlets."
        )
    elif features.get("serial_repeat_flag", 0) == 1:
        reasons.append(
            f"Duplicate Serial Claim: Serial {req.product_serial} was previously claimed {features.get('serial_reuse_count', 1)} time(s)."
        )

    # 3. Financial Disproportion
    p_info = products_map.get(req.sku, {})
    list_price = p_info.get("list_price_inr", 6000.0)
    ratio = amt / list_price if list_price > 0 else 0
    if ratio > 0.40:
        reasons.append(
            f"High Price Ratio: Repair claim of Rs {amt:,.0f} represents {ratio*100:.1f}% of appliance list price (Rs {list_price:,.0f})."
        )

    # 4. Near Warranty Expiry Rush
    if features.get("near_expiry_flag", 0) == 1:
        reasons.append(
            f"Near-Expiry Rush: Claim filed within {int(features.get('days_to_warranty_expiry', 15))} days of warranty expiration."
        )

    # 5. Customer Repeat Claims
    prior = req.customer_prior_claims or 0
    if prior >= 2:
        reasons.append(
            f"Elevated Customer Risk: Customer has {prior} previous warranty claims on record."
        )
    elif prior == 1:
        reasons.append("Customer has 1 prior warranty claim on record.")

    # 6. Partner Risk & Vintage
    part_info = partners_map.get(req.partner_id, {})
    partner_type = part_info.get("partner_type", "service_centre").replace("_", " ").title()
    city = part_info.get("city", "Unknown City")
    
    if features.get("is_brand_new_partner", 0) == 1:
        reasons.append(
            f"Partner Tenure: {req.partner_id} ({city}, {partner_type}) was onboarded recently (<45 days active)."
        )

    # Fallback if low risk
    if not reasons:
        reasons.append("Standard claim profile within normal historical operating bounds.")
        reasons.append("No abnormal serial clustering or policy threshold gaming detected.")

    return reasons

@app.post("/score", response_model=ScoreResponse)
def score_claim(claim: ClaimRequest):
    try:
        p_info = products_map.get(claim.sku, {"list_price_inr": 6000.0, "warranty_months": 12})
        list_price = float(p_info.get("list_price_inr", 6000.0))
        warranty_months = float(p_info.get("warranty_months", 12))
        
        part_info = partners_map.get(claim.partner_id, {
            "city": "Unknown", "partner_type": "franchise", "onboarded_date": "2023-01-01"
        })
        
        onboarded_dt = pd.to_datetime(part_info.get("onboarded_date", "2023-01-01"))
        submitted_dt = pd.to_datetime(claim.submitted_at or "2026-07-15")
        partner_tenure = max(0, (submitted_dt - onboarded_dt).days)
        is_new_partner = 1 if onboarded_dt >= pd.to_datetime("2025-05-01") else 0
        is_brand_new = 1 if partner_tenure <= 45 else 0

        amt = float(claim.claim_amount_inr)
        is_under_2k = 1 if amt < 2000.0 else 0
        is_bunching_18_20 = 1 if 1800.0 <= amt < 2000.0 else 0
        is_bunching_15_20 = 1 if 1500.0 <= amt < 2000.0 else 0
        diff_2000 = max(0.0, 2000.0 - amt) if amt < 2000.0 else 0.0

        warranty_days = warranty_months * 30.4375
        days_purchase = float(claim.days_since_purchase)
        consumed_ratio = min(2.0, days_purchase / warranty_days if warranty_days > 0 else 0.5)
        days_to_expiry = warranty_days - days_purchase
        near_expiry_flag = 1 if (0 <= days_to_expiry <= 45) else 0

        c_serial = clean_serial(claim.product_serial)
        s_hist = serial_history.get(c_serial, {'count': 0, 'partners': 1})
        serial_reuse_cnt = s_hist['count']
        serial_distinct_pts = max(1, s_hist['partners'])

        p_hist = partner_history.get(claim.partner_id, {'recent_cnt': 5.0, 'recent_u2k': 2.0, 'recent_rate': 0.4})

        partner_type = part_info.get("partner_type", "franchise")
        desc = str(claim.claim_description or "").strip().lower()
        post_may = 1 if submitted_dt >= pd.to_datetime("2026-05-01") else 0

        # Build feature dictionary matching feature_cols (43 cols) exactly
        feat_dict = {
            "claim_amount_inr": amt,
            "log_claim_amount": np.log1p(amt),
            "list_price_inr": list_price,
            "warranty_months": warranty_months,
            "days_since_purchase": days_purchase,
            "amt_to_price_ratio": amt / list_price if list_price > 0 else 0.3,
            "is_under_2000": is_under_2k,
            "is_bunching_1800_2000": is_bunching_18_20,
            "is_bunching_1500_2000": is_bunching_15_20,
            "diff_from_2000": diff_2000,
            "days_to_warranty_expiry": days_to_expiry,
            "near_expiry_flag": near_expiry_flag,
            "warranty_consumed_ratio": consumed_ratio,
            "serial_reuse_count": serial_reuse_cnt,
            "serial_repeat_flag": 1 if serial_reuse_cnt > 0 else 0,
            "serial_distinct_partners": serial_distinct_pts,
            "serial_multi_partner_flag": 1 if serial_distinct_pts > 1 else 0,
            "customer_prior_claims": claim.customer_prior_claims or 0,
            "customer_has_prior_claims": 1 if (claim.customer_prior_claims or 0) > 0 else 0,
            "customer_high_prior_claims": 1 if (claim.customer_prior_claims or 0) >= 2 else 0,
            "photo_attached_flag": 1 if claim.photo_attached == "Y" else 0,
            "partner_tenure_days": partner_tenure,
            "is_new_partner": is_new_partner,
            "is_brand_new_partner": is_brand_new,
            "new_partner_small_claim": is_new_partner * is_under_2k,
            "post_may": post_may,
            "post_may_x_under_2k": post_may * is_under_2k,
            "post_may_x_bunching_1800_2000": post_may * is_bunching_18_20,
            "post_may_x_new_partner": post_may * is_new_partner,
            "partner_claims_last_30d": p_hist['recent_cnt'],
            "partner_under_2k_last_30d": p_hist['recent_u2k'],
            "partner_under_2k_rate_30d": p_hist['recent_rate'],
            "partner_type_authorised_service_centre": 1 if partner_type == "authorised_service_centre" else 0,
            "partner_type_franchise": 1 if partner_type == "franchise" else 0,
            "partner_type_freelance_technician": 1 if partner_type == "freelance_technician" else 0,
            "desc_blade_jammed": 1 if desc == "blade jammed" else 0,
            "desc_motor_not_running": 1 if desc == "motor not running" else 0,
            "desc_loud_noise_while_running": 1 if desc == "loud noise while running" else 0,
            "desc_power_button_not_working": 1 if desc == "power button not working" else 0,
            "desc_water_leaking": 1 if desc == "water leaking" else 0,
            "desc_remote_not_working": 1 if desc == "remote not working" else 0,
            "desc_filter_indicator_stuck": 1 if desc == "filter indicator stuck" else 0,
            "desc_burning_smell": 1 if desc == "burning smell" else 0,
        }

        # Vectorize with feature names
        x_df = pd.DataFrame([[feat_dict[c] for c in feature_cols]], columns=feature_cols)
        x_scaled = scaler.transform(x_df)

        # Score with cost-weighted model
        score = float(model.predict_proba(x_scaled)[0, 1])
        score_rounded = round(score, 4)

        # Thresholds based on monthly 40-claim capacity
        if score >= 0.70:
            rec = "Hold for Priority Investigation Desk (Top-40 Review)"
            tier = "CRITICAL / HIGH RISK"
        elif score >= 0.35:
            rec = "Standard Verification Queue (Secondary Review)"
            tier = "MODERATE RISK"
        else:
            rec = "Auto-Approve for Payout"
            tier = "LOW RISK"

        reasons = generate_plain_reasons(claim, feat_dict, score)

        return ScoreResponse(
            claim_id=claim.claim_id,
            fraud_score=score_rounded,
            recommendation=rec,
            risk_tier=tier,
            reasons=reasons,
            claim_amount_inr=amt,
            net_exposure_inr=amt if score >= 0.70 else 0.0
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "service": "kestrel-fraud-scoring", "version": "1.0.0"}

# Serve dashboard static assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_home():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Kestrel Warranty Fraud Service running. Visit /docs for Swagger API."}
