# Project Assumptions — Kestrel Home Warranty Fraud

This document records all operational, financial, and modeling assumptions made during the investigation and model development, including the rationale for each.

---

## 1. Business & Financial Assumptions

| Assumption ID | Area | Assumption | Rationale |
| :--- | :--- | :--- | :--- |
| **ASSUMP-001** | Desk Capacity | The investigation team inspects at most 40 claims per calendar month. | Stated as a hard operational constraint in Ops Policy v4.1 (§5) and Farhan Sheikh's directive. |
| **ASSUMP-002** | Cost of False Positive | Holding a genuine claim incurs an average goodwill cost of Rs 380. | Defined in Ops Policy v4.1 (§4). Represents repair delay and customer friction. |
| **ASSUMP-003** | Cost of False Negative | Payout of an undetected fraudulent claim loses 100% of the claimed amount (`claim_amount_inr`). | Defined in Ops Policy v4.1 (§4): "A fraudulent claim paid out costs the full claim amount." |
| **ASSUMP-004** | Service Contact Cost | Blended cost of a service contact is Rs 260. | Defined in Ops Policy v4.1 (§4). Applied to genuine inquiries or contacts if flagged. |
| **ASSUMP-005** | Decision Horizon | Claims must be evaluated strictly *prior to payout*. | Any feature created must only use information available at claim submission or before final payout authorization. Post-investigation data cannot be used. |

---

## 2. Data & Feature Engineering Assumptions

| Assumption ID | Area | Assumption | Rationale |
| :--- | :--- | :--- | :--- |
| **ASSUMP-006** | Label Censoring | Unlabelled claims (`is_fraud.isna()`) represent unadjudicated claims, not legitimate claims. | Treating undecided cases as non-fraud would introduce negative label noise. |
| **ASSUMP-007** | Duplicate Claims | Multiple rows with the same `claim_id` represent successive submission attempts for the same claim event. | Identical amounts and labels across submissions indicate partners re-submitting bounced claims, confirmed by IT Admin Tanmay Kulkarni. |
| **ASSUMP-008** | Serial Formatting | Differences in case, whitespace, and hyphens represent partner typing variance rather than distinct product serials. | Normalizing serials reveals authentic repeat claim attempts on the same hardware unit. |
| **ASSUMP-009** | Policy Distribution Shift | Post-May 1, 2026 data reflects behavioral exploitation of the auto-approval threshold (< Rs 2,000). | Claims in Rs 1,900–2,000 surged 5x (from 1.56% to 8.05%) after May 1, 2026. Models must generalize to this shifted behavior. |

