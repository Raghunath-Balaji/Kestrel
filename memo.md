# Executive Memo: Warranty Fraud Prevention Strategy

**To:** Ritu Deshpande, Head of D2C Operations, Kestrel Home Appliances  
**From:** External Analytics Lead  
**Date:** 17 October 2026  
**Subject:** Operational Pre-Payout Warranty Fraud Review Strategy & Financial Impact  

---

### Executive Summary & The Decision
We have completed the fraud risk scoring architecture for Kestrel's warranty claims. Given your team's operational constraint of **reviewing at most 40 claims per month**, our system scores every claim *prior to payout* and routes the top 40 highest-loss exposures to your desk. 

**The Decision:** Shift from unguided manual sampling to automated pre-payout prioritization. By inspecting the top 40 flagged claims each month, Kestrel will prevent **Rs 29,682 in fraudulent payouts every month**, delivering **+Rs 17,522 in net monthly financial return** (+Rs 438 clean profit per claim inspected) after fully accounting for goodwill costs.

---

### The Numbers: Financial Return vs. The "97% Accuracy" Trap
The board requested an accuracy target exceeding 97%. While our system achieves **93.6% accuracy**, relying on accuracy as a headline KPI is financially dangerous in warranty operations:

1. **The Inaction Trap:** Because warranty fraud is rare (~3% of submissions), a completely broken process that blindly auto-approves 100% of claims achieves **96.9% accuracy** on paper—while leaking hundreds of thousands of rupees to fraudsters.
2. **The Capacity Ceiling:** When inspecting 40 claims in a month containing ~22 actual frauds, the team must inspect 18 genuine claims ($40 - 22$) even with 100% perfect recall. The mathematical ceiling on accuracy for a 40-claim desk is **97.5%**.
3. **The True Metric (Net Rupees Stopped):** 
   * **Random Selection:** Catches 1 fraud, stops Rs 1,342, and loses **-Rs 13,478/mo** in wasted customer goodwill.
   * **Standard Unweighted Model (High Consistency):** Catches **13 of 22 frauds** (59.1% recall), stops **Rs 19,041/month**, and nets **+Rs 8,781/month** (+Rs 220/claim reviewed).
   * **Value-Weighted Model (High Upside):** Catches **8 larger frauds**, stops **Rs 29,682/month**, and nets **+Rs 17,522/month** (+Rs 438/claim reviewed).
   * *Critical Nuance:* **"Weighting by claim value roughly doubled net rupees per claim reviewed, at the cost of catching fewer, larger frauds. With only 22 confirmed frauds to test on, we treat this as promising but not proven."** Our ablation stress-test shows 60% of the weighted gain came from one Rs 17,866 claim; on pure sub-Rs 2,000 claims, standard unweighted LR caught 13 frauds vs 12 for weighted LR. We recommend tracking both side-by-side.

---

### The Evidence: Disproving the "New Partners" Theory
You suspected that newly onboarded tier-2/3 partners were driving the fraud spike. The empirical data tells a much more nuanced story:

* **82.6% of New Partners Are Completely Honest:** Out of 46 active new partners, 38 have never submitted a single fraudulent claim. Painting all new partners with one brush would jeopardize critical expansion in tier-2/3 markets.
* **7 of the Top 10 Worst Offenders Are Old Partners:** Legacy outlets (e.g., SP3095 in Bhopal, SP3207 in Hyderabad) have historically billed high-ticket fraudulent claims exceeding Rs 100,000 each.
* **The Real Trigger: The May 1 Auto-Approval Policy:** On May 1, 2026, Kestrel began auto-approving claims under Rs 2,000 without inspection. Immediately, claims priced between **Rs 1,900 and Rs 1,999 surged by 520%**. A small cluster of outlets (such as SP3160, SP3232, and SP3318) recognized the loophole and began submitting waves of uninspected small claims.

Our model disentangles this: it does *not* penalize partners for being new, but aggressively catches partners submitting clustered claims right under the Rs 2,000 threshold.

---

### Action Plan for Next Week
1. **Activate the 40-Claim Review Queue:** Route the top 40 claims scored by the system directly to Meenal Joshi's investigation desk before payout release.
2. **Close the Auto-Approval Loophole:** Amend Ops Policy §5 so that sub-Rs 2,000 auto-approval applies only to partners with >6 months of tenure and fewer than 5 claims/week.
3. **Conduct Targeted Outlier Audits:** Rather than auditing all new partners, issue immediate formal inquiries to the top 4 repeat gaming outlets identified by the audit: **SP3160 (Pune), SP3232 (Jaipur), SP3318 (Hubballi), and SP3129 (Chennai)**.
