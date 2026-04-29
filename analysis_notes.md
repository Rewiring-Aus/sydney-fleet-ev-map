# Sydney Fleet Age & EV Analysis — Methodology Notes

**Data snapshot:** 31 January 2025  
**Analysis date:** April 2026  
**Author:** Rewiring Australia

---

## Data Sources

### Vehicle registration data
BITRE *Road Vehicles Australia, January 2025*, released via data.gov.au (dataset `f6e0a290-7d47-4b88-ac3b-34824b0ab334`).

Two files used:

| File | Dimensions | Rows |
|------|-----------|------|
| `rva-2025-vehtype-streg-poareg-yom-mk-rpc.csv` | vehicle type × state × postcode × year of manufacture × make | 3.04M |
| `rva-2025-mvs-vehtype-streg-poareg-mtvpwr-rpc.csv` | vehicle type × state × postcode × motive power | 78k |

### Geographic boundaries
ABS Postal Areas (POA), ASGS Edition 3, GDA2020 shapefile, 2021 edition.

---

## Scope

- **Geography:** Greater Sydney postcodes — 2000–2234, 2555–2574, 2745–2786 (standard ABS Greater Sydney range)
- **Vehicle type:** `Passenger vehicles` only (excludes light commercial, trucks, motorcycles, campervans)
- **State:** NSW registrations only (`state_abb == 'NSW'`)
- **YOM range:** 1980–2025 (excludes historic/veteran vehicles and data quality outliers)

After filtering: **261 postcodes**, **~2.79 million** registered passenger vehicles represented.

---

## Map 1 — Average Fleet Age

**Method:** Weighted mean year of manufacture (YOM) across all registered passenger vehicles per postcode, weighted by vehicle count. Fleet age = 2025 − mean YOM.

**Data:** YOM × postcode file.

**Limitation:** BITRE perturbs small cells (<3 vehicles) and suppresses some counts. Low-confidence postcodes (< 200 total vehicles) are flagged in the CSV but included in the map with a note.

---

## Map 2 — ICE Overrepresentation

**Intended metric:** (ICE share of new 2022–2025 registrations) − (ICE share of total fleet), by postcode.

**Limitation:** No single BITRE file crosses postcode × year of manufacture × motive power simultaneously. The YOM file has make but not motive power; the motive power file has no YOM breakdown.

**Approximation used:** Each postcode's deviation from the Sydney-wide average ICE share (using total fleet motive power, not just new registrations). This shows which postcodes are structurally more ICE-heavy than the Sydney mean, rather than measuring ICE over-purchase in recent years specifically.

Formula: `ice_overrep_proxy = (1 − ev_share_postcode) − (1 − ev_share_sydney_mean)`

A positive value means the postcode has a higher ICE share than the Sydney average; negative means more EV-leaning.

**Better approach (future):** If BITRE releases a postcode × YOM × motive power cross-tab, replace this with the direct calculation.

---

## Map 3 — EV Share

**Method:** Battery/fuel-cell electric vehicles as a share of all registered passenger vehicles per postcode.

This is the same metric typically reported in media coverage (e.g. SMH suburb-level EV density maps). Maps 1 and 2 demonstrate that this metric correlates strongly with fleet age and fleet turnover, not with structural leadership in energy transition.

---

## Key Findings

| Metric | Value |
|--------|-------|
| Pearson r(avg_fleet_age, ev_share) | **−0.560** (p < 0.001, n = 257) |
| Pearson r(new_reg_rate, ev_share) | **+0.416** (p < 0.001, n = 257) |
| Pearson r(avg_fleet_age, new_reg_rate) | **−0.855** |
| Sydney mean fleet age | 10.3 years |
| Sydney mean EV share | 2.4% |
| Top-decile EV share (newest fleets) | ~5.5% |
| Bottom-decile EV share (oldest fleets) | ~0.6% |

**Interpretation:** The strong negative correlation between fleet age and EV share (r = −0.56) means that the suburbs with the most EVs are simply the suburbs that buy the most new cars — not suburbs that are choosing EVs over ICE at an unusual rate. Fleet turnover explains ~31% of the variation in EV share (r² ≈ 0.31). This reframes high-EV suburbs as high-income, high-consumption suburbs, not energy transition leaders.

---

## "New registration" proxy

We define "new registrations" as vehicles with year of manufacture 2022–2025 (the three most recent complete years before the January 2025 snapshot). This conflates brand-new cars with used imports and used-car sales but is the best available proxy from registration data alone. Noted in methodology.

---

## Confidentialisation

BITRE perturbs cell counts to prevent re-identification. Cells below 3 vehicles are suppressed; others may be slightly adjusted. This introduces minor noise, particularly in small postcodes. Postcodes with fewer than 200 total registered passenger vehicles are flagged as `low_confidence = True` in the output CSV.
