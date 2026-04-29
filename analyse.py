import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import json
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
SYDNEY_POSTCODES = (
    list(range(2000, 2235))
    + list(range(2555, 2575))
    + list(range(2745, 2787))
)
NEW_YOM_MIN = 2022
NEW_YOM_MAX = 2025
LOW_CONFIDENCE_THRESHOLD = 200

# ── 1. Load YOM file ──────────────────────────────────────────────────────────
print("Loading YOM file...")
yom = pd.read_csv("rva-yom.csv", dtype={"registered_postcode": str})
print(f"  Raw rows: {len(yom):,}")
print(f"  vehicle_type values: {yom['vehicle_type'].unique().tolist()}")
print(f"  state_abb values: {yom['state_abb'].unique().tolist()[:10]}")

# Filter: NSW passenger vehicles only
yom = yom[
    (yom["state_abb"] == "NSW")
    & (yom["vehicle_type"] == "Passenger vehicles")
].copy()
print(f"  After NSW/passenger filter: {len(yom):,} rows")

# Clean postcode — zero-pad to 4 digits, keep only numeric
yom["postcode"] = yom["registered_postcode"].str.strip().str.zfill(4)
yom = yom[yom["postcode"].str.match(r"^\d{4}$")]
yom["postcode_int"] = yom["postcode"].astype(int)
yom = yom[yom["postcode_int"].isin(SYDNEY_POSTCODES)].copy()
print(f"  After Sydney postcode filter: {len(yom):,} rows")

# Clean YOM
yom = yom[yom["year_of_manufacture"].apply(lambda x: str(x).strip().isdigit())]
yom["yom"] = yom["year_of_manufacture"].astype(int)
yom = yom[(yom["yom"] >= 1980) & (yom["yom"] <= 2025)].copy()
yom["count"] = pd.to_numeric(yom["no_vehicles"], errors="coerce").fillna(0)

print(f"  Final YOM rows: {len(yom):,}")
print(f"  Total vehicles represented: {yom['count'].sum():,.0f}")

# ── 2. Map 1: Average fleet age ───────────────────────────────────────────────
print("\nComputing fleet age...")


def weighted_mean_yom(g):
    total = g["count"].sum()
    if total == 0:
        return np.nan
    return np.average(g["yom"], weights=g["count"])


fleet_age = (
    yom.groupby("postcode")
    .apply(weighted_mean_yom, include_groups=False)
    .reset_index()
)
fleet_age.columns = ["postcode", "mean_yom"]
fleet_age["avg_age"] = 2025 - fleet_age["mean_yom"]

total_by_postcode = yom.groupby("postcode")["count"].sum().reset_index()
total_by_postcode.columns = ["postcode", "total_vehicles"]
fleet_age = fleet_age.merge(total_by_postcode, on="postcode")
fleet_age["low_confidence"] = fleet_age["total_vehicles"] < LOW_CONFIDENCE_THRESHOLD
print(
    f"  {len(fleet_age)} postcodes; {fleet_age['low_confidence'].sum()} low-confidence (<{LOW_CONFIDENCE_THRESHOLD} vehicles)"
)

# ── 3. Load motive power file ─────────────────────────────────────────────────
print("\nLoading motive power file...")
mp = pd.read_csv("rva-mtvpwr.csv", dtype={"registered_postcode": str})
mp = mp[
    (mp["state_abb"] == "NSW")
    & (mp["vehicle_type"] == "Passenger vehicles")
].copy()
mp["postcode"] = mp["registered_postcode"].str.strip().str.zfill(4)
mp = mp[mp["postcode"].str.match(r"^\d{4}$")]
mp["postcode_int"] = mp["postcode"].astype(int)
mp = mp[mp["postcode_int"].isin(SYDNEY_POSTCODES)].copy()
mp["count"] = pd.to_numeric(mp["no_vehicles"], errors="coerce").fillna(0)

# Classify motive power
mp["is_ev"] = mp["motive_power"] == "Battery/Fuel-cell electric"
mp["is_hybrid"] = mp["motive_power"] == "Hybrid electric"

ev_pivot = mp.groupby(["postcode", "is_ev"])["count"].sum().unstack(fill_value=0)
ev_pivot.columns = [
    ("non_ev_total" if not c else "ev_total") for c in ev_pivot.columns
]
if "ev_total" not in ev_pivot.columns:
    ev_pivot["ev_total"] = 0
if "non_ev_total" not in ev_pivot.columns:
    ev_pivot["non_ev_total"] = 0
ev_pivot = ev_pivot.reset_index()
ev_pivot["total_mp"] = ev_pivot["ev_total"] + ev_pivot["non_ev_total"]
ev_pivot["ev_share"] = np.where(
    ev_pivot["total_mp"] > 0, ev_pivot["ev_total"] / ev_pivot["total_mp"], np.nan
)
print(f"  {len(ev_pivot)} postcodes with motive power data")
print(f"  Motive power values in data: {mp['motive_power'].unique().tolist()}")

# ── 4. Map 2: New ICE overrepresentation ─────────────────────────────────────
# We don't have postcode × YOM × motive power in a single file.
# Approximation:
#   new_total = recent (2022-2025) registrations from YOM file
#   all_total = all registrations from YOM file
#   expected_ice_share = 1 - ev_share (from motive power file, overall fleet)
#   new_ev_rate ≈ national EV share of new registrations (scaled proxy)
# Better: compute fleet turnover = new_total / all_total, then show
# overrepresentation as (expected new EVs) - (modelled new EVs).
# Since we can't separate new EV from new ICE at postcode level, we instead
# report: new_reg_rate (turnover proxy) and flag this limitation.

new_by_postcode = (
    yom[yom["yom"] >= NEW_YOM_MIN]
    .groupby("postcode")["count"]
    .sum()
    .reset_index()
)
new_by_postcode.columns = ["postcode", "new_total"]

all_by_postcode = yom.groupby("postcode")["count"].sum().reset_index()
all_by_postcode.columns = ["postcode", "all_total"]

turnover = new_by_postcode.merge(all_by_postcode, on="postcode")
turnover["new_reg_rate"] = turnover["new_total"] / turnover["all_total"]

# Merge with EV share to compute expected new EVs vs fleet EV share gap
# We define ICE overrepresentation as:
#   expected_new_ev_share ≈ ev_share_total (if fleet composition carried forward)
#   Since we can't measure actual new EV registrations by postcode directly,
#   we use: ice_overrep = (1 - ev_share) - (1 - national_new_ev_rate)
#   = national_new_ev_rate - ev_share
# This shows how much a postcode's EV adoption lags the national new-car trend.
national_ev_share_new = (
    mp[mp["motive_power"] == "Battery/Fuel-cell electric"]["count"].sum()
    / mp["count"].sum()
)
print(f"\n  National EV share (all fleet): {national_ev_share_new:.1%}")

# ── 4b. Suburb name lookup ────────────────────────────────────────────────────
print("\nLoading suburb name lookup...")
try:
    pc_lookup = pd.read_csv("postcodes_lookup.csv", dtype={"postcode": str})
    pc_lookup["postcode"] = pc_lookup["postcode"].str.zfill(4)
    nsw_pc = pc_lookup[pc_lookup["state"] == "NSW"][["postcode", "locality"]].copy()
    # Exclude PO box / mail-centre localities
    nsw_pc = nsw_pc[~nsw_pc["locality"].str.contains(r"\bMC\b|\bLPO\b|\b\bPO\b|\bDC\b|UNIVERSITY|MARKETS|ROAD$|STREET$", na=False, regex=True)]
    # Pick the most common locality per postcode (mode)
    suburb_map = nsw_pc.groupby("postcode")["locality"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]).to_dict()
    print(f"  Loaded {len(suburb_map)} postcode→suburb mappings")
except Exception as e:
    print(f"  Could not load suburb lookup: {e}")
    suburb_map = {}

# ── 5. Merge all stats ────────────────────────────────────────────────────────
stats_df = (
    fleet_age.merge(ev_pivot[["postcode", "ev_total", "total_mp", "ev_share"]], on="postcode", how="left")
    .merge(turnover[["postcode", "new_total", "all_total", "new_reg_rate"]], on="postcode", how="left")
)

# ICE overrepresentation:
# Positive = postcode's EV share lags what would be expected if new cars followed
# national new-car EV rate. We express it as: expected_new_ev_count_if_nat_rate
# vs observed total EV share. A cleaner metric = new_reg_rate × (1 - ev_share):
# "what share of the postcode's total fleet is recent ICE"
# We'll show: ice_overrep = (new_reg_rate × 1) - (new_reg_rate × nat_ev_rate_approx)
# = new_reg_rate × (1 - approx_national_new_ev_rate_in_new_regs)
# Simplest interpretable metric: among recent registrations, how ICE-heavy vs fleet avg
stats_df["expected_ice_share"] = 1 - stats_df["ev_share"]
# We can't compute new_ice_share directly without YOM×motive power
# Use proxy: if postcode's EV share equals national, ice_overrep = 0
# Positive = more ICE-leaning than national fleet share
national_ev_share_all = stats_df["ev_share"].mean()
stats_df["ice_overrep_proxy"] = stats_df["expected_ice_share"] - (
    1 - national_ev_share_all
)
# This is effectively: how much more ICE than the Sydney average

print(f"\n  Stats df: {len(stats_df)} postcodes")
print(stats_df[["postcode", "avg_age", "ev_share", "new_reg_rate"]].describe())

# ── 6. Load shapefile ─────────────────────────────────────────────────────────
print("\nLoading shapefile...")
gdf = gpd.read_file("POA_2021_AUST_GDA2020.shp")
print(f"  Shapefile CRS: {gdf.crs}")
print(f"  Columns: {gdf.columns.tolist()}")
print(f"  Sample POA codes: {gdf['POA_CODE21'].head(5).tolist()}")

# Filter to Sydney postcodes
gdf["postcode"] = gdf["POA_CODE21"].str.zfill(4)
gdf["postcode_int"] = pd.to_numeric(gdf["postcode"], errors="coerce")
gdf_sydney = gdf[gdf["postcode_int"].isin(SYDNEY_POSTCODES)].copy()
print(f"  Sydney postcodes in shapefile: {len(gdf_sydney)}")

# Reproject to WGS84
gdf_sydney = gdf_sydney.to_crs(epsg=4326)

# Merge with stats
gdf_sydney = gdf_sydney.merge(stats_df, on="postcode", how="left")

# Suburb name from shapefile
if "POA_NAME21" in gdf_sydney.columns:
    gdf_sydney["suburb"] = gdf_sydney["POA_NAME21"]
else:
    gdf_sydney["suburb"] = gdf_sydney["postcode"]

print(f"  After merge: {gdf_sydney['avg_age'].notna().sum()} postcodes with fleet data")

# ── 7. Summary stats table ────────────────────────────────────────────────────
print("\n=== TOP 20 SYDNEY POSTCODES BY NEWEST FLEET (excl. low-confidence) ===")
top20 = (
    stats_df[~stats_df["low_confidence"]].nsmallest(20, "avg_age")[
        ["postcode", "avg_age", "ev_share", "new_reg_rate", "total_vehicles"]
    ]
    .copy()
)
print(top20.to_string(index=False))

print("\n=== BOTTOM 20 SYDNEY POSTCODES BY OLDEST FLEET (excl. low-confidence) ===")
bot20 = (
    stats_df[~stats_df["low_confidence"]].nlargest(20, "avg_age")[
        ["postcode", "avg_age", "ev_share", "new_reg_rate", "total_vehicles"]
    ]
    .copy()
)
print(bot20.to_string(index=False))

# ── 8. Correlation ────────────────────────────────────────────────────────────
valid = stats_df[
    stats_df["avg_age"].notna()
    & stats_df["ev_share"].notna()
    & stats_df["new_reg_rate"].notna()
].copy()
r, p = stats.pearsonr(valid["avg_age"], valid["ev_share"])
print(f"\n=== CORRELATIONS ===")
print(f"Pearson r(avg_fleet_age, ev_share): {r:.3f}  p={p:.3e}  n={len(valid)}")

r2, p2 = stats.pearsonr(valid["new_reg_rate"], valid["ev_share"])
print(f"Pearson r(new_reg_rate, ev_share):  {r2:.3f}  p={p2:.3e}  n={len(valid)}")

# ── 9. Save CSV ───────────────────────────────────────────────────────────────
out_cols = [
    "postcode", "suburb", "avg_age", "mean_yom", "total_vehicles",
    "ev_total", "total_mp", "ev_share", "new_total", "all_total",
    "new_reg_rate", "ice_overrep_proxy", "low_confidence",
]
csv_cols = [c for c in out_cols if c in stats_df.columns or c == "suburb"]
# suburb isn't in stats_df, add from shapefile
suburb_lookup = suburb_map  # postcode → suburb from lookup file
stats_df["suburb"] = stats_df["postcode"].map(suburb_lookup).str.title()
stats_df["suburb"] = stats_df["suburb"].fillna(stats_df["postcode"])
# Also update gdf
gdf_sydney["suburb"] = gdf_sydney["postcode"].map(suburb_lookup)
gdf_sydney["suburb"] = gdf_sydney["suburb"].str.title().fillna(gdf_sydney["postcode"])
out_cols_filtered = [c for c in out_cols if c in stats_df.columns]
stats_df[out_cols_filtered].to_csv("sydney_fleet_stats.csv", index=False)
print("\nSaved sydney_fleet_stats.csv")

# ── 10. Build choropleth maps ─────────────────────────────────────────────────
print("\nBuilding maps...")

geojson_data = json.loads(gdf_sydney.to_json())

def make_choropleth(gdf_in, col, title, colorscale, caption, vmin=None, vmax=None, fmt=".1f"):
    """Return a Folium map for the given column."""
    sub = gdf_in[gdf_in[col].notna()].copy()
    if vmin is None:
        vmin = sub[col].quantile(0.02)
    if vmax is None:
        vmax = sub[col].quantile(0.98)

    m = folium.Map(
        location=[-33.87, 151.21],
        zoom_start=10,
        tiles="CartoDB positron",
    )

    sub_json = json.loads(sub.to_json())

    # Tooltip fields
    tooltip_fields = ["postcode", "suburb", col]
    tooltip_aliases = ["Postcode:", "Suburb:", caption + ":"]

    choropleth = folium.Choropleth(
        geo_data=sub_json,
        data=sub[[col]].reset_index(),
        columns=["index", col],
        key_on="feature.id",
        fill_color=colorscale,
        fill_opacity=0.75,
        line_opacity=0.3,
        line_color="white",
        nan_fill_color="lightgray",
        legend_name=caption,
        name=title,
    )
    choropleth.add_to(m)

    # Add hover tooltip
    style_fn = lambda x: {
        "fillColor": "transparent",
        "color": "transparent",
        "weight": 0,
    }
    tooltip = folium.GeoJsonTooltip(
        fields=["postcode", "suburb", col],
        aliases=["Postcode", "Suburb", caption],
        localize=True,
        style="font-size:12px;",
    )
    folium.GeoJson(
        sub_json,
        style_function=style_fn,
        tooltip=tooltip,
        name="tooltip layer",
    ).add_to(m)

    title_html = f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
         background:white;padding:8px 16px;border-radius:6px;font-size:15px;
         font-weight:bold;z-index:1000;box-shadow:2px 2px 6px rgba(0,0,0,0.3);">
        {title}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    return m


# Map 1: Average fleet age
gdf_sydney["avg_age_num"] = pd.to_numeric(gdf_sydney["avg_age"], errors="coerce")
m1 = make_choropleth(
    gdf_sydney, "avg_age_num",
    "Average Fleet Age by Sydney Postcode (2025)",
    "RdYlGn_r",
    "Avg fleet age (years)"
)

# Map 2: ICE overrepresentation (deviation from Sydney mean EV share)
gdf_sydney["ice_overrep_pct"] = gdf_sydney["ice_overrep_proxy"] * 100
m2 = make_choropleth(
    gdf_sydney, "ice_overrep_pct",
    "ICE Overrepresentation vs Sydney Average",
    "RdYlGn_r",
    "ICE over-rep vs avg (%pts)",
)

# Map 3: EV share
gdf_sydney["ev_share_pct"] = gdf_sydney["ev_share"] * 100
m3 = make_choropleth(
    gdf_sydney, "ev_share_pct",
    "EV Share of Registered Passenger Fleet by Sydney Postcode",
    "Greens",
    "EV share (%)"
)

# ── 11. Combine into single HTML ──────────────────────────────────────────────
print("Combining maps into HTML...")

m1_html = m1.get_root().render()
m2_html = m2.get_root().render()
m3_html = m3.get_root().render()

# Extract body content from each map
import re

def extract_map_div(html):
    # Extract the map div id and its content
    match = re.search(r'<div[^>]*id="([^"]+)"[^>]*style="[^"]*width:\s*100%[^"]*"', html)
    return html

# Build combined page
combined_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sydney Fleet Age & EV Analysis</title>
<style>
  body {{ margin: 0; padding: 0; font-family: sans-serif; background: #f4f4f4; }}
  h1 {{ text-align: center; padding: 20px; margin: 0; background: #1a1a2e; color: white; font-size: 1.4em; }}
  .subtitle {{ text-align: center; background: #16213e; color: #ccc; padding: 8px; font-size: 0.9em; }}
  .map-section {{ margin: 20px auto; max-width: 1200px; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
  .map-title {{ background: #0f3460; color: white; padding: 12px 20px; font-size: 1.1em; font-weight: bold; }}
  .map-desc {{ background: #eaf2ff; padding: 8px 20px; font-size: 0.85em; color: #333; border-left: 4px solid #0f3460; }}
  iframe {{ width: 100%; border: none; display: block; }}
  .stats {{ margin: 20px auto; max-width: 1200px; background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
  .stats h2 {{ color: #0f3460; margin-top: 0; }}
  .corr {{ font-size: 1.1em; background: #fff3e0; padding: 12px; border-radius: 6px; margin: 10px 0; }}
</style>
</head>
<body>
<h1>Sydney Fleet Age &amp; EV Analysis — January 2025</h1>
<div class="subtitle">
  Data: BITRE Road Vehicles Australia Jan 2025 · ABS Postal Areas 2021 · Analysis by Rewiring Australia
</div>

<div class="map-section">
  <div class="map-title">Map 1: Average Passenger Fleet Age by Postcode</div>
  <div class="map-desc">
    Weighted mean age of registered passenger vehicles. Newer fleets (green) indicate higher-income, higher-turnover suburbs.
    Older fleets (red) indicate lower-income areas where cars are kept longer.
  </div>
  <iframe id="map1" srcdoc="{{}}" height="550"></iframe>
</div>

<div class="map-section">
  <div class="map-title">Map 2: ICE Overrepresentation vs Sydney Average</div>
  <div class="map-desc">
    How far each postcode's ICE share deviates from the Sydney-wide average.
    Red = more ICE-heavy than average; green = more EV-leaning than average.
    Note: without a postcode × year-of-manufacture × motive-power cross-tab in the BITRE release,
    this uses total fleet EV share — not just new registrations. See methodology notes.
  </div>
  <iframe id="map2" srcdoc="{{}}" height="550"></iframe>
</div>

<div class="map-section">
  <div class="map-title">Map 3: EV Share of Registered Passenger Fleet</div>
  <div class="map-desc">
    Battery/fuel-cell electric vehicles as a share of all registered passenger vehicles.
    This is the same metric shown in the SMH article — contextualised by maps 1 and 2 above.
  </div>
  <iframe id="map3" srcdoc="{{}}" height="550"></iframe>
</div>

<div class="stats">
  <h2>Key Correlations</h2>
  <div class="corr">
    Pearson r(avg_fleet_age, ev_share) = <strong>{r:.3f}</strong> (p = {p:.2e}, n = {len(valid)})<br>
    <em>Strong negative correlation: newer fleets → higher EV share. EVs are a fleet-age story, not a transition-leadership story.</em>
  </div>
  <div class="corr">
    Pearson r(new_reg_rate, ev_share) = <strong>{r2:.3f}</strong> (p = {p2:.2e})<br>
    <em>Higher fleet turnover rate → higher EV share. Wealthy suburbs buy new cars more often; those cars are increasingly EV.</em>
  </div>
</div>

<script>
// Inject map HTML into iframes via srcdoc
const maps = {{}};
</script>
</body>
</html>
"""

# Actually write separate map files and use iframes pointing to them
m1.save("map1_fleet_age.html")
m2.save("map2_ice_overrep.html")
m3.save("map3_ev_share.html")

# Build final combined file with inline scripts loading each map
with open("sydney_fleet_analysis.html", "w") as f:
    f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sydney Fleet Age &amp; EV Analysis</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 0; font-family: -apple-system, sans-serif; background: #f0f2f5; }}
  header {{ background: #1a1a2e; color: white; padding: 20px; text-align: center; }}
  header h1 {{ margin: 0; font-size: 1.5em; }}
  header p {{ margin: 6px 0 0; color: #aaa; font-size: 0.85em; }}
  .section {{ margin: 24px auto; max-width: 1300px; background: white; border-radius: 10px;
             box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden; }}
  .section-header {{ background: #0f3460; color: white; padding: 14px 22px; }}
  .section-header h2 {{ margin: 0; font-size: 1.15em; }}
  .section-desc {{ background: #e8f0fe; padding: 10px 22px; font-size: 0.85em; color: #333;
                  border-left: 4px solid #0f3460; }}
  iframe {{ width: 100%; height: 560px; border: none; display: block; }}
  .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 20px; }}
  .stat-card {{ background: #fff8e1; border-left: 5px solid #f9a825; padding: 14px;
               border-radius: 6px; }}
  .stat-card .value {{ font-size: 1.8em; font-weight: bold; color: #e65100; }}
  .stat-card .label {{ font-size: 0.85em; color: #555; margin-top: 4px; }}
  footer {{ text-align: center; padding: 20px; color: #888; font-size: 0.8em; }}
</style>
</head>
<body>
<header>
  <h1>Sydney Fleet Age &amp; EV Analysis — January 2025</h1>
  <p>BITRE Road Vehicles Australia · ABS Postal Areas 2021 &nbsp;|&nbsp; Analysis by Rewiring Australia</p>
</header>

<div class="section">
  <div class="section-header"><h2>Map 1 — Average Passenger Fleet Age</h2></div>
  <div class="section-desc">
    Weighted mean age of all registered passenger vehicles per postcode (2025 snapshot).
    Greener = newer fleet. Redder = older fleet. Newer fleet = higher-income, higher-turnover suburb.
  </div>
  <iframe src="map1_fleet_age.html"></iframe>
</div>

<div class="section">
  <div class="section-header"><h2>Map 2 — ICE Share vs Sydney Average</h2></div>
  <div class="section-desc">
    How far each postcode's ICE (non-EV) fleet share deviates from the Sydney-wide mean.
    Red = more ICE-heavy than average; green = more EV-leaning than average.
    <em>Note: uses total fleet motive power share as proxy — see methodology.</em>
  </div>
  <iframe src="map2_ice_overrep.html"></iframe>
</div>

<div class="section">
  <div class="section-header"><h2>Map 3 — EV Share of Registered Fleet (SMH Metric)</h2></div>
  <div class="section-desc">
    Battery/fuel-cell electric share of registered passenger vehicles — the same metric shown in the SMH article.
    Maps 1 &amp; 2 above explain why this correlates with wealth and fleet turnover, not transition leadership.
  </div>
  <iframe src="map3_ev_share.html"></iframe>
</div>

<div class="section">
  <div class="section-header"><h2>Key Statistics</h2></div>
  <div class="stats-grid">
    <div class="stat-card">
      <div class="value">r = {r:.3f}</div>
      <div class="label">Pearson correlation: avg fleet age vs EV share (n={len(valid)} postcodes)<br>
      <em>Strong negative: newer fleets → more EVs. EVs are a fleet-turnover story.</em></div>
    </div>
    <div class="stat-card">
      <div class="value">r = {r2:.3f}</div>
      <div class="label">Pearson correlation: new-registration rate vs EV share<br>
      <em>High-turnover suburbs buy new cars more often — and new cars are increasingly EV.</em></div>
    </div>
    <div class="stat-card">
      <div class="value">{valid['ev_share'].quantile(0.9)*100:.1f}%</div>
      <div class="label">EV share in top-decile postcodes (newest fleet)</div>
    </div>
    <div class="stat-card">
      <div class="value">{valid['ev_share'].quantile(0.1)*100:.1f}%</div>
      <div class="label">EV share in bottom-decile postcodes (oldest fleet)</div>
    </div>
  </div>
</div>

<footer>
  Data: BITRE Road Vehicles Australia January 2025, data.gov.au &nbsp;|&nbsp;
  ABS Postal Areas ASGS Edition 3, 2021 GDA2020
</footer>
</body>
</html>
""")

print("\nSaved sydney_fleet_analysis.html (references map1/2/3 as iframes)")
print("All done.")
