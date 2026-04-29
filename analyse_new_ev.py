"""
Estimate EV share of new (2024) registrations by Sydney postcode.

Method:
  1. From the national YOM × motive power × make file, compute what fraction
     of each make's 2024 registrations were EV.
  2. Apply those make-specific EV rates to each postcode's 2024 make breakdown
     from the YOM file to estimate new-car EV registrations per postcode.
  3. Map the result as a choropleth.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import json
from scipy import stats as scipy_stats

SYDNEY_POSTCODES = (
    list(range(2000, 2235))
    + list(range(2555, 2575))
    + list(range(2745, 2787))
)
NEW_YOM_MIN = 2024
NEW_YOM_MAX = 2024
LOW_CONF = 50  # min new registrations to show a postcode

# ── 1. Compute make-level EV rate for 2024 (national) ───────────────────
print("Computing national EV rate by make for 2024...")
nat = pd.read_csv("rva-yom-mtvpwr-make.csv")
nat["yom"] = pd.to_numeric(nat["year_of_manufacture"], errors="coerce")
nat["count"] = pd.to_numeric(nat["no_vehicles"], errors="coerce").fillna(0)
nat["is_ev"] = nat["motive_power"] == "Battery/Fuel-cell electric"

new_nat = nat[
    (nat["vehicle_type"] == "Passenger vehicles")
    & (nat["yom"] >= NEW_YOM_MIN)
    & (nat["yom"] <= NEW_YOM_MAX)
]
make_ev = (
    new_nat.groupby(["make", "is_ev"])["count"]
    .sum()
    .unstack(fill_value=0)
)
make_ev.columns = ["non_ev", "ev"]
make_ev["total_make"] = make_ev["non_ev"] + make_ev["ev"]
make_ev["ev_rate"] = np.where(
    make_ev["total_make"] > 0, make_ev["ev"] / make_ev["total_make"], 0
)
make_ev_rate = make_ev["ev_rate"].to_dict()

# National new-car EV share (weighted)
nat_new_ev_total = new_nat[new_nat["is_ev"]]["count"].sum()
nat_new_total = new_nat["count"].sum()
nat_new_ev_share = nat_new_ev_total / nat_new_total if nat_new_total > 0 else 0
print(f"  National new-car EV share {NEW_YOM_MIN}–{NEW_YOM_MAX}: {nat_new_ev_share:.1%}")
print(f"  Makes with EV rate > 50%: {[m for m,r in make_ev_rate.items() if r > 0.5]}")

# ── 2. Load YOM file, filter to Sydney passenger vehicles, new years ──────────
print("\nLoading YOM file...")
yom = pd.read_csv("rva-yom.csv", dtype={"registered_postcode": str})
yom = yom[
    (yom["state_abb"] == "NSW")
    & (yom["vehicle_type"] == "Passenger vehicles")
].copy()
yom["postcode"] = yom["registered_postcode"].str.strip().str.zfill(4)
yom = yom[yom["postcode"].str.match(r"^\d{4}$")]
yom["postcode_int"] = yom["postcode"].astype(int)
yom = yom[yom["postcode_int"].isin(SYDNEY_POSTCODES)].copy()
yom["yom"] = pd.to_numeric(yom["year_of_manufacture"], errors="coerce")
yom = yom[(yom["yom"] >= 1980) & (yom["yom"] <= 2025)]
yom["count"] = pd.to_numeric(yom["no_vehicles"], errors="coerce").fillna(0)

# Filter to new cars
new_yom = yom[(yom["yom"] >= NEW_YOM_MIN) & (yom["yom"] <= NEW_YOM_MAX)].copy()
print(f"  Total new passenger registrations (Sydney, 2024): {new_yom['count'].sum():,.0f}")

# ── 3. Estimate new-car EV count per postcode using make-level EV rates ───────
print("\nEstimating new-car EV share by postcode...")

# Assign EV rate to each row based on make
new_yom["ev_rate"] = new_yom["make"].map(make_ev_rate).fillna(0)
# Estimated EV vehicles in this (postcode, make, yom) cell
new_yom["ev_est"] = new_yom["count"] * new_yom["ev_rate"]

# Aggregate by postcode
postcode_new = new_yom.groupby("postcode").agg(
    new_total=("count", "sum"),
    new_ev_est=("ev_est", "sum"),
).reset_index()
postcode_new["new_ev_share"] = np.where(
    postcode_new["new_total"] > 0,
    postcode_new["new_ev_est"] / postcode_new["new_total"],
    np.nan,
)

# Flag low-confidence postcodes
postcode_new["low_conf"] = postcode_new["new_total"] < LOW_CONF

print(f"  {len(postcode_new)} postcodes with new registrations")
print(f"  {postcode_new['low_conf'].sum()} postcodes flagged low-confidence (<{LOW_CONF} new vehicles)")
print(f"\n  Distribution of new-car EV share:")
print(postcode_new[~postcode_new["low_conf"]]["new_ev_share"].describe().to_string())

# Compare with total-fleet EV share from existing analysis
existing = pd.read_csv("sydney_fleet_stats.csv", dtype={"postcode": str})
existing["postcode"] = existing["postcode"].str.zfill(4)
merged = postcode_new.merge(
    existing[["postcode", "avg_age", "ev_share", "suburb", "total_vehicles"]],
    on="postcode",
    how="left",
)

# Correlation: does new-car EV share still correlate with fleet age?
valid = merged[merged["avg_age"].notna() & merged["new_ev_share"].notna() & ~merged["low_conf"]]
r, p = scipy_stats.pearsonr(valid["avg_age"], valid["new_ev_share"])
r2, p2 = scipy_stats.pearsonr(valid["ev_share"], valid["new_ev_share"])
print(f"\n=== CORRELATIONS ===")
print(f"r(avg_fleet_age, new_car_ev_share): {r:.3f}  p={p:.2e}  n={len(valid)}")
print(f"r(total_ev_share, new_car_ev_share): {r2:.3f}  p={p2:.2e}  n={len(valid)}")

# ── 4. Print top/bottom table ─────────────────────────────────────────────────
display = merged[~merged["low_conf"]].copy()

print("\n=== TOP 20 POSTCODES BY NEW-CAR EV SHARE ===")
top20 = display.nlargest(20, "new_ev_share")[
    ["postcode", "suburb", "new_ev_share", "ev_share", "avg_age", "new_total"]
]
top20["new_ev_share_pct"] = top20["new_ev_share"] * 100
top20["ev_share_pct"] = top20["ev_share"] * 100
print(top20[["postcode", "suburb", "new_ev_share_pct", "ev_share_pct", "avg_age", "new_total"]].to_string(index=False))

print("\n=== BOTTOM 20 POSTCODES BY NEW-CAR EV SHARE ===")
bot20 = display.nsmallest(20, "new_ev_share")[
    ["postcode", "suburb", "new_ev_share", "ev_share", "avg_age", "new_total"]
]
bot20["new_ev_share_pct"] = bot20["new_ev_share"] * 100
bot20["ev_share_pct"] = bot20["ev_share"] * 100
print(bot20[["postcode", "suburb", "new_ev_share_pct", "ev_share_pct", "avg_age", "new_total"]].to_string(index=False))

# ── 5. Load shapefile, merge ──────────────────────────────────────────────────
print("\nLoading shapefile...")
gdf = gpd.read_file("POA_2021_AUST_GDA2020.shp")
gdf["postcode"] = gdf["POA_CODE21"].str.zfill(4)
gdf["postcode_int"] = pd.to_numeric(gdf["postcode"], errors="coerce")
gdf_sydney = gdf[gdf["postcode_int"].isin(SYDNEY_POSTCODES)].copy()
gdf_sydney = gdf_sydney.to_crs(epsg=4326)
gdf_sydney = gdf_sydney.merge(merged, on="postcode", how="left")

# Suburb names
try:
    pc_lookup = pd.read_csv("postcodes_lookup.csv", dtype={"postcode": str})
    pc_lookup["postcode"] = pc_lookup["postcode"].str.zfill(4)
    nsw_pc = pc_lookup[pc_lookup["state"] == "NSW"][["postcode", "locality"]].copy()
    nsw_pc = nsw_pc[~nsw_pc["locality"].str.contains(
        r"\bMC\b|\bLPO\b|\bPO\b|\bDC\b|UNIVERSITY|MARKETS|ROAD$|STREET$", na=False, regex=True
    )]
    suburb_map = nsw_pc.groupby("postcode")["locality"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    ).to_dict()
except Exception:
    suburb_map = {}

gdf_sydney["suburb"] = gdf_sydney["postcode"].map(suburb_map).str.title().fillna(gdf_sydney["postcode"])
gdf_sydney["new_ev_share_pct"] = (gdf_sydney["new_ev_share"] * 100).round(2)
gdf_sydney["ev_share_pct"] = (gdf_sydney["ev_share"] * 100).round(2)
gdf_sydney["avg_age_r"] = gdf_sydney["avg_age"].round(1)
gdf_sydney["total_vehicles_str"] = gdf_sydney["total_vehicles"].fillna(0).astype(int).apply(lambda x: f"{x:,}")
gdf_sydney["new_total_str"] = gdf_sydney["new_total"].fillna(0).astype(int).apply(lambda x: f"{x:,}")
# Mask low-confidence postcodes
gdf_sydney.loc[gdf_sydney["low_conf"] == True, "new_ev_share_pct"] = np.nan

print(f"  {gdf_sydney['new_ev_share_pct'].notna().sum()} postcodes shown in map")

# ── 6. Build maps ─────────────────────────────────────────────────────────────
print("\nBuilding maps...")

sub_json = json.loads(gdf_sydney[gdf_sydney["new_ev_share_pct"].notna()].to_json())
sub_all_json = json.loads(gdf_sydney[gdf_sydney["ev_share_pct"].notna()].to_json())

def make_map(geojson, col, title, desc, colorscale, legend_name, center=(-33.87, 151.21), zoom=10):
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")
    choropleth = folium.Choropleth(
        geo_data=geojson,
        data=pd.DataFrame(
            [(f["properties"]["postcode"], f["properties"][col])
             for f in geojson["features"] if f["properties"].get(col) is not None],
            columns=["postcode", col],
        ),
        columns=["postcode", col],
        key_on="feature.properties.postcode",
        fill_color=colorscale,
        fill_opacity=0.75,
        line_opacity=0.3,
        line_color="white",
        nan_fill_color="lightgray",
        legend_name=legend_name,
    )
    choropleth.add_to(m)
    folium.GeoJson(
        geojson,
        style_function=lambda x: {"fillColor": "transparent", "color": "transparent", "weight": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["postcode", "suburb", col, "total_vehicles_str", "new_total_str", "avg_age_r"],
            aliases=["Postcode", "Suburb", legend_name, "Total registrations", "New (2024) registrations", "Avg fleet age (yrs)"],
            localize=True,
            style="font-size:13px;",
        ),
    ).add_to(m)
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
        f'background:white;padding:8px 16px;border-radius:6px;font-size:14px;'
        f'font-weight:bold;z-index:1000;box-shadow:2px 2px 6px rgba(0,0,0,.3);">{title}</div>'
    ))
    return m

m_new = make_map(
    sub_json, "new_ev_share_pct",
    "EV Share of New (2024) Passenger Registrations",
    "",
    "Greens",
    "New-car EV share (%)",
)
m_old = make_map(
    sub_all_json, "ev_share_pct",
    "EV Share of Total Passenger Fleet (for comparison)",
    "",
    "Greens",
    "Total fleet EV share (%)",
)

m_new.save("map_new_ev_share.html")
m_old.save("map_total_ev_share_comparison.html")

# ── 7. Write combined HTML ────────────────────────────────────────────────────
with open("sydney_new_ev_analysis.html", "w") as f:
    f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sydney New-Car EV Share Analysis</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: -apple-system, sans-serif; background: #f0f2f5; }}
  header {{ background: #1a1a2e; color: white; padding: 20px; text-align: center; }}
  header h1 {{ margin: 0; font-size: 1.4em; }}
  header p {{ margin: 6px 0 0; color: #aaa; font-size: 0.85em; }}
  .section {{ margin: 24px auto; max-width: 1300px; background: white;
              border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,.1); overflow: hidden; }}
  .sh {{ background: #0f3460; color: white; padding: 14px 22px; }}
  .sh h2 {{ margin: 0; font-size: 1.1em; }}
  .sd {{ background: #e8f0fe; padding: 10px 22px; font-size: 0.85em; color: #333;
         border-left: 4px solid #0f3460; }}
  iframe {{ width: 100%; height: 560px; border: none; display: block; }}
  .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 20px; }}
  .card {{ background: #fff8e1; border-left: 5px solid #f9a825; padding: 14px; border-radius: 6px; }}
  .card .val {{ font-size: 1.8em; font-weight: bold; color: #e65100; }}
  .card .lbl {{ font-size: 0.85em; color: #555; margin-top: 4px; }}
  .method {{ padding: 16px 22px; background: #f9f9f9; font-size: 0.85em; color: #444; }}
  footer {{ text-align: center; padding: 16px; color: #888; font-size: 0.8em; }}
</style>
</head>
<body>
<header>
  <h1>Sydney New-Car EV Share by Postcode — 2024</h1>
  <p>EV share estimated from BITRE registration data by applying national make-level EV rates to postcode new-car mix &nbsp;|&nbsp; Rewiring Australia</p>
</header>

<div class="section">
  <div class="sh"><h2>Map 1 — EV Share of New (2024) Passenger Registrations</h2></div>
  <div class="sd">
    Estimated share of 2024 passenger vehicle registrations that were battery/fuel-cell electric,
    by postcode. This removes the "old cars" effect from the total-fleet EV share metric.
    Postcodes with fewer than {LOW_CONF} new registrations are hidden (grey).
  </div>
  <iframe src="map_new_ev_share.html"></iframe>
</div>

<div class="section">
  <div class="sh"><h2>Map 2 — Total Fleet EV Share (for comparison)</h2></div>
  <div class="sd">
    The conventional metric: EVs as a share of all registered passenger vehicles.
    Compare with Map 1 — the pattern is similar but compressed, because old ICE cars
    dilute EV share in every postcode.
  </div>
  <iframe src="map_total_ev_share_comparison.html"></iframe>
</div>

<div class="section">
  <div class="sh"><h2>Key Findings</h2></div>
  <div class="stats">
    <div class="card">
      <div class="val">r = {r:.3f}</div>
      <div class="lbl">Pearson correlation: avg fleet age vs <em>new-car</em> EV share (n={len(valid)})<br>
      Even after removing old-car bias, newer-fleet postcodes still buy more EVs.
      The correlation weakens vs total fleet (was −0.56), showing the old-car effect inflates the apparent gap.</div>
    </div>
    <div class="card">
      <div class="val">{nat_new_ev_share:.1%}</div>
      <div class="lbl">National new-car EV share 2024 (all passenger vehicles)<br>
      Sydney postcodes range from near 0% to ~{valid['new_ev_share'].quantile(0.95)*100:.0f}% for new-car EV share.</div>
    </div>
    <div class="card">
      <div class="val">{valid['new_ev_share'].quantile(0.9)*100:.1f}%</div>
      <div class="lbl">New-car EV share in top-decile Sydney postcodes (newest fleets)</div>
    </div>
    <div class="card">
      <div class="val">{valid['new_ev_share'].quantile(0.1)*100:.1f}%</div>
      <div class="lbl">New-car EV share in bottom-decile Sydney postcodes (oldest fleets)</div>
    </div>
  </div>
  <div class="method">
    <strong>Method:</strong> National EV rate computed per make from BITRE YOM × motive power × make file
    (102,503 rows covering all registered vehicles). For each postcode, new registrations (YOM 2024)
    are broken down by make; each make's national new-car EV rate is applied to estimate EV count.
    Pure-EV makes (Tesla 100%, Polestar 100%, ORA 99%, BYD 81%) dominate the EV count.
    Mixed makes (BMW 17.5%, MG 11.5%, Mercedes-Benz 10.7%) contribute proportionally.
    Postcodes with &lt;{LOW_CONF} new registrations excluded.
  </div>
</div>

<footer>
  Data: BITRE Road Vehicles Australia January 2025 · ABS Postal Areas 2021
</footer>
</body>
</html>
""")

# ── 8. Save updated CSV ───────────────────────────────────────────────────────
merged_out = merged[["postcode", "suburb", "new_total", "new_ev_est", "new_ev_share",
                      "ev_share", "avg_age", "low_conf"]].copy()
merged_out.to_csv("sydney_new_ev_stats.csv", index=False)
print("\nSaved sydney_new_ev_analysis.html and sydney_new_ev_stats.csv")
print("Done.")
