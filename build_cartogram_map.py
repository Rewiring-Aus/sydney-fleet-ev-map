"""
Bubble cartogram: EV share of new registrations by Sydney postcode, 2018-2024.
  - Background: Voronoi cells (from postcode centroids, clipped to Sydney)
  - Foreground: circles sized by sqrt(total_vehicles), coloured by EV share
  - Year slider + play/pause
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd

SYDNEY_POSTCODES = set(
    list(range(2000, 2235)) + list(range(2555, 2575)) + list(range(2745, 2787))
)
YEARS = list(range(2018, 2025))
MIN_NEW = 20

# ── 1. Make-level EV rate per year (same as before) ───────────────────────────
print("Computing make-level EV rates per year...")
nat = pd.read_csv("rva-yom-mtvpwr-make.csv")
nat["yom"] = pd.to_numeric(nat["year_of_manufacture"], errors="coerce")
nat["count"] = pd.to_numeric(nat["no_vehicles"], errors="coerce").fillna(0)
nat["is_ev"] = nat["motive_power"] == "Battery/Fuel-cell electric"
nat_pv = nat[nat["vehicle_type"] == "Passenger vehicles"]

make_ev_by_year = {}
for yr in YEARS:
    sub = nat_pv[nat_pv["yom"] == yr]
    by_make = sub.groupby(["make", "is_ev"])["count"].sum().unstack(fill_value=0)
    by_make.columns = ["non_ev", "ev"]
    by_make["total"] = by_make["non_ev"] + by_make["ev"]
    by_make["ev_rate"] = np.where(by_make["total"] > 0, by_make["ev"] / by_make["total"], 0)
    make_ev_by_year[yr] = by_make["ev_rate"].to_dict()

# ── 2. Postcode new-car EV share per year ─────────────────────────────────────
print("Computing postcode EV share per year...")
yom = pd.read_csv("rva-yom.csv", dtype={"registered_postcode": str})
yom = yom[(yom["state_abb"] == "NSW") & (yom["vehicle_type"] == "Passenger vehicles")].copy()
yom["postcode"] = yom["registered_postcode"].str.strip().str.zfill(4)
yom = yom[yom["postcode"].str.match(r"^\d{4}$")]
yom["postcode_int"] = yom["postcode"].astype(int)
yom = yom[yom["postcode_int"].isin(SYDNEY_POSTCODES)].copy()
yom["yom"] = pd.to_numeric(yom["year_of_manufacture"], errors="coerce")
yom["count"] = pd.to_numeric(yom["no_vehicles"], errors="coerce").fillna(0)

postcode_years = {}
for yr in YEARS:
    sub = yom[yom["yom"] == yr].copy()
    sub["ev_rate"] = sub["make"].map(make_ev_by_year[yr]).fillna(0)
    sub["ev_est"] = sub["count"] * sub["ev_rate"]
    agg = sub.groupby("postcode").agg(new_total=("count", "sum"), ev_est=("ev_est", "sum")).reset_index()
    agg["ev_share"] = np.where(agg["new_total"] >= MIN_NEW, agg["ev_est"] / agg["new_total"], np.nan)
    for _, row in agg.iterrows():
        pc = row["postcode"]
        if pc not in postcode_years:
            postcode_years[pc] = {}
        postcode_years[pc][yr] = {
            "ev_share": round(row["ev_share"] * 100, 2) if not np.isnan(row["ev_share"]) else None,
            "new_total": int(row["new_total"]),
        }

# Per-year scale bounds
nat_shares, year_scales = {}, {}
for yr in YEARS:
    sub = nat_pv[nat_pv["yom"] == yr]
    nat_shares[yr] = round(sub[sub["is_ev"]]["count"].sum() / sub["count"].sum() * 100, 1)
    vals = sorted([
        postcode_years[pc][yr]["ev_share"]
        for pc in postcode_years
        if yr in postcode_years[pc] and postcode_years[pc][yr]["ev_share"] is not None
    ])
    year_scales[yr] = {
        "lo": 0,
        "hi": round(np.percentile(vals, 95), 1),
        "mean": round(np.mean(vals), 1),
    }
    print(f"  {yr}: mean={year_scales[yr]['mean']}%  p95={year_scales[yr]['hi']}%")

# ── 3. Load shapefile, compute centroids ──────────────────────────────────────
print("\nLoading shapefile + computing centroids...")
gdf = gpd.read_file("POA_2021_AUST_GDA2020.shp")
gdf["postcode"] = gdf["POA_CODE21"].str.zfill(4)
gdf["postcode_int"] = pd.to_numeric(gdf["postcode"], errors="coerce")
gdf_sydney = gdf[gdf["postcode_int"].isin(SYDNEY_POSTCODES)].copy()

# Work in a projected CRS for accurate centroids, then reproject
gdf_proj = gdf_sydney.to_crs(epsg=7856)   # GDA2020 / MGA zone 56
gdf_proj["centroid"] = gdf_proj.geometry.centroid

# ── 4. Simplified boundary polygons for background ────────────────────────────
print("Building background boundary layer...")
# Simplify to ~100m tolerance in projected CRS to keep file size down
gdf_simplified = gdf_proj.copy()
gdf_simplified["geometry"] = gdf_proj.geometry.simplify(100)
gdf_wgs_poly = gdf_simplified[["postcode", "geometry"]].to_crs(epsg=4326)
boundaries_geojson = json.loads(gdf_wgs_poly.to_json())
print(f"  {len(gdf_wgs_poly)} boundary polygons")

# ── 5. Suburb lookup ──────────────────────────────────────────────────────────
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

# ── 6. Build circle data ──────────────────────────────────────────────────────
print("Building circle data...")
stats = pd.read_csv("sydney_fleet_stats.csv", dtype={"postcode": str})
stats["postcode"] = stats["postcode"].str.zfill(4)
stats_lookup = stats.set_index("postcode")[["avg_age", "total_vehicles"]].to_dict("index")

# Centroids in WGS84 — build a clean point GeoDataFrame
centroids_wgs = gpd.GeoDataFrame(
    gdf_proj[["postcode"]].copy(),
    geometry=gdf_proj.geometry.centroid,
    crs=gdf_proj.crs,
).to_crs(epsg=4326)

# Radius scaling: area ∝ fleet size → radius ∝ sqrt(fleet)
# Scale so largest postcode (70929 veh) gets ~1600m radius
K = 3000 / np.sqrt(70929)
MIN_RADIUS = 400  # metres floor

circles = []
for _, row in centroids_wgs.iterrows():
    pc = row["postcode"]
    st = stats_lookup.get(pc, {})
    total_veh = int(st.get("total_vehicles") or 0)
    avg_age = round(float(st.get("avg_age") or 0), 1)
    suburb = suburb_map.get(pc, pc)
    suburb = suburb.title() if isinstance(suburb, str) else pc

    radius_m = max(MIN_RADIUS, K * np.sqrt(total_veh)) if total_veh > 0 else MIN_RADIUS

    yr_data = postcode_years.get(pc, {})
    obj = {
        "postcode": pc,
        "suburb": suburb,
        "lat": round(row.geometry.y, 5),
        "lon": round(row.geometry.x, 5),
        "total_vehicles": total_veh,
        "avg_age": avg_age,
        "radius_m": round(radius_m),
    }
    for yr in YEARS:
        yd = yr_data.get(yr, {})
        obj[f"ev_{yr}"] = yd.get("ev_share")
        obj[f"new_{yr}"] = yd.get("new_total", 0)
    circles.append(obj)

circles_str = json.dumps(circles)
boundaries_str = json.dumps(boundaries_geojson)
print(f"  {len(circles)} circles, data size: {(len(circles_str)+len(boundaries_str))/1e6:.1f} MB")

# ── 7. Write HTML ─────────────────────────────────────────────────────────────
print("Writing HTML...")
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sydney EV Bubble Map 2018–2024</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/chroma-js@2.4.2/chroma.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #1a1a2e; color: white; display: flex; flex-direction: column; height: 100vh; }}
  #header {{ padding: 10px 20px; background: #0f3460; display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
  #header h1 {{ font-size: 1.05em; font-weight: 600; white-space: nowrap; }}
  #controls {{ display: flex; align-items: center; gap: 14px; flex: 1; min-width: 0; }}
  #slider-wrap {{ flex: 1; min-width: 0; }}
  #slider {{ width: 100%; accent-color: #4fc3f7; cursor: pointer; height: 6px; display: block; }}
  #year-label {{ font-size: 2em; font-weight: 800; color: #4fc3f7; min-width: 4ch; text-align: center; flex-shrink: 0; }}
  #sydney-share {{ font-size: 0.8em; color: #aaa; white-space: nowrap; flex-shrink: 0; }}
  #play-btn {{
    flex-shrink: 0; background: #4fc3f7; border: none; border-radius: 50%;
    width: 36px; height: 36px; cursor: pointer; font-size: 16px; line-height: 36px;
    text-align: center; color: #0f3460; font-weight: bold; transition: background 0.15s;
  }}
  #play-btn:hover {{ background: #81d4fa; }}
  .year-ticks {{ display: flex; justify-content: space-between; font-size: 11px; color: #aaa; margin-top: 3px; padding: 0 2px; }}
  #map {{ flex: 1; }}
  .legend {{ background: white; color: #333; padding: 10px 14px; border-radius: 6px; font-size: 12px; line-height: 1.6; }}
  .legend-title {{ font-weight: bold; margin-bottom: 6px; font-size: 13px; }}
  .legend-bar {{ width: 180px; height: 12px; border-radius: 3px; margin-bottom: 4px;
    background: linear-gradient(to right, #f7fcf5, #74c476, #00441b); }}
  .legend-labels {{ display: flex; justify-content: space-between; font-size: 11px; color: #666; }}
  .legend-divider {{ border-top: 1px solid #eee; margin: 8px 0; }}
  .bubble-scale {{ display: flex; align-items: flex-end; gap: 8px; margin-top: 4px; }}
  .bubble-scale svg {{ overflow: visible; }}
  .info-box {{ background: white; color: #333; padding: 10px 14px; border-radius: 6px; font-size: 13px; min-width: 190px; }}
  .info-box .metric {{ font-size: 1.6em; font-weight: 700; color: #1b5e20; margin: 4px 0; }}
  .info-box .row {{ display: flex; justify-content: space-between; gap: 10px; color: #555; font-size: 12px; border-top: 1px solid #eee; padding-top: 5px; margin-top: 5px; }}
</style>
</head>
<body>
<div id="header">
  <h1>Sydney new-car EV share</h1>
  <div id="controls">
    <button id="play-btn" title="Play / Pause">▶</button>
    <div id="slider-wrap">
      <input type="range" id="slider" min="2018" max="2024" value="2024" step="1">
      <div class="year-ticks">{' '.join(f'<span>{y}</span>' for y in YEARS)}</div>
    </div>
    <div id="year-label">2024</div>
    <div id="sydney-share">Sydney-wide avg EV share: {year_scales[2024]['mean']}%</div>
  </div>
</div>
<div id="map"></div>

<script>
const CIRCLES = {circles_str};
const BOUNDARIES = {boundaries_str};
const NAT_SHARES = {json.dumps(nat_shares)};
const YEAR_SCALES = {json.dumps(year_scales)};
const YEARS = {json.dumps(YEARS)};

let scale = chroma.scale(['#f7fcf5','#c7e9c0','#74c476','#238b45','#00441b']);

function updateScale(year) {{
  const s = YEAR_SCALES[year];
  scale = chroma.scale(['#f7fcf5','#c7e9c0','#74c476','#238b45','#00441b']).domain([0, s.hi]);
  const loEl  = document.getElementById('leg-lo');
  const midEl = document.getElementById('leg-mid');
  const hiEl  = document.getElementById('leg-hi');
  if (loEl)  loEl.textContent  = '0%';
  if (midEl) midEl.textContent = s.mean + '%';
  if (hiEl)  hiEl.textContent  = s.hi  + '%';
}}

const map = L.map('map', {{ center: [-33.87, 151.10], zoom: 10 }});

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_nolabels/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '© OpenStreetMap contributors © CARTO', maxZoom: 18,
}}).addTo(map);

// Real postcode boundaries — faint grey fill + outline for reference
L.geoJSON(BOUNDARIES, {{
  style: {{ color: '#bbb', weight: 0.6, fillColor: '#e8e8e8', fillOpacity: 0.25, opacity: 0.6 }},
  interactive: false,
}}).addTo(map);

// Info box
const info = L.control({{ position: 'topright' }});
info.onAdd = function() {{
  this._div = L.DomUtil.create('div', 'info-box');
  this.update();
  return this._div;
}};
info.update = function(d, year) {{
  if (!d) {{ this._div.innerHTML = '<b>Hover over a bubble</b>'; return; }}
  const ev = d['ev_' + year];
  const evStr = ev != null ? ev.toFixed(1) + '%' : 'n/a';
  this._div.innerHTML = `
    <b>${{d.suburb}}</b> <span style="color:#888;font-size:11px">${{d.postcode}}</span><br>
    <div class="metric">${{evStr}}</div>
    <span style="font-size:11px;color:#555">EV share of ${{year}} new cars</span>
    <div class="row"><span>${{year}} registrations</span><span><b>${{(d['new_' + year]||0).toLocaleString()}}</b></span></div>
    <div class="row"><span>Total fleet</span><span><b>${{d.total_vehicles.toLocaleString()}}</b></span></div>
    <div class="row"><span>Avg fleet age</span><span><b>${{d.avg_age}} yrs</b></span></div>
  `;
}};
info.addTo(map);

// Legend
const legend = L.control({{ position: 'bottomright' }});
legend.onAdd = function() {{
  const div = L.DomUtil.create('div', 'legend');
  div.innerHTML = `
    <div class="legend-title">EV share of new cars</div>
    <div class="legend-bar"></div>
    <div class="legend-labels"><span id="leg-lo">0%</span><span id="leg-mid">–</span><span id="leg-hi">–</span></div>
    <div style="font-size:10px;color:#999;margin-top:3px">p5 · mean · p95 across postcodes</div>
    <div class="legend-divider"></div>
    <div style="font-size:11px;font-weight:bold;margin-bottom:4px">Bubble = total fleet size</div>
    <div class="bubble-scale">
      <svg width="110" height="44">
        <circle cx="10"  cy="40" r="4"  fill="#ccc" stroke="#999" stroke-width="0.5"/>
        <circle cx="35"  cy="40" r="10" fill="#ccc" stroke="#999" stroke-width="0.5"/>
        <circle cx="75"  cy="40" r="20" fill="#ccc" stroke="#999" stroke-width="0.5"/>
        <text x="10"  y="34" text-anchor="middle" font-size="8" fill="#666">1k</text>
        <text x="35"  y="27" text-anchor="middle" font-size="8" fill="#666">6k</text>
        <text x="75"  y="17" text-anchor="middle" font-size="8" fill="#666">25k</text>
      </svg>
      <span style="font-size:10px;color:#888">vehicles</span>
    </div>
    <div style="margin-top:6px;font-size:10px;color:#999">Grey = &lt;{MIN_NEW} new registrations</div>
  `;
  return div;
}};
legend.addTo(map);

let currentYear = 2024;
updateScale(currentYear);

// Build circles
let circleLayer = L.layerGroup();

function buildCircles(year) {{
  const lg = L.layerGroup();
  // Sort so smaller circles render on top
  const sorted = [...CIRCLES].sort((a, b) => b.total_vehicles - a.total_vehicles);
  for (const d of sorted) {{
    const val = d['ev_' + year];
    const fillColor = val != null ? scale(val).hex() : '#cccccc';
    const fillOpacity = val != null ? 0.82 : 0.35;
    const c = L.circle([d.lat, d.lon], {{
      radius: d.radius_m,
      fillColor,
      fillOpacity,
      color: 'white',
      weight: 0.8,
      opacity: 0.7,
    }});
    c.on('mouseover', function() {{
      this.setStyle({{ weight: 2.5, opacity: 1, color: '#444' }});
      this.bringToFront();
      info.update(d, currentYear);
    }});
    c.on('mouseout', function() {{
      this.setStyle({{ weight: 0.8, opacity: 0.7, color: 'white' }});
      info.update();
    }});
    c.addTo(lg);
  }}
  return lg;
}}

circleLayer = buildCircles(currentYear);
circleLayer.addTo(map);

function goToYear(yr) {{
  currentYear = yr;
  slider.value = yr;
  yearLabel.textContent = yr;
  sydneyShare.textContent = 'Sydney-wide avg EV share: ' + YEAR_SCALES[yr].mean + '%';
  updateScale(yr);
  info.update();
  // Update circle colours in place
  const sorted = [...CIRCLES].sort((a, b) => b.total_vehicles - a.total_vehicles);
  let i = 0;
  circleLayer.eachLayer(function(c) {{
    const d = sorted[i++];
    if (!d) return;
    const val = d['ev_' + yr];
    c.setStyle({{
      fillColor: val != null ? scale(val).hex() : '#cccccc',
      fillOpacity: val != null ? 0.82 : 0.35,
      color: 'white', weight: 0.8, opacity: 0.7,
    }});
  }});
}}

const slider     = document.getElementById('slider');
const yearLabel  = document.getElementById('year-label');
const sydneyShare = document.getElementById('sydney-share');
const playBtn    = document.getElementById('play-btn');

slider.addEventListener('input', function() {{
  stopPlay();
  goToYear(parseInt(this.value));
}});

const FIRST_YEAR = YEARS[0], LAST_YEAR = YEARS[YEARS.length - 1];
const STEP_MS = 1000, PAUSE_MS = 4000;
let playTimer = null;

function isPlaying() {{ return playTimer !== null; }}
function stopPlay() {{
  if (playTimer) {{ clearTimeout(playTimer); playTimer = null; }}
  playBtn.textContent = '▶';
}}
function scheduleNext(ms) {{ playTimer = setTimeout(stepPlay, ms); }}
function stepPlay() {{
  playTimer = null;
  const next = currentYear < LAST_YEAR ? currentYear + 1 : FIRST_YEAR;
  goToYear(next);
  scheduleNext(next === LAST_YEAR ? PAUSE_MS : STEP_MS);
}}
playBtn.addEventListener('click', function() {{
  if (isPlaying()) {{ stopPlay(); }}
  else {{
    playBtn.textContent = '⏸';
    if (currentYear === LAST_YEAR) goToYear(FIRST_YEAR);
    scheduleNext(STEP_MS);
  }}
}});
</script>
</body>
</html>
"""

with open("sydney_ev_bubbles.html", "w") as f:
    f.write(html)

size_mb = len(html) / 1e6
print(f"\nSaved sydney_ev_bubbles.html ({size_mb:.1f} MB)")
print("Done.")
