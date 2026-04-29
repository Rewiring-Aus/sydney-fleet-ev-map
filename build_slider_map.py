"""
Build a self-contained HTML with a year slider (2018–2024) showing
estimated EV share of new passenger registrations by Sydney postcode.
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd

SYDNEY_POSTCODES = set(
    list(range(2000, 2235)) + list(range(2555, 2575)) + list(range(2745, 2787))
)
YEARS = list(range(2018, 2025))
MIN_NEW = 20  # postcodes with fewer new cars in a given year get NaN

# ── 1. Make-level EV rate per year ────────────────────────────────────────────
print("Computing make-level EV rates per year...")
nat = pd.read_csv("rva-yom-mtvpwr-make.csv")
nat["yom"] = pd.to_numeric(nat["year_of_manufacture"], errors="coerce")
nat["count"] = pd.to_numeric(nat["no_vehicles"], errors="coerce").fillna(0)
nat["is_ev"] = nat["motive_power"] == "Battery/Fuel-cell electric"
nat_pv = nat[nat["vehicle_type"] == "Passenger vehicles"]

make_ev_by_year = {}  # year → {make: ev_rate}
for yr in YEARS:
    sub = nat_pv[nat_pv["yom"] == yr]
    by_make = sub.groupby(["make", "is_ev"])["count"].sum().unstack(fill_value=0)
    by_make.columns = ["non_ev", "ev"]
    by_make["total"] = by_make["non_ev"] + by_make["ev"]
    by_make["ev_rate"] = np.where(by_make["total"] > 0, by_make["ev"] / by_make["total"], 0)
    make_ev_by_year[yr] = by_make["ev_rate"].to_dict()
    nat_share = sub[sub["is_ev"]]["count"].sum() / sub["count"].sum()
    print(f"  {yr}: national new-car EV share {nat_share:.1%}")

# ── 2. Postcode new-car EV share per year ─────────────────────────────────────
print("\nComputing postcode EV share per year...")
yom = pd.read_csv("rva-yom.csv", dtype={"registered_postcode": str})
yom = yom[(yom["state_abb"] == "NSW") & (yom["vehicle_type"] == "Passenger vehicles")].copy()
yom["postcode"] = yom["registered_postcode"].str.strip().str.zfill(4)
yom = yom[yom["postcode"].str.match(r"^\d{4}$")]
yom["postcode_int"] = yom["postcode"].astype(int)
yom = yom[yom["postcode_int"].isin(SYDNEY_POSTCODES)].copy()
yom["yom"] = pd.to_numeric(yom["year_of_manufacture"], errors="coerce")
yom["count"] = pd.to_numeric(yom["no_vehicles"], errors="coerce").fillna(0)

# Also load avg_age and total_vehicles from existing stats
stats = pd.read_csv("sydney_fleet_stats.csv", dtype={"postcode": str})
stats["postcode"] = stats["postcode"].str.zfill(4)

postcode_years = {}  # postcode → {year: {"ev_share": x, "new_total": n}}

for yr in YEARS:
    sub = yom[yom["yom"] == yr].copy()
    sub["ev_rate"] = sub["make"].map(make_ev_by_year[yr]).fillna(0)
    sub["ev_est"] = sub["count"] * sub["ev_rate"]
    agg = sub.groupby("postcode").agg(
        new_total=("count", "sum"),
        ev_est=("ev_est", "sum"),
    ).reset_index()
    agg["ev_share"] = np.where(agg["new_total"] >= MIN_NEW, agg["ev_est"] / agg["new_total"], np.nan)
    for _, row in agg.iterrows():
        pc = row["postcode"]
        if pc not in postcode_years:
            postcode_years[pc] = {}
        postcode_years[pc][yr] = {
            "ev_share": round(row["ev_share"] * 100, 2) if not np.isnan(row["ev_share"]) else None,
            "new_total": int(row["new_total"]),
        }

# ── 3. Load shapefile + suburb lookup ─────────────────────────────────────────
print("\nLoading shapefile...")
gdf = gpd.read_file("POA_2021_AUST_GDA2020.shp")
gdf["postcode"] = gdf["POA_CODE21"].str.zfill(4)
gdf["postcode_int"] = pd.to_numeric(gdf["postcode"], errors="coerce")
gdf_sydney = gdf[gdf["postcode_int"].isin(SYDNEY_POSTCODES)].copy()
gdf_sydney = gdf_sydney.to_crs(epsg=4326)

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

stats_lookup = stats.set_index("postcode")[["avg_age", "total_vehicles"]].to_dict("index")

# ── 4. Build GeoJSON with all-year data embedded ──────────────────────────────
print("Building GeoJSON...")
features = []
for _, row in gdf_sydney.iterrows():
    pc = row["postcode"]
    suburb = suburb_map.get(pc, pc)
    suburb = suburb.title() if isinstance(suburb, str) else pc
    st = stats_lookup.get(pc, {})
    avg_age = round(st.get("avg_age", None) or 0, 1)
    total_veh = int(st.get("total_vehicles", None) or 0)

    props = {
        "postcode": pc,
        "suburb": suburb,
        "avg_age": avg_age,
        "total_vehicles": total_veh,
    }
    yr_data = postcode_years.get(pc, {})
    for yr in YEARS:
        yd = yr_data.get(yr, {})
        props[f"ev_{yr}"] = yd.get("ev_share")     # None = masked
        props[f"new_{yr}"] = yd.get("new_total", 0)

    geom = row["geometry"].__geo_interface__
    features.append({"type": "Feature", "properties": props, "geometry": geom})

geojson_str = json.dumps({"type": "FeatureCollection", "features": features})
print(f"  {len(features)} features, GeoJSON size: {len(geojson_str)/1e6:.1f} MB")

# ── 5. National EV share + per-year scale bounds ──────────────────────────────
nat_shares = {}
year_scales = {}  # year → {lo, hi, mean} based on postcode distribution
for yr in YEARS:
    sub = nat_pv[nat_pv["yom"] == yr]
    share = sub[sub["is_ev"]]["count"].sum() / sub["count"].sum()
    nat_shares[yr] = round(share * 100, 1)

    # Gather all non-null postcode EV shares for this year
    vals = [
        postcode_years[pc][yr]["ev_share"]
        for pc in postcode_years
        if yr in postcode_years[pc] and postcode_years[pc][yr]["ev_share"] is not None
    ]
    vals = sorted(vals)
    n = len(vals)
    lo  = round(np.percentile(vals, 5),  1)
    hi  = round(np.percentile(vals, 95), 1)
    mean = round(np.mean(vals), 1)
    year_scales[yr] = {"lo": lo, "hi": hi, "mean": mean}
    print(f"  {yr} scale: p5={lo}%  mean={mean}%  p95={hi}%  (n={n} postcodes)")

# ── 6. Write HTML ─────────────────────────────────────────────────────────────
print("Writing HTML...")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sydney New-Car EV Share 2018–2024</title>
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
  #nat-share {{ font-size: 0.8em; color: #aaa; white-space: nowrap; flex-shrink: 0; }}
  #play-btn {{
    flex-shrink: 0; background: #4fc3f7; border: none; border-radius: 50%;
    width: 36px; height: 36px; cursor: pointer; font-size: 16px; line-height: 36px;
    text-align: center; color: #0f3460; font-weight: bold; transition: background 0.15s;
  }}
  #play-btn:hover {{ background: #81d4fa; }}
  #map {{ flex: 1; }}
  .legend {{ background: white; color: #333; padding: 10px 14px; border-radius: 6px; line-height: 1.6; font-size: 12px; }}
  .legend-title {{ font-weight: bold; margin-bottom: 6px; font-size: 13px; }}
  .legend-bar {{ width: 180px; height: 12px; border-radius: 3px; margin-bottom: 4px;
    background: linear-gradient(to right, #f7fcf5, #74c476, #00441b); }}
  .legend-labels {{ display: flex; justify-content: space-between; font-size: 11px; color: #666; }}
  .info-box {{ background: white; color: #333; padding: 10px 14px; border-radius: 6px; font-size: 13px; min-width: 180px; }}
  .info-box b {{ font-size: 15px; }}
  .info-box .metric {{ font-size: 1.6em; font-weight: 700; color: #1b5e20; margin: 4px 0; }}
  .info-box .row {{ display: flex; justify-content: space-between; gap: 10px; color: #555; font-size: 12px; border-top: 1px solid #eee; padding-top: 5px; margin-top: 5px; }}
  .year-ticks {{ display: flex; justify-content: space-between; font-size: 11px; color: #aaa; margin-top: 3px; padding: 0 2px; }}
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
    <div id="nat-share">Sydney-wide avg EV share: {year_scales[2024]['mean']}%</div>
  </div>
</div>
<div id="map"></div>

<script>
const GEOJSON = {geojson_str};
const NAT_SHARES = {json.dumps(nat_shares)};
const YEAR_SCALES = {json.dumps(year_scales)};
const YEARS = {json.dumps(YEARS)};

// Colour scale — domain updated per year
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

const map = L.map('map', {{
  center: [-33.87, 151.10],
  zoom: 10,
  zoomControl: true,
}});

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_nolabels/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '© OpenStreetMap contributors © CARTO',
  maxZoom: 18,
}}).addTo(map);

// Info box (hover)
const info = L.control({{ position: 'topright' }});
info.onAdd = function() {{
  this._div = L.DomUtil.create('div', 'info-box');
  this.update();
  return this._div;
}};
info.update = function(props, year) {{
  if (!props) {{
    this._div.innerHTML = '<b>Hover over a postcode</b>';
    return;
  }}
  const ev = props['ev_' + year];
  const newTotal = props['new_' + year] || 0;
  const evStr = ev != null ? ev.toFixed(1) + '%' : 'n/a';
  this._div.innerHTML = `
    <b>${{props.suburb}}</b> <span style="color:#888;font-size:11px">${{props.postcode}}</span><br>
    <div class="metric">${{evStr}}</div>
    <span style="font-size:11px;color:#555">EV share of ${{year}} new cars</span>
    <div class="row">
      <span>${{year}} registrations</span><span><b>${{newTotal.toLocaleString()}}</b></span>
    </div>
    <div class="row">
      <span>Total fleet</span><span><b>${{(props.total_vehicles||0).toLocaleString()}}</b></span>
    </div>
    <div class="row">
      <span>Avg fleet age</span><span><b>${{props.avg_age}} yrs</b></span>
    </div>
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
    <div class="legend-labels">
      <span id="leg-lo">–</span><span id="leg-mid">–</span><span id="leg-hi">–</span>
    </div>
    <div style="margin-top:5px;font-size:10px;color:#999">p5 · mean · p95 across postcodes</div>
    <div style="margin-top:3px;font-size:10px;color:#999">Grey = &lt;{MIN_NEW} registrations</div>
  `;
  return div;
}};
legend.addTo(map);

let currentYear = 2024;
let geojsonLayer;
updateScale(currentYear);  // legend is in DOM now

function styleFeature(feature, year) {{
  const val = feature.properties['ev_' + year];
  return {{
    fillColor: val != null ? scale(val).hex() : '#cccccc',
    fillOpacity: val != null ? 0.78 : 0.35,
    color: 'white',
    weight: 0.5,
    opacity: 0.8,
  }};
}}

function onEachFeature(feature, layer) {{
  layer.on({{
    mouseover: function(e) {{
      // Only change stroke — never touch fillColor or Leaflet will
      // fall back to the stroke colour for the fill.
      e.target.setStyle({{ weight: 2.5, opacity: 1, color: '#444' }});
      e.target.bringToFront();
      info.update(feature.properties, currentYear);
    }},
    mouseout: function(e) {{
      // Re-apply the full style directly rather than using resetStyle,
      // which has unreliable fillColor behaviour in Leaflet 1.9.
      e.target.setStyle(styleFeature(e.target.feature, currentYear));
      info.update();
    }},
  }});
}}

function buildLayer(year) {{
  return L.geoJSON(GEOJSON, {{
    style: f => styleFeature(f, year),
    onEachFeature,
  }});
}}

// Initial render
geojsonLayer = buildLayer(currentYear);
geojsonLayer.addTo(map);

// Slider + play/pause
const slider   = document.getElementById('slider');
const yearLabel = document.getElementById('year-label');
const natShare  = document.getElementById('nat-share');
const playBtn   = document.getElementById('play-btn');

function goToYear(yr) {{
  currentYear = yr;
  slider.value = yr;
  yearLabel.textContent = yr;
  natShare.textContent = 'Sydney-wide avg EV share: ' + YEAR_SCALES[yr].mean + '%';
  updateScale(yr);
  info.update();
  geojsonLayer.setStyle(f => styleFeature(f, currentYear));
}}

slider.addEventListener('input', function() {{
  stopPlay();
  goToYear(parseInt(this.value));
}});

// Play / pause
const FIRST_YEAR = YEARS[0];
const LAST_YEAR  = YEARS[YEARS.length - 1];
const STEP_MS    = 1000;
const PAUSE_MS   = 4000;
let playTimer = null;

function isPlaying() {{ return playTimer !== null; }}

function stopPlay() {{
  if (playTimer) {{ clearTimeout(playTimer); playTimer = null; }}
  playBtn.textContent = '▶';
}}

function scheduleNext(delayMs) {{
  playTimer = setTimeout(stepPlay, delayMs);
}}

function stepPlay() {{
  playTimer = null;
  const next = currentYear < LAST_YEAR ? currentYear + 1 : FIRST_YEAR;
  goToYear(next);
  // Pause for longer on the last year, then loop
  scheduleNext(next === LAST_YEAR ? PAUSE_MS : STEP_MS);
}}

playBtn.addEventListener('click', function() {{
  if (isPlaying()) {{
    stopPlay();
  }} else {{
    playBtn.textContent = '⏸';
    // If already on last year, restart from beginning
    if (currentYear === LAST_YEAR) goToYear(FIRST_YEAR);
    scheduleNext(STEP_MS);
  }}
}});
</script>
</body>
</html>
"""

with open("sydney_ev_slider.html", "w") as f:
    f.write(html)

size_mb = len(html) / 1e6
print(f"Saved sydney_ev_slider.html ({size_mb:.1f} MB)")
print("Done.")
