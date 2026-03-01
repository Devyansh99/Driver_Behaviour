"""
V3 Improved Fix — 10m merge radius + reclassify weak Hard Braking

Changes from original V3:
1. Merge radius reduced from 20m to 10m (more conservative)
2. Weak Hard Braking (accel_x < 2.0) reclassified based on actual features
3. Hard Braking near potholes → absorbed if within 10m
"""
import pandas as pd
import numpy as np
import folium
import os
from math import radians, sin, cos, sqrt, atan2

# ── Load V3 results ──────────────────────────────────────────────────
v3_path = os.path.join('..', 'v3', 'road_events_v3.csv')
v3 = pd.read_csv(v3_path)
print(f'V3 events loaded: {len(v3)}')
print('\nOriginal distribution:')
print(v3['event_type'].value_counts().to_string())

DATA_ROOT = os.path.join('..', '')
folders = ['A', 'B', 'C', 'D', 'E', 'F']
route_gps = {}
for f in folders:
    route_gps[f] = pd.read_csv(os.path.join(DATA_ROOT, f, 'Location.csv'))

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Reclassify weak Hard Braking (accel_x < 2.0)
# ═══════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print('STEP 1: Reclassify weak Hard Braking')
print(f'{"="*60}')

df = v3.copy()
hb_mask = df['event_type'] == 'Hard Braking'
hb = df[hb_mask]
strong_hb = hb[hb['max_accel_x'] >= 2.0]
weak_hb = hb[hb['max_accel_x'] < 2.0]

print(f'Hard Braking total: {len(hb)}')
print(f'  Strong (accel_x >= 2.0): {len(strong_hb)} → keep as Hard Braking')
print(f'  Weak (accel_x < 2.0): {len(weak_hb)} → reclassify')

# Reclassify weak Hard Braking based on their actual features
reclassified = {'Minor Anomaly': 0, 'Road Bump': 0, 'Rough Patch': 0}
for idx in weak_hb.index:
    row = df.loc[idx]
    dip_dev = row['max_dip_dev']
    bump_dev = row['max_bump_dev']
    has_dip = dip_dev < -0.5
    has_bump = bump_dev > 0.5

    if has_bump and not has_dip:
        df.at[idx, 'event_type'] = 'Road Bump'
        df.at[idx, 'confidence'] = 0.50
        reclassified['Road Bump'] += 1
    elif has_dip:
        df.at[idx, 'event_type'] = 'Rough Patch'
        df.at[idx, 'confidence'] = 0.30
        reclassified['Rough Patch'] += 1
    else:
        df.at[idx, 'event_type'] = 'Minor Anomaly'
        df.at[idx, 'confidence'] = 0.20
        reclassified['Minor Anomaly'] += 1

print(f'\nReclassification results:')
for etype, n in reclassified.items():
    print(f'  → {etype}: {n}')

print(f'\nAfter Step 1:')
print(df['event_type'].value_counts().to_string())


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Merge overlapping Pothole + Road Bump within 10m
# ═══════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print('STEP 2: Merge overlapping Pothole + Road Bump (10m radius)')
print(f'{"="*60}')

MERGE_RADIUS = 10  # meters

df = df.reset_index(drop=True)
pothole_idx = df[df['event_type'] == 'Pothole'].index.tolist()
bump_idx = df[df['event_type'] == 'Road Bump'].index.tolist()

# For each bump, find closest pothole within radius on same route
bump_to_pothole = {}
for bi in bump_idx:
    b = df.loc[bi]
    best_pi, best_dist = None, float('inf')
    for pi in pothole_idx:
        p = df.loc[pi]
        if p['route'] != b['route']:
            continue
        d = haversine_m(p['latitude'], p['longitude'], b['latitude'], b['longitude'])
        if d <= MERGE_RADIUS and d < best_dist:
            best_dist = d
            best_pi = pi
    if best_pi is not None:
        bump_to_pothole[bi] = (best_pi, best_dist)

# Group bumps by target pothole
pothole_bumps = {}
for bi, (pi, dist) in bump_to_pothole.items():
    pothole_bumps.setdefault(pi, []).append(bi)

# Merge bump features into pothole
for pi, bump_list in pothole_bumps.items():
    for bi in bump_list:
        b = df.loc[bi]
        p = df.loc[pi]
        df.at[pi, 'n_detections'] = int(p['n_detections']) + int(b['n_detections'])
        df.at[pi, 'severity'] = max(float(p['severity']), float(b['severity']))
        df.at[pi, 'max_bump_dev'] = max(float(p['max_bump_dev']), float(b['max_bump_dev']))
        df.at[pi, 'max_gyro_peak'] = max(float(p['max_gyro_peak']), float(b['max_gyro_peak']))
        df.at[pi, 'abs_max_dev'] = max(float(p['abs_max_dev']), float(b['abs_max_dev']))
        df.at[pi, 'total_duration'] = float(p['total_duration']) + float(b['total_duration'])
        df.at[pi, 'confidence'] = max(float(p['confidence']), float(b['confidence']))

absorbed_bumps = set(bump_to_pothole.keys())
df = df.drop(index=absorbed_bumps).reset_index(drop=True)

print(f'Road Bumps absorbed into Potholes: {len(absorbed_bumps)}')
print(f'Potholes that absorbed bumps: {len(pothole_bumps)}')


# ═══════════════════════════════════════════════════════════════════════
# FINAL RESULTS
# ═══════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print('FINAL RESULTS')
print(f'{"="*60}')

print(f'\n{"Event Type":25s}  {"Original":>8s}  →  {"Fixed":>8s}  {"Change":>8s}')
print(f'{"-"*60}')
all_types = sorted(set(v3['event_type'].unique()) | set(df['event_type'].unique()))
for etype in all_types:
    before = len(v3[v3['event_type'] == etype])
    after = len(df[df['event_type'] == etype])
    diff = after - before
    sign = '+' if diff >= 0 else ''
    print(f'  {etype:23s}: {before:5d}  →  {after:5d}  ({sign}{diff})')
print(f'  {"─"*54}')
print(f'  {"TOTAL":23s}: {len(v3):5d}  →  {len(df):5d}')

# Per route breakdown
print(f'\nPer-route breakdown:')
for route in folders:
    r1 = v3[v3['route'] == route]
    r2 = df[df['route'] == route]
    print(f'\n  Route {route}: {len(r1)} → {len(r2)}')
    for etype in all_types:
        b = len(r1[r1['event_type'] == etype])
        a = len(r2[r2['event_type'] == etype])
        if b > 0 or a > 0:
            print(f'    {etype:20s}: {b:3d} → {a:3d}')


# ── Save ──────────────────────────────────────────────────────────────
df.to_csv('v3_events_fixed.csv', index=False)
print(f'\nSaved → v3_events_fixed.csv ({len(df)} events)')


# ── Generate Map ──────────────────────────────────────────────────────
EVENT_COLORS = {
    'Pothole': 'red',
    'Road Bump': '#22AA22',
    'Speed Breaker': '#FFD700',
    'Rough Patch': '#8B4513',
    'Hard Braking': 'orange',
    'Acceleration': '#4488CC',
    'Minor Anomaly': '#AAAAAA',
}
EVENT_OPACITY = {
    'Pothole': 0.9,
    'Road Bump': 0.5,
    'Speed Breaker': 0.7,
    'Rough Patch': 0.4,
    'Hard Braking': 0.3,
    'Acceleration': 0.25,
    'Minor Anomaly': 0.15,
}
ROUTE_COLORS = {'A': 'red', 'B': 'blue', 'C': 'green', 'D': 'orange', 'E': 'purple', 'F': 'darkred'}

centre = [df['latitude'].mean(), df['longitude'].mean()]
m = folium.Map(location=centre, zoom_start=14, tiles='OpenStreetMap')

for folder in folders:
    loc = route_gps[folder]
    coords = list(zip(loc['latitude'], loc['longitude']))
    folium.PolyLine(coords, color=ROUTE_COLORS[folder], weight=4,
                    opacity=0.6, tooltip=f'Route {folder}').add_to(m)

# --- Layer groups by event type ---
layer_groups = {}
for etype in EVENT_COLORS:
    fg = folium.FeatureGroup(name=etype)
    layer_groups[etype] = fg
    m.add_child(fg)

for _, row in df.iterrows():
    etype = row['event_type']
    color = EVENT_COLORS.get(etype, 'gray')
    opacity = EVENT_OPACITY.get(etype, 0.2)
    sev = row['severity']
    conf = row['confidence']

    if etype == 'Pothole':
        radius = max(8, min(20, sev * 2.2))
    elif etype in ('Speed Breaker', 'Road Bump'):
        radius = max(5, min(12, sev * 1.3))
    else:
        radius = 3

    popup_text = (f'<b>{etype}</b> (Route {row["route"]})<br>'
                  f'Severity: {sev:.1f}/10<br>'
                  f'Confidence: {conf:.0%}<br>'
                  f'Prophet dip dev: {row.get("max_dip_dev", 0):.2f} m/s²<br>'
                  f'Prophet bump dev: {row.get("max_bump_dev", 0):.2f} m/s²<br>'
                  f'Max gyro: {row["max_gyro_peak"]:.3f} rad/s<br>'
                  f'Detections merged: {int(row["n_detections"])}<br>'
                  f'Speed: {row["avg_speed"]:.1f} m/s ({row["avg_speed"]*3.6:.0f} km/h)<br>'
                  f'<a href="https://www.google.com/maps?q={row["latitude"]},{row["longitude"]}" '
                  f'target="_blank">Google Maps</a>')

    fg = layer_groups.get(etype)
    if fg is None:
        fg = folium.FeatureGroup(name=etype)
        layer_groups[etype] = fg
        m.add_child(fg)

    folium.CircleMarker(
        [row['latitude'], row['longitude']],
        radius=radius, color=color, fill=True,
        fill_color=color, fill_opacity=opacity,
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=f'{etype} | sev={sev:.1f} | conf={conf:.0%}'
    ).add_to(fg)

# Legend
legend_html = '''
<div style="position:fixed; bottom:30px; left:30px; z-index:9999;
            background:white; padding:12px; border:2px solid gray;
            border-radius:8px; font-size:13px; opacity:0.92;">
<b>V3 Fixed — Road Surface</b><br>
<span style="color:red">&#11044;</span> Pothole (Z dip + gyro)<br>
<span style="color:#22AA22">&#11044;</span> Road Bump (Z uplift)<br>
<span style="color:#FFD700">&#11044;</span> Speed Breaker<br>
<span style="color:#8B4513">&#11044;</span> Rough Patch<br>
<br><b>Driver Behavior</b><br>
<span style="color:orange">&#11044;</span> Hard Braking<br>
<span style="color:#4488CC">&#11044;</span> Acceleration<br>
<span style="color:#AAAAAA">&#11044;</span> Minor Anomaly<br>
<br><b>Routes</b><br>
<span style="color:red">&#9473;</span> A &nbsp;
<span style="color:blue">&#9473;</span> B &nbsp;
<span style="color:green">&#9473;</span> C<br>
<span style="color:orange">&#9473;</span> D &nbsp;
<span style="color:purple">&#9473;</span> E &nbsp;
<span style="color:darkred">&#9473;</span> F
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))
folium.LayerControl(collapsed=False).add_to(m)
m.save('v3_fixed_map.html')
print(f'Saved → v3_fixed_map.html')
print('\nDone.')
