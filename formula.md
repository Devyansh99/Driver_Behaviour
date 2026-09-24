# Sensor Algorithm Mathematics — Road Anomaly Detection (Canonical)

> **Canonical doc for SadakVision trigger mechanism.** This file is the single source of truth for the trigger described in `SadakVision_corrected (1).docx` §III-A-1 (page 4, Eqs. 1-4) and matches `v3/pothole_v3_prophet_zaccel.ipynb` exactly. Other md files have been archived to `_archive_md/` to remove redundancy.

Complete math reference for V1 → V2 → V3 → Pawar → Hybrid pipelines.
All code truth is in `v3/pothole_v3_prophet_zaccel.ipynb`, `pawar_new/hybrid_pawar_v3_fixed.ipynb`, `pawar_new/v3_overlap_fix.py`.
Word mapping: Eq1 (sensor model + speed regressor) → §4.1, Eq2a (anomaly outside bounds) → §4.2, Eq3a (deviation D(t)) → §4.3, Eq4 (Dmin/Dmax + ±0.5 thresholds) → §4.3 flags.

---

## 1. Sensor Model

| Sensor | Columns | Unit | Sampling |
|--------|---------|------|----------|
| Accelerometer (linear) | `x, y, z` | m/s² | ~100 Hz |
| TotalAcceleration | `x, y, z` | m/s² (includes gravity) | ~100 Hz |
| Gyroscope | `x, y, z` | rad/s | ~100 Hz |
| GPS Location | `lat, lon, alt, speed` | deg, deg, m, m/s | ~1 Hz |

**Gravity reference:**
```
g = 9.81 m/s²
```
- On smooth road: `Z_total ≈ 9.81` (phone Z-axis aligned with vehicle vertical)
- Z drop below 9.81 → wheel falls into hole (dip)
- Z rise above 9.81 → bounce-back (bump)

**Coordinate axes (phone frame, vehicle-mounted):**
- X: longitudinal (negative = braking, positive = acceleration)
- Y: lateral
- Z: vertical

---

## 2. Derived Signals (Magnitudes)

### 2.1 Gyroscope magnitude (vehicle rocking)
```
gyro_mag(t) = √( gx(t)² + gy(t)² + gz(t)² )          [rad/s]
gyro_peak   = max_t gyro_mag(t)                        # max rocking over segment
gyro_rms    = mean_t gyro_mag(t)                       # average rocking over segment
```

### 2.2 Acceleration magnitude (used in Pawar features)
```
mag(t) = √( x(t)² + y(t)² + z(t)² )                   [m/s²]
```

### 2.3 Z deviation from gravity
```
total_z_dev = max_t | Z_total(t) − 9.81 |             [m/s²]
dip_mag     = max( 0, 9.81 − min_t Z_total(t) )       [m/s²]   # downward drop
bump_mag    = max( 0, max_t Z_total(t) − 9.81 )       [m/s²]   # upward rise
```

---

## 3. Speed Context (GPS interpolation)

GPS speed (1 Hz) interpolated onto sensor timestamps (100 Hz):

```
v_sensor(t) = np.interp( t, t_gps, v_gps )            # linear interpolation
```

Start / stop phase masks (windows of 3 s):

```
is_start_phase(t) = ( v_sensor(τ) < 1 m/s  for some τ ∈ [t−3, t] )
is_stop_phase(t)  = ( v_sensor(τ) < 1 m/s  for some τ ∈ [t, t+3] )
```

**Filters (reject segment):**
```
reject if  mean(v_segment) < 2.0 m/s       # engine idle / start-stop jerks
reject if  any is_start_phase(segment)     # momentum change after stop
reject if  any is_stop_phase(segment)      # braking to stop
```

---

## 4. V3 Detection — Prophet on Z-axis

### 4.1 Model

```
y(t) = Z_total(t)            # response
ds(t) = timestamp            # time
x(t)  = v_sensor(t)          # extra regressor: speed

y(t) = trend(t) + β · v_sensor(t) + ε(t)
```

Prophet hyperparameters:
```
interval_width      = 0.99      # 99% prediction interval
changepoint_prior   = 0.15      # trend adaptability
seasonality         = none      # daily/weekly/yearly disabled
```

### 4.2 Anomaly decision (per sample t)

```
ŷ(t)      = Prophet mean forecast
ŷ_low(t)  = lower 99% bound
ŷ_high(t) = upper 99% bound

anomaly(t) = [ y(t) < ŷ_low(t) ]  OR  [ y(t) > ŷ_high(t) ]
```

### 4.3 Prophet deviation (severity raw signal)

```
dev(t) =   y(t) − ŷ_high(t)    if y(t) > ŷ_high(t)   (bump, positive)
        =   y(t) − ŷ_low(t)    if y(t) < ŷ_low(t)    (dip, negative)
        =   0                  otherwise
```

Per anomaly segment (group of consecutive anomaly samples):
```
max_dip_dev  = min_t dev(t)      # negative → deepest dip below bound
max_bump_dev = max_t dev(t)      # positive → highest bump above bound
abs_max_dev  = max_t |dev(t)|    # overall severity proxy
```

**Event detection flags (thresholds):**
```
has_dip  = (max_dip_dev  < −0.5)   # i.e. Z ≤ 9.31 m/s²  →  5% drop from g
has_bump = (max_bump_dev >  0.5)   # i.e. Z ≥ 10.31 m/s² →  5% rise from g
dip_mag  = |max_dip_dev|
bump_mag =  max_bump_dev
```

---

## 5. Segment Features (per detected anomaly segment)

| Feature | Formula | Physical meaning |
|---------|---------|------------------|
| `duration` | `t_end − t_start` | time over anomaly |
| `z_min`, `z_max`, `z_mean`, `z_std` | stats of `Z_total` in segment | Z profile |
| `gyro_peak` | `max √(gx²+gy²+gz²)` | max rocking |
| `gyro_rms` | `mean √(gx²+gy²+gz²)` | avg rocking |
| `accel_x_min` | `min X_linear` | braking force |
| `accel_x_max` | `max X_linear` | acceleration force |
| `accel_x_peak` | `max \|X_linear\|` | worst longitudinal force |
| `braking_before` | `min X ∈ [t−1, t) < −1.5` | braking in the 1 s before |
| `braking_after` | `min X ∈ (t, t+1] < −1.5` | braking in the 1 s after |
| `speed` | mean interpolated GPS speed | context |
| `latitude, longitude` | GPS at segment midpoint | location |

**Segment midpoint time:** `t_mid = (t_start + t_end) / 2`

---

## 6. Z-axis Kurtosis (Hybrid only)

Sharpness of the Z signal in the event window — separates impulsive potholes (kurtosis ≈ 6–10) from gradual bumps (kurtosis ≈ 0–2).

```
Excess kurtosis (Fisher):
                          (1/n) Σ (zᵢ − z̄)⁴
z_kurt =  ────────────────────────────────────  −  3
          [ (1/n) Σ (zᵢ − z̄)² ]²
```

Validation (from results):

| Event type | Avg kurtosis |
|------------|--------------|
| Pothole | 6.37 |
| Speed Bump | 0.18 |
| Road Bump | 0.02 |

---

## 7. Pawar ANN (Supervised ML, hybrid candidate generator)

### 7.1 Windowing (2 s sliding windows, 100 Hz → ~200 samples)
```
window_w = 2 s
windows slide by 2 s (non-overlapping)
```

### 7.2 16 features per window
For each axis `a ∈ {X, Y, Z}` and magnitude `M = √(x²+y²+z²)`:
```
f = { min(a), max(a), mean(a), std(a) }     → 4 features × 4 signals = 16
```

### 7.3 Z-score standardization
```
z_score = (f − μ_train) / σ_train
```
(train statistics from India_Data — makes model transferable across devices/units)

### 7.4 Architecture
```
Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
```
```
p = σ( ANN(z_scored_features) )        # ∈ [0, 1]  pothole probability
```

### 7.5 Candidate gate
```
candidate_window  ⇔  p ≥ 0.4          # relaxed threshold, physics validates later
```

---

## 8. Hybrid Classification (rule priority)

Inputs: `p` (Pawar prob), `dip`, `bump`, `tz_dev`, `z_kurt`, `g_peak`, `x_peak`, `x_min`, `x_max`, `brk_bef`, `brk_aft`.

Gate flags:
```
has_dip  = dip  > 0.5
has_bump = bump > 0.5
```

### Priority 1 — Hard Braking / Acceleration (driver behavior)
```
IF  x_peak > 2.0  AND  g_peak < 0.3  AND  tz_dev < 2.0 :
    sev  = min(10, x_peak · 1.5)
    IF x_min < −2.5  → 'Hard Braking', conf 0.80
    IF x_max >  2.5  → 'Acceleration', conf 0.75
    ELSE → 'Hard Braking' if |x_min| > x_max else 'Acceleration', conf 0.70

ELIF brk_bef AND g_peak < 0.3 AND tz_dev < 2.0 :
    'Hard Braking', sev = min(10, x_peak · 1.2), conf 0.65
```

### Priority 2 — Pothole (high confidence)
```
IF has_dip AND ( g_peak > 0.5  OR  (g_peak > 0.3 AND z_kurt > 5) ):
    sev  = min(10, dip·1.0 + g_peak·2.0 + min(z_kurt,10)·0.2)
    conf = min(0.95, 0.55 + dip·0.03 + g_peak·0.15 + p·0.1 + min(z_kurt,10)·0.01)
    IF brk_aft: conf = min(1.0, conf + 0.05)

IF has_dip AND dip > 5.0 AND z_kurt > 3:
    sev  = min(10, dip·0.8 + min(z_kurt,10)·0.3)
    conf = min(0.80, 0.50 + dip·0.02 + p·0.1)
```

### Priority 3 — Road Bump (bump only)
```
IF has_bump AND NOT has_dip:
    sev = min(10, bump·1.5)
    conf = 0.70 if bump > 2.0 else 0.55
```

### Priority 4 — Speed Bump (gentle oscillation: low gyro + low kurtosis)
```
IF has_dip AND has_bump AND g_peak < 0.3 AND z_kurt < 2:
    total_osc = dip + bump
    sev = min(10, total_osc·0.7)
    'Speed Bump' conf 0.65 if total_osc > 3.0  else 'Road Bump' conf 0.50
```

### Priority 5 — Pothole (moderate)
```
IF has_dip AND g_peak > 0.3 AND z_kurt > 3:
    sev  = min(10, dip·0.8 + g_peak·2.0 + p·1.0)
    conf = min(0.80, 0.45 + g_peak·0.15 + p·0.1)

IF has_dip AND z_kurt > 5 AND dip > 2.0:
    sev  = min(10, dip·0.8 + min(z_kurt,10)·0.3)
    conf = min(0.70, 0.40 + p·0.15 + dip·0.02)

IF has_dip AND p > 0.8 AND dip > 1.5 AND z_kurt > 3:
    sev  = min(10, dip·0.8 + p·2.0)
    conf = min(0.70, 0.40 + p·0.15)
```

### Priority 6 — Road/Speed Bump (moderate gyro, low kurtosis)
```
IF has_dip AND has_bump AND z_kurt < 3 AND g_peak < 0.5:
    total_osc = dip + bump
    sev = min(10, total_osc·0.6)
    'Speed Bump' conf 0.55  if total_osc > 4.0
    'Road Bump'  conf 0.45  if bump ≥ dip·0.7
    'Rough Patch' sev min(5, total_osc·0.4) conf 0.35  otherwise
```

### Priority 7 — Speed Bump catchall
```
IF has_dip AND has_bump AND g_peak > 0.3 AND z_kurt < 3:
    sev = min(10, (dip+bump)·0.6), conf 0.50
```

### Priority 8 — Rough Patch / Road Bump (low energy)
```
IF has_dip:  'Rough Patch', sev = min(5, dip·0.8),  conf 0.30
IF has_bump: 'Road Bump',   sev = min(5, bump·0.8), conf 0.35
```

### Priority 9 — Possible Pothole (ML confident, physics ambiguous)
```
IF p > 0.8:
    'Possible Pothole', sev = min(6, tz_dev·0.8 + p·2), conf 0.40
```

### Priority 10 — Minor Anomaly (catchall)
```
'Minor Anomaly', sev = min(4, tz_dev·0.5), conf 0.20
```

---

## 9. V3 (standalone) Classification

Uses Prophet deviations instead of raw gravity deviation. `has_dip = dip_dev < −0.5`, `has_bump = bump_dev > 0.5`.

| # | Condition | Label | Severity | Confidence |
|---|-----------|-------|----------|------------|
| 1 | `x_peak > 2.0 AND g_peak < 0.3 AND abs_dev < 2.0` | Hard Braking / Acceleration | `min(10, x_peak·1.5)` | 0.70–0.80 |
| 2 | `brk_bef AND g_peak < 0.3 AND abs_dev < 2.0` | Hard Braking | `min(10, x_peak·1.2)` | 0.65 |
| 3 | `has_bump AND NOT has_dip` | Speed Breaker if `dur > 0.15 AND bump_mag > 2.0`, else Road Bump | `min(10, bump_mag·1.5)` | 0.70 / 0.50 |
| 4 | `has_dip AND has_bump AND dur > 0.15 AND g_peak < 0.8` and `(dip+bump) > 3.0` | Speed Breaker | `min(10, (dip+bump)·0.8)` | 0.65 |
| 5 | `has_dip AND g_peak > 0.3` | **Pothole** | `min(10, dip_mag·1.2 + g_peak·2.0)` | `min(0.95, 0.65 + dip_mag·0.05 + g_peak·0.15)` (+0.05 if brk_aft) |
| 6 | `has_dip` (low gyro) | Rough Patch | — | — |
| 7 | none | Minor Anomaly | — | — |

---

## 10. DBSCAN Spatial Clustering

```
DBSCAN( eps = 20 m , min_samples = 1 )
```
Per event type separately (pothole ≠ braking merge).

**Haversine distance between events:**
```
a = sin²(Δφ/2) + cos φ₁ · cos φ₂ · sin²(Δλ/2)
c = 2 · atan2( √a, √(1−a) )
d = R · c          with R = 6,371,000 m
```

Cluster merge: keep max severity, max confidence, sum merged count.

---

## 11. V3 Overlap Fix (post-processing)

### 11.1 Merge pothole + bounce-back bump (10 m radius)
```
for each Road Bump b:
    find closest Pothole p on same route with d(p, b) ≤ 10 m
    if found → absorb b into p (p gains bump features, severity, count)
```
Rationale: same physical pothole produces Z-dip then Z-bounce 2–5 m apart; Prophet splits them.

### 11.2 Fake Hard Braking reclassification
```
keep as Hard Braking  iff  max_accel_x ≥ 2.0
weak events (max_accel_x < 2.0):
    has bump, no dip  → Road Bump
    has dip           → Rough Patch
    otherwise         → Minor Anomaly
```

---

## 12. Complete Threshold Reference

| Threshold | Value | Physical meaning |
|-----------|-------|------------------|
| Gravity `g` | 9.81 m/s² | baseline Z on smooth road |
| Speed filter | < 2.0 m/s reject | engine idle / start-stop |
| Start/stop phase | speed < 1 m/s in ±3 s | momentum transitions |
| Dip detection | Z ≤ 9.31 m/s² (5% drop) | `dev < −0.5` |
| Bump detection | Z ≥ 10.31 m/s² (5% rise) | `dev > +0.5` |
| Deep dip | Z ≤ 6.81 m/s² (30% drop) | `dip > 3.0` |
| Pothole gyro | > 0.3 rad/s (17°/s) | felt rocking |
| Violent rocking | > 0.5 rad/s (29°/s) | high-confidence pothole |
| Speed-bump gyro cap | < 0.8 rad/s (46°/s) | above = deep hole, not hump |
| Braking force | \|X\| > 2.0 m/s² (0.2g) | moderate braking |
| Firm braking | \|X\| > 2.5 m/s² (0.25g) | confirmed hard braking |
| Total oscillation | dip + bump > 3.0 m/s² | up-down-up hump pattern |
| Pawar candidate gate | p ≥ 0.4 | catch borderline |
| Pawar confidence boost | p > 0.8 | trust ML when physics weak |
| Kurtosis (pothole) | > 3–5 (avg 6.37) | sharp impulse |
| Kurtosis (bump) | < 2 (avg 0.18) | gradual oscillation |
| Prophet interval | 0.99 (99%) | only extreme 1% flagged |
| DBSCAN eps | 20 m (10 m for merge) | GPS accuracy 5–10 m |
| Window | 2 s | at 30 km/h ≈ 17 m path |

---

## 13. Key Constants (code)

```
g               = 9.81        m/s²
MIN_SPEED       = 2.0         m/s
DIP_THRESHOLD   = 0.5         m/s²  below g
BUMP_THRESHOLD  = 0.5         m/s²  above g
GYRO_POTHOLE    = 0.3         rad/s
GYRO_VIOLENT    = 0.5         rad/s
GYRO_SB_CAP     = 0.8         rad/s
BRAKE_X         = 2.0         m/s²
FIRM_BRAKE_X    = 2.5         m/s²
PAWAR_THRESHOLD = 0.4         (prob)
DBSCAN_EPS      = 20          m
MERGE_EPS       = 10          m
WINDOW          = 2           s
PROPHET_IW      = 0.99
PROPHET_CP      = 0.15
```
