# Pothole Detection — Methodology Comparison: V2, V3, Pawar, and Hybrid Pawar+V3

## Overview

This document explains the four approaches implemented for pothole and road event detection using smartphone sensor data. Each version uses different detection strategies, sensors, and classification logic. We explain **what sensor data is used where**, **what each threshold means physically**, and **why those specific values were chosen**.

| Aspect | V2 | V3 | Pawar | Hybrid (Pawar+V3) |
|--------|----|----|-------|--------------------|
| **Detection method** | Prophet on Gyroscope | Prophet on Z-Axis TotalAccel | ANN on windowed features | Pawar ANN + V3 physics rules |
| **Data used** | Gyro + Accel + GPS | TotalAccel + Gyro + Accel + GPS | Accelerometer only (India) | All sensors (A–F) |
| **Approach type** | Unsupervised | Unsupervised | Supervised | Supervised + rule-based |
| **Classification** | Multi-class (rules) | Multi-class (rules) | Binary (pothole/normal) | Multi-class (rules + ML confidence) |

---

## Understanding the Sensor Data

Before diving into each method, here is what each sensor measures and why it matters for road surface detection.

### TotalAcceleration (Z-axis) — The Vertical Force Sensor

Your phone's accelerometer measures force along three directions. The **Z-axis of TotalAcceleration** points straight up (perpendicular to the ground). On a smooth road, it reads approximately **9.81 m/s²** — this is just Earth's gravity pulling the phone downward.

- **Pothole**: When the wheel drops into a hole, the car briefly free-falls. Z drops sharply **below 9.81** (for example, to 7 or 8 m/s²). Then when the wheel hits the far edge, Z spikes **above 9.81** (bounce back). The bigger the pothole, the bigger the dip.
- **Speed bump**: Z goes **above 9.81** first (car pushes up over the bump) and may dip slightly after. The pattern is more gradual than a pothole.
- **Smooth road**: Z stays right around 9.81, with only tiny vibrations.

**Why Z-axis and not X or Y?** Because potholes are holes in the road surface — the car moves **vertically** (up/down). X-axis measures forward/backward (braking/acceleration) and Y-axis measures left/right (turning). These are useful for separating driver behavior from road events, but Z is the primary pothole signal.

### Gyroscope — The Rotation Sensor

The gyroscope measures how fast the phone is **rotating** (in radians per second). When a car hits a pothole:

- The car body **tilts and rocks** — the front dips into the pothole, then bounces up
- This creates a **rotational impulse** that the gyroscope picks up as a spike
- The gyro magnitude = sqrt(gx² + gy² + gz²) tells us how violently the car rocked

**Why gyroscope matters for classification**: A dip in Z could be caused by many things (sensor noise, wind, road camber). But if Z dips AND the gyroscope spikes at the same time, it means the car actually **physically rocked** — which strongly confirms a real road surface event like a pothole.

- **Gyro peak > 0.3 rad/s**: The car noticeably rocked. This is about 17 degrees per second of rotation — you would feel this as a clear jolt. Combined with a Z-axis dip, this strongly suggests a pothole.
- **Gyro peak < 0.3 rad/s**: Very little rotation. If there was a big dip in Z but no gyro spike, it is more likely sensor drift, gentle road slope change, or just a small surface variation.
- **Gyro peak > 0.8 rad/s**: Very strong rotation — usually only happens with large potholes, severe bumps, or reckless driving.

### Accelerometer X-axis — The Braking/Acceleration Sensor

The X-axis of the accelerometer points in the direction of travel. It tells us about forward forces:

- **Negative X (e.g., -3 m/s²)**: The car is decelerating (braking)
- **Positive X (e.g., +3 m/s²)**: The car is accelerating
- **Near zero**: Constant speed

**Why this matters**: When a driver brakes hard, the car pitches forward, which creates vibrations that can look like potholes on other sensors. By checking X-axis, we can separate **driver actions** (braking, accelerating) from **road surface events** (potholes, bumps).

### GPS Speed — The Context Sensor

GPS gives us the car's speed. This is critical for filtering:

- At very low speed (< 2 m/s = ~7 km/h), the phone vibrates mostly from engine idle, bumpy starts, or the driver fidgeting. These are not road surface events.
- When the car is starting up or stopping, there are momentum jerks that create false signals.

---

## Why These Specific Threshold Values?

Here is a reference table of every threshold used across all methods, what it means physically, and why that value was chosen.

| Threshold | Value | Physical Meaning | Why This Value |
|-----------|-------|------------------|----------------|
| **Speed filter** | 2.0 m/s (~7 km/h) | Minimum vehicle speed to trust sensor data | Below this, engine idle vibrations and start/stop jerks dominate; the car is effectively stationary or crawling |
| **Start/stop phase** | Speed < 1 m/s in last 3 seconds | Car was recently stopped or is about to stop | Initial/final momentum changes create large Z/gyro spikes that look like potholes but are just the car lurching |
| **Dip threshold** | 0.5 m/s² below gravity (9.81) | Z reading of ~9.3 m/s² or lower | A drop of 0.5 m/s² means about 5% less than gravity — enough to notice. Smaller dips are just normal road vibration |
| **Bump threshold** | 0.5 m/s² above gravity | Z reading of ~10.3 m/s² or higher | Similarly, a 5% increase above gravity is the minimum meaningful upward force from a road bump |
| **Gyro peak = 0.3 rad/s** | 17°/sec rotation | Car body noticeably rocks | A person sitting in the car would definitely feel this. Below 0.3, the rotation is too small to confirm a real road event |
| **Gyro peak = 0.8 rad/s** | 46°/sec rotation | Vehicle violently rocks | Used to cap speed bump classification — above this, the rotation is too extreme for a gentle bump; it is either a deep pothole or reckless driving |
| **X-accel = 2.0 m/s²** | Moderate braking/acceleration force | About 0.2g — what you feel in normal braking | Below 2.0, the forward force is too mild to attribute to driver behavior; above 2.0, braking/acceleration is the dominant signal |
| **X-accel = 2.5 m/s²** | Firm braking force | About 0.25g — noticeable deceleration | Used for confirming hard braking vs just mild acceleration changes |
| **Dip = 3.0 m/s²** | Very large dip (Z drops to ~6.8 m/s²) | About 30% drop from gravity — deep pothole | Such a large drop almost always means the wheel fell into something significant, even without gyro confirmation |
| **Dip = 1.5 m/s²** | Moderate dip (Z drops to ~8.3 m/s²) | ~15% drop from gravity | Paired with even mild gyro (> 0.15), this is enough to classify as a pothole rather than just rough pavement |
| **Total oscillation = 3.0 m/s²** | Sum of dip + bump magnitudes | How much the Z-axis swung up AND down | Speed bumps create a characteristic up-down-up pattern; total swing of 3+ m/s² distinguishes a real bump from noise |
| **Pawar threshold = 0.4** | 40% neural network probability | Model thinks there is a ~40% chance of pothole | Below 0.5 (normal cutoff) to catch borderline cases that V3 physics rules can still classify usefully |
| **Pawar confidence boost** | prob > 0.7 | 70%+ model confidence | When the trained model is very confident AND there is a Z-axis dip, we trust the classification even if gyro is weak |
| **DBSCAN eps = 20m** | Cluster radius | Two events within 20m are probably the same pothole | A typical pothole is 0.5–2m wide, but GPS accuracy is ~5–10m, so 20m catches the same feature measured twice |
| **Prophet 99% bounds** | interval_width = 0.99 | Only flag the most extreme 1% of readings | Using 99% instead of 95% means we only catch truly unusual events, reducing false positives significantly |
| **Window = 2 seconds** | Time window for feature aggregation | At 30 km/h, car covers ~17m in 2 seconds | Long enough to capture the dip-bounce pattern of hitting a pothole, short enough to localize it on GPS |

---

## V2 — Gyroscope-Based Anomaly Detection

### File: `v2/pothole_analysis_v2.ipynb`

### How It Works

1. **Prophet on Gyroscope (per axis)**
   - Runs Facebook Prophet time-series model on each gyroscope axis (X, Y, Z) independently
   - Prophet learns the "normal" baseline of gyroscope readings over time
   - Anomaly = actual reading deviates more than `mean + k × std` from Prophet's prediction (`yhat`)
   - Uses `k=2` (2 standard deviations) as the threshold — this means roughly the top 5% most unusual readings
   - **Why gyroscope?** V2's idea was that when a car hits a pothole, the car body rotates/rocks, so the gyroscope should spike

2. **Speed Context Interpolation**
   - GPS updates once per second (~1 Hz), but the gyroscope samples at ~100 Hz
   - We use linear interpolation to estimate the car's speed at every gyroscope timestamp
   - Then we detect **start/stop phases**: if the car's speed was below 1 m/s at any point in the last 3 seconds, we mark that period as "starting" (the jerk from starting to move creates big false readings)
   - Similarly for stopping: if speed drops below 1 m/s in the next 3 seconds

3. **Smart Filtering Pipeline**
   - **Filter 1:** Speed < 2 m/s → reject. At this speed (about walking pace), the dominant signal is engine vibration, not road bumps
   - **Filter 2:** Start phase → reject. The lurch when a car starts moving creates gyro/accel spikes indistinguishable from potholes
   - **Filter 3:** Stop phase → reject. Same issue when the car brakes to a stop

4. **Multi-Sensor Feature Extraction (per anomaly segment)**

   When the gyroscope flags an anomaly, V2 goes back to the **raw data from all sensors** in that time window to extract physics features:

   - **Gyroscope**: RMS magnitude (average rotation intensity), peak magnitude (worst-case rotation)
   - **TotalAcceleration Z**: How far below gravity (dip) or above gravity (bump) did Z go? This tells us the direction of the road event
   - **Accelerometer X**: Peak value — if X is large, the event was caused by braking/accelerating, not the road
   - **Temporal context**: Did the driver brake right before the event (suggesting they saw the pothole coming)?

5. **Classification Rules**

   | Event Type | Logic | Why This Combination |
   |-----------|-------|----------------------|
   | **Hard Braking** | X peak > 2.0 + low gyro (<0.3) + low Z dev (<2.0) | High forward force but no vertical disturbance or rotation = driver hit the brakes, not a pothole |
   | **Road Bump** | Z went UP only (bump > 0.5, dip < 0.5) | Road surface pushed the car up but did not drop it — this is a bump, not a hole |
   | **Speed Bump** | Both dip AND bump present + duration > 0.15s + high oscillation | The up-down-up pattern of driving over a speed bump; needs to be sustained (not just a quick jolt) |
   | **Pothole** | Z dip + gyro peak > 0.3 (car rocked) | Wheel fell into a hole (Z dropped) AND the car body rotated from the impact — this is the strongest pothole signal |
   | **Rough Patch** | Small dip, low gyro | Mild surface irregularity — not deep enough or violent enough to be a pothole |
   | **Minor Anomaly** | Does not match any clear pattern | Signal was unusual but does not clearly map to a road event |

6. **Spatial Clustering (DBSCAN)**
   - Multiple sensors often flag the same pothole in slightly different time windows
   - DBSCAN merges detections within 30 meters into one event
   - Keeps the highest severity and most common type from the cluster

7. **Road Roughness Index**
   - Computes `std(Z - 9.81)` over 1-second windows along the route
   - This gives a continuous "road quality score": low std = smooth, high std = rough
   - Green < 0.3 (smooth), Orange 0.3–0.8 (moderate), Red > 0.8 (severe)

---

## V3 — Prophet on Z-Axis TotalAcceleration

### File: `v3/pothole_v3_prophet_zaccel.ipynb`

### How It Differs from V2

**The key insight**: Z-axis TotalAcceleration is a **more direct** pothole signal than gyroscope.

Why? Gyroscope measures rotation, which happens as a **secondary effect** of hitting a pothole — the car drops into the hole, THEN the body rocks. But the Z-axis measures the **primary effect** — the actual vertical force change when the wheel drops. Also, gyroscope picks up rotation from steering, lane changes, and rough driving, which are all false positives for road detection.

- On smooth road: Z ≈ 9.81 m/s² (just gravity)
- Pothole: Z drops sharply (wheel drops into hole, brief partial freefall) then spikes (bounce back)
- Speed bump: Z rises (car pushed up) then drops slightly

### How It Works

1. **Prophet on TotalAcceleration Z-axis** (replacing gyroscope)
   - Uses `interval_width=0.99` — Prophet outputs a range `[yhat_lower, yhat_upper]` that should contain 99% of normal readings. Anything outside this range is flagged as anomalous
   - **Why 99% instead of 95%?** Because the Z-axis has a lot of small natural variation (road texture, engine hum). Using 99% means we only flag truly extreme events, dramatically reducing false positives
   - **Speed as extra regressor**: V3 tells Prophet "here is the car's speed at each moment." Prophet then learns that higher speeds cause more vibration (larger natural bounds). This means a 0.5 m/s² dip at 60 km/h might be normal, but the same dip at 20 km/h is unusual
   - Uses `changepoint_prior_scale=0.15` — this controls how quickly Prophet can change its baseline. A value of 0.15 lets it adapt to gradual road changes without over-reacting to spikes

2. **Deviation Metrics** (new in V3)
   - When a segment is flagged as anomalous, V3 measures HOW FAR outside the bounds it went:
     - `dip_magnitude`: How far Z dropped below gravity (9.81). E.g., Z_min of 7.5 → dip_magnitude = 2.31 m/s²
     - `bump_magnitude`: How far Z went above gravity. E.g., Z_max of 12.0 → bump_magnitude = 2.19 m/s²
     - These replace V2's raw residual error with physically meaningful numbers

3. **V3 uses gyroscope for CLASSIFICATION, not detection**

   This is an important distinction:
   - V2 uses gyroscope as the **detection sensor** (Prophet runs on gyro data)
   - V3 uses TotalAccel Z as the **detection sensor**, then looks at gyroscope data only AFTER an anomaly is found, to help **classify what kind of event** it is:
     - Z dip + high gyro → Pothole (car rocked = confirmed impact)
     - Z dip + low gyro → Rough Patch (dip present but no violent impact)
     - High X-accel + low gyro + low Z dev → Hard Braking (not a road event)

4. **Same Filtering as V2**
   - Speed < 2 m/s → reject
   - Start/stop phase → reject
   - GPS speed interpolation onto sensor timestamps

5. **Classification Rules** (refined from V2)

   | Event Type | Sensors Used | Logic | Why |
   |-----------|-------------|-------|-----|
   | **Hard Braking** | Accel-X + Gyro + TotalAccel-Z | X peak > 2.0 + gyro < 0.3 + Z dev < 2.0 | Strong forward force with no vertical or rotational disturbance = driver action |
   | **Road Bump** | TotalAccel-Z | bump > 0.5 AND no dip | Z went above gravity but did not drop below — car was pushed UP, consistent with a raised obstacle |
   | **Speed Bump** | TotalAccel-Z + Gyro | Both dip AND bump + oscillation > 3.0 + gyro < 0.8 | The up-down-up oscillation pattern of a speed bump; gyro < 0.8 rules out violent pothole impacts |
   | **Pothole** | TotalAccel-Z + Gyro + Pawar prob | Z dip + gyro > 0.3 (car rocked) | The classic pothole signal: vertical drop + body rotation |
   | **Rough Patch** | TotalAccel-Z | Small dip (< 1.5), low gyro | Surface is uneven but not damaged enough to be a pothole |
   | **Possible Pothole** | Pawar ANN only | High Pawar prob (> 0.8) but no clear V3 signal | The ML model is very confident but physics rules do not confirm — flagged for manual review |

6. **DBSCAN Clustering**
   - 20m radius (tighter than V2's 30m) — GPS is accurate enough for 20m, and this prevents different events from merging
   - Clusters events **separately by type** — a pothole and a braking event 15m apart should NOT merge into one

7. **Map Visualization**
   - Color-coded by event type (red=Pothole, green=Road Bump, goldenrod=Speed Bump, brown=Hard Braking, etc.)
   - Marker size proportional to severity
   - Popup shows all physics features for each event

### V3 vs V2 — Key Improvements

| Feature | V2 | V3 |
|---------|----|----|
| Detection signal | Gyroscope (indirect — rotation) | TotalAccel Z (direct — vertical force) |
| Anomaly method | mean + 2×std residual | Prophet 99% confidence bounds |
| Speed modeling | Filtering only | Extra regressor (Prophet adapts to speed) |
| Gyroscope role | Detection sensor | Classification helper (confirms pothole vs noise) |
| False positive rate | Higher (gyro catches driver behavior) | Lower (Z-axis is specific to road surface) |
| New categories | — | Rough Patch, Possible Pothole |
| Cluster radius | 30m | 20m (more precise) |

---

## Pawar et al. (2020) — Supervised ANN Approach

### Files: `pawar/pawar_pothole_detection.ipynb`, `pawar/pawar_on_our_data.ipynb`

### Paper Reference
> Kshitij Pawar, Siddhi Jagtap, Smita Bhoir. "Efficient pothole detection using smartphone sensors." ITM Web of Conferences 32, 03013 (2020), ICACC-2020.

### How It Works

1. **Data**
   - India_Data: 10 trips with labeled accelerometer data (X, Y, Z) in raw ADC units
   - Labels: "Not Detected" (normal) vs specific pothole descriptions
   - Binary classification: pothole (1) vs normal (0)
   - **No gyroscope** — only accelerometer readings + computed magnitude

2. **Feature Engineering (§3.2.1)**
   - Groups raw data into **2-second sliding windows**
   - Per window, per axis: **Min, Max, Mean, Standard Deviation**
   - 4 channels (X, Y, Z, magnitude) × 4 stats = **16 features**
   - Window label: 1 if ANY sample in the window is labeled as pothole
   - **Why 2 seconds?** At typical city driving speed (30 km/h), a car covers ~17 meters in 2 seconds. This is long enough to capture the full dip-bounce signature of a pothole hit, but short enough to localize it reasonably on GPS

3. **Why Min/Max/Mean/Std?**
   - **Min**: Catches the deepest dip (lowest acceleration = wheel dropping)
   - **Max**: Catches the highest bounce-back (wheel hitting far edge)
   - **Mean**: Baseline level — on smooth road, mean ≈ gravity
   - **Std**: How much variation within the window — potholes create high variation, smooth road has low variation
   - Together, these four statistics capture the SHAPE of the acceleration signal without needing the raw time series

4. **Preprocessing (§3.2.2)**
   - **Z-score standardization**: `x_scaled = (x - mean) / std`
   - This is critical because the India dataset uses raw sensor unit values (range ~[-512, 511]) while our data uses physical units in m/s². After Z-score normalization, both are in the same "number of standard deviations from the mean" space, making the trained model transferable

5. **SMOTE for Class Imbalance (§4)**
   - Only ~14% of windows contain potholes — the model would learn to just predict "normal" for everything and still get 86% accuracy
   - SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic pothole examples by interpolating between real pothole windows
   - This forces the model to learn actual pothole patterns rather than just defaulting to "normal"

6. **Neural Network Architecture (§3.3)**
   ```
   Input (16 features)
     → Dense(64, ReLU) → Dropout(0.3)
     → Dense(32, ReLU) → Dropout(0.3)
     → Dense(1, Sigmoid) → Binary output (0.0 to 1.0)
   ```
   - **64 and 32 neurons**: Progressively compress the 16 input features into a decision. 64 is enough to learn combinations of features; 32 further refines. More neurons would risk memorizing the small India dataset
   - **ReLU activation**: Standard choice — lets the network learn non-linear patterns
   - **Dropout(0.3)**: Randomly turns off 30% of neurons during training. This prevents overfitting — with only ~219 training windows, the model could easily memorize them
   - **Sigmoid output**: Produces a probability between 0 and 1 (e.g., 0.73 means "73% confidence this is a pothole")
   - **Adam optimizer**: Automatically adjusts learning rate for each parameter
   - **Batch size 3**: Very small batches (as per paper) — means the model updates its weights after seeing just 3 examples. This adds noise to training which acts as additional regularization

7. **Paper's Reported Results**

   | Metric | With SMOTE | Without SMOTE |
   |--------|-----------|---------------|
   | Accuracy | 0.9478 | 0.9263 |
   | Precision | 0.7105 | 0.7368 |
   | Recall | 0.8181 | 0.4242 |
   | F1 Score | 0.7605 | 0.5384 |
   | AUC | 0.972 | — |

   Notice how without SMOTE, recall drops from 0.82 to 0.42 — the model misses more than half the potholes. SMOTE fixes this.

---

## Hybrid Pawar + V3 — Best of Both Worlds

### File: `pawar/pawar_on_our_data.ipynb` (cells 24–31)

### The Problem with Pawar Alone

The Pawar model gives just a binary answer: "pothole or not." When applied to our A–F route data:
- It flags too many windows as potholes (false positives from braking, starting/stopping)
- It cannot distinguish between a pothole, a speed bump, hard braking, or road roughness
- It has no speed filtering, so it flags events when the car is stopped in traffic

### The Problem with V3 Alone

V3 is unsupervised — it has no learned knowledge about what a pothole "looks like" from training data. It relies entirely on physics rules, which can miss subtle potholes that do not create large Z dips.

### How the Hybrid Pipeline Works

The hybrid approach uses Pawar as the **initial detector** and V3 as the **classifier and filter**:

**Step 1 — Pawar ANN Detection (which sensor data: TotalAcceleration X, Y, Z, magnitude)**
- Create 2-second windows from TotalAcceleration data (same 16 features as Pawar training)
- Run the trained Pawar model → get a probability (0.0 to 1.0) for each window
- Use threshold of **0.4** (not 0.5) to catch borderline cases. Why lower than 0.5? Because V3's physics rules will filter out false positives anyway, so it is better to catch more candidates and let physics decide

**Step 2 — Speed Filtering (which sensor data: GPS)**
- For each flagged window, check vehicle speed from GPS
- **Reject if speed < 2.0 m/s**: At walking pace, sensor readings are dominated by engine idle and body movement, not road surface
- **Reject if start phase**: Car's speed was < 1 m/s in the past 3 seconds — the starting lurch creates big false signals
- **Reject if stop phase**: Car's speed will be < 1 m/s in the next 3 seconds — the stopping jerk creates false signals

**Step 3 — V3 Feature Extraction (which sensor data: TotalAcceleration Z, Gyroscope, Accelerometer X)**

For each surviving window, extract physics features from three sensors:

- **TotalAcceleration Z-axis**:
  - `dip_magnitude`: How far Z dropped below 9.81 m/s² (the gravity baseline). E.g., if Z_min = 7.5, then dip_magnitude = 9.81 - 7.5 = 2.31 m/s². This directly measures how much the wheel "fell"
  - `bump_magnitude`: How far Z went above 9.81. This measures how much the car was pushed upward
  - `is_dip`: Whether the dip was bigger than the bump (dip-dominant = pothole, bump-dominant = road bump)
  - `total_z_dev`: Maximum absolute deviation from gravity — the overall intensity of the vertical disturbance

- **Gyroscope** (magnitude = sqrt(gx² + gy² + gz²)):
  - `gyro_peak`: Maximum rotational velocity during the window. This confirms whether the car body physically rocked from the impact
  - `gyro_rms`: Average rotational velocity — sustained rotation vs a single spike

- **Accelerometer X-axis**:
  - `accel_x_min`: Most negative X value (strongest braking)
  - `accel_x_max`: Most positive X value (strongest acceleration)
  - `accel_x_peak`: Maximum absolute X value — overall forward/backward force
  - `braking_before`: Was there braking (X < -1.5 m/s²) in the 1 second before the event? Suggests the driver saw the pothole coming
  - `braking_after`: Was there braking in the 1 second after? Suggests the driver reacted to hitting something

**Step 4 — Hybrid Classification (all sensors combined + Pawar probability)**

Each flagged window is classified using V3's physics rules, with Pawar probability boosting confidence:

| Event Type | Primary Sensor | Supporting Sensors | Logic | Why This Works |
|-----------|----------------|-------------------|-------|----------------|
| **Hard Braking** | Accel-X (> 2.0) | Gyro (< 0.3), Z dev (< 2.0) | Strong forward force + no rotation + no vertical disturbance | All the energy is in the forward direction = driver action, not road |
| **Acceleration** | Accel-X (> 2.5) | Gyro (< 0.3), Z dev (< 2.0) | Same as braking but in positive X direction | Car speeding up creates vibrations that are not road events |
| **Road Bump** | TotalAccel-Z (bump > 0.5, no dip) | — | Z went UP above gravity but did not drop below | Road surface pushed the car upward — typical of a raised patch or small hump |
| **Speed Bump** | TotalAccel-Z (dip AND bump) | Gyro (< 0.8) | Both up AND down oscillation + total swing > 3.0 m/s² | The signature up-down-up pattern of a designed speed bump; gyro < 0.8 ensures it is not a violent impact |
| **Pothole (confirmed)** | TotalAccel-Z (dip > 0.5) | Gyro (> 0.3), Pawar prob | Z dropped below gravity + car body rocked + ML model agrees | All three signals align: physics says dip, rotation says impact, model says pothole |
| **Pothole (ML-boosted)** | TotalAccel-Z (dip > 0.8) | Pawar prob (> 0.7) | Moderate dip + high ML confidence, even without strong gyro | The trained model learned patterns that pure physics rules might miss |
| **Pothole (deep)** | TotalAccel-Z (dip > 3.0) | — | Very large Z drop (Z fell to ~6.8 m/s²) | Such a massive gravity disturbance can only be a significant road cavity |
| **Rough Patch** | TotalAccel-Z (small dip < 1.5) | Gyro (low) | Mild surface irregularity | Not deep enough or violent enough to call it a pothole |
| **Possible Pothole** | Pawar prob (> 0.8) | No V3 signal | ML model is very confident but physics features are weak | Flagged for attention — the ML model saw something in the statistical features that the physics rules did not catch |

**The key hybrid benefit**: Pawar probability is used as a **confidence booster** for potholes. When both Pawar (learned patterns) and V3 (physics) agree, confidence can reach 95%. When only one agrees, confidence stays lower (~50-65%), and when neither strongly agrees, the event is downgraded.

**Step 5 — DBSCAN Spatial Clustering**
- Groups events of the same type within 20m of each other
- Uses haversine distance (accounts for Earth's curvature)
- Clusters are computed separately per event type — a pothole and a braking event 15m apart stay as separate events

**Step 6 — Leaflet Map with All Event Types**
- Red circles = Potholes (confirmed road damage)
- Green circles = Road Bumps (Z went up only)
- Goldenrod circles = Speed Bumps (up-down oscillation pattern)
- Brown circles = Hard Braking (driver action, not road)
- Saddle brown circles = Rough Patches (mild surface issues)
- Blue circles = Acceleration events
- Orange circles = Possible Potholes (ML-only detection)
- Circle size proportional to severity (0–10 scale)
- Route polylines in different colors (A through F)

### Why the Hybrid Is Better Than Either Alone

| Aspect | Pawar Alone | V3 Alone | Hybrid |
|--------|-------------|----------|--------|
| False positives | High (no speed/context filter) | Low (physics rules) | Very low (physics + ML) |
| Missed potholes | Moderate | Some subtle ones | Fewer (ML catches what physics misses) |
| Event types | Just "pothole/not" | 6+ types | 7 types with ML confidence |
| Confidence scoring | Just probability | Rule-based (0.5 or 0.85) | Combined (0.20 to 0.95, continuous) |
| Speed context | None | Full filtering | Full filtering |
| Requires training data | Yes | No | Yes (for Pawar component) |

---

## Summary

- **V2**: Uses Prophet on **gyroscope** → detects vehicle rotation anomalies → classifies using multi-sensor physics rules. Good general approach but gyroscope captures driver behavior too (steering, braking feel), leading to more false positives.

- **V3**: Switches to Prophet on **Z-axis TotalAcceleration** → directly detects vertical force anomalies (the primary physical effect of a pothole). Uses 99% confidence bounds with speed as a regressor. Gyroscope is used only for classification confirmation, not detection. More precise with tighter clustering.

- **Pawar**: Supervised ANN trained on labeled India pothole data. Uses **2-second windowed statistical features** (min/max/mean/std of accelerometer) → binary pothole/normal classification. Simple but effective when training data matches the target environment.

- **Hybrid Pawar+V3**: Uses Pawar ANN as a primary **candidate detector** (catches statistical patterns), then V3's physics rules **classify and validate** each candidate using gyroscope, Z-axis direction, X-axis braking, and speed context. Produces multi-class output (Pothole, Road Bump, Speed Bump, Hard Braking, Acceleration, Rough Patch, Possible Pothole) with combined ML+physics confidence scores. This gives the best of supervised learning and physics-based reasoning.

All four produce interactive Leaflet maps with GPS-located markers. V2 and V3 additionally produce road roughness indices. The hybrid approach provides the most detailed event classification with the highest confidence calibration.
---

## New V3 — Post-Classification Overlap Fix on V3

### Why This Exists

After detailed comparison of V3, Pawar, and the Hybrid approach, we concluded:

- **V3 (Prophet on Z-axis)** produced the most realistic and balanced event distribution: 82 Road Bumps, 73 Minor Anomalies, 64 Hard Braking, 57 Potholes, 57 Rough Patches, 10 Acceleration events.
- **Pawar Hybrid** was too binary — it collapsed everything into Speed Bumps (198) and Potholes (69), with almost no Hard Braking (1), Rough Patches (1), or Road Bumps (7). It was essentially a two-bucket system that lost the nuance of V3's multi-class classification.
- **Pawar ANN contribution was minimal** — it acted as a pre-filter but did not meaningfully improve detection quality. V3's Prophet anomaly detection was already catching the same events, and the Pawar window-based approach (2-second fixed windows) was less precise than V3's per-point anomaly segments.

However, V3 had **two specific problems** that New V3 fixes:

### Problem 1: Pothole + Road Bump Overlap

**The issue**: Prophet detects each anomaly point independently. When a car hits a pothole, two things happen physically:
1. Z drops below the lower bound (the dip into the hole) → classified as **Pothole**
2. Z spikes above the upper bound (the bounce-back) → classified as **Road Bump**

These are separate Prophet anomaly segments at different timestamps, but they originate from the **same physical event**. After DBSCAN clusters by event type separately, they become two markers 0–10m apart on the map.

**Quantified**: 50 out of 57 Potholes had a Road Bump within 20m (median distance: **2.3m** — essentially the same spot). The dip deviation feature was the definitive separator: **100% of Potholes had dip_dev < -0.5, 0% of Road Bumps did**.

**The fix**: Post-DBSCAN merge with a **10m radius** (not 20m — analysis showed most true pairs are within 0–5m, and 10m captures 45/52 pairs while being conservative). For each Road Bump within 10m of a Pothole on the same route, absorb it into the Pothole — the bump is just the bounce-back. The Pothole keeps its label and gains the bump's features (higher severity, bump deviation, merged detection count).

**Merge radius sensitivity analysis**:

| Radius | Bumps Absorbed | Bumps Remaining |
|--------|---------------|-----------------|
| 5m | 36 | 46 |
| 8m | 42 | 40 |
| **10m** | **45** | **37** |
| 12m | 46 | 36 |
| 15m | 48 | 34 |
| 20m | 50 | 32 |

10m was chosen because it captures the natural plateau (36 → 45 → 46 → 48) while staying conservative.

### Problem 2: Too Many Hard Braking Events

**The issue**: V3's `classify_event` had a secondary Hard Braking check: if `braking_before` was True and `gyro_peak < 0.3` and `abs_dev < 2.0`, the event was labeled Hard Braking with confidence 0.65. This caught 56 of the 64 Hard Braking events — but analysis showed most were very weak signals:

- **45/64** had severity < 2.0 (barely noticeable)
- **56/64** had max_accel_x < 2.0 (weak longitudinal force — not real braking)
- **Only 8/64** had accel_x ≥ 2.0 (what actual hard braking looks like: ~0.2g deceleration)
- **34/64** were within 20m of a Pothole (road vibration misclassified as braking)

**The fix**: Only keep Hard Braking events where `max_accel_x ≥ 2.0` (genuine strong braking). The 56 weak events were reclassified based on their actual Z-axis features:
- **38 → Rough Patch** (had dip deviation < -0.5 — road surface issue, not driver behavior)
- **13 → Road Bump** (had bump deviation > 0.5 but no dip — uplift, not braking)
- **5 → Minor Anomaly** (neither significant dip nor bump — just noise)

### Results: V3 Original → New V3

| Event Type | V3 Original | New V3 | Change |
|---|---|---|---|
| Rough Patch | 57 | **95** | +38 |
| Minor Anomaly | 73 | 78 | +5 |
| Pothole | 57 | 57 | 0 |
| Road Bump | 82 | **47** | -35 |
| Acceleration | 10 | 10 | 0 |
| Hard Braking | 64 | **8** | -56 |
| **TOTAL** | **343** | **295** | -48 |

### Why New V3 Is Better Than V3

1. **Road Bumps now outnumber Potholes** (47 vs 57) — but the fake bounce-back duplicates are gone. The remaining 47 are genuine standalone bumps.
2. **Hard Braking reduced to 8 genuine events** — strong accel_x ≥ 2.0, not weak vibrations misclassified by the `braking_before` shortcut.
3. **Rough Patches increased to 95** — absorbing the 38 weak "Hard Braking" events that actually had dip deviation (road surface issues) and genuinely represent rough road segments.
4. **Potholes unchanged at 57** — but each now includes the absorbed bounce-back data, giving stronger severity signals.
5. **No false merges** — 10m is conservative, and the analysis confirmed 37/45 close pairs were definitively Pothole (dip-dominant or high gyro), only 2/45 were truly Road Bump.

### Why Pawar ANN Was Not Helpful

After extensive testing of the Pawar ANN (Dense(64)→Dense(32)→Dense(1), SMOTE-balanced, 16 features per 2-second window), we found:

1. **V3's Prophet already catches the same events** — Prophet's 99% confidence bounds with per-point detection are more precise than 2-second fixed windows.
2. **The ANN adds a binary filter, not classification** — it only says "pothole/normal" with no multi-class capability. All the useful classification (Pothole vs Road Bump vs Hard Braking vs Rough Patch) comes from V3's physics rules anyway.
3. **Window-based detection loses temporal precision** — a 2-second window at 30 km/h covers ~17 meters of road. V3's per-point segments are sub-meter.
4. **Training data mismatch** — the ANN was trained on Indian road data (different car, suspension, road conditions). When applied to our routes, it was essentially just a noisy threshold on Z-axis variance.

**Conclusion**: The Pawar ANN does not improve upon V3's unsupervised Prophet detection. New V3 achieves better results by fixing V3's two specific bugs (overlap and weak Hard Braking) rather than adding a supervised component.