<h1 align="center">Demoiselle Crane Migration Analysis</h1>

This project analyzes the migration of **Demoiselle Cranes (*Anthropoides virgo*)** using irregularly sampled GPS tracking data. 

## Objectives

1. **Detect stopovers:** identify key stopover sites and events, and generate an interactive stopover atlas.
2. **Estimate phenology:** infer seasonal departure and arrival dates, and produce individual- and population-level summaries for each migratory phase.
3. **Map corridors:** extract migration corridors and generate an interactive map of **core (50% utilization distribution, UD)** and **broad (95% UD)** corridors.

## Pipeline

### Stage 1: Read Data
This stage loads the raw GPS tracking data, cleans it and applies the following filtering:
1. **Filters out low-coverage individuals**: removes individuals with fewer than `min_count` total GPS tracking points. For this study, only those individuals who had more than 1000 GPS tracking points were selected.
2. **Keeps only relevant wintering individuals**: filters individual who winter in the intended study area as specified by `max_latitude`. For this study, only the African wintering individuals were selected.

**Output**: A dataset of irregularly sampled tracks of the selected individuals.

### Stage 2: Random Walk (RW) Model
This stage converts the irregularly sampled GPS tracks into smooth, regularly sampled movement trajectories using a Random Walk (RW) model.

It performs the following functions:
1. **Segments tracks at large temporal gaps**: each individual's track is split into segments whenever the gap between observations exceeds `gap_split_days`. This prevents the model from interpolating across biologically unrealistic breaks.
2. **Applies a Random Walk model per segment**: each segment is modeled independently and resampled to a fixed interval (defined by `time_step_hours`), producing smooth and evenly spaced locations.
3. **Removes very short segments**: segments with fewer than `min_points` are discarded, as they do not provide enough information for reliable modeling.

A Continuous-Time Correlated Random Walk (CTCRW) model was not used here because:
1. Many individuals made sharp turns during migration. Because CTCRW enforces persistence in direction and speed, the fitted trajectories overshot at these turns and then curved back toward the observed locations, producing loops and detours that were inconsistent with the recorded tracks.
2. Many individuals made sharp turns during migration. Because CTCRW enforces persistence in direction and speed, the fitted trajectories overshot at these turns and then curved back toward the observed locations, producing loops and detours that were inconsistent with the recorded tracks.

Given these outcomes, a simpler Random Walk model, combined with  gap-based segmentation, produced more reliable, regularly sampled tracks for downstream analyses.

**Output**: A dataset of regularly sampled track segments of the selected individuals.

### Stage 3: Stopover Detection
This stage uses the regularly sampled track segments to identify key stopover sites and stopover events.

It uses the following process:
1. **Scans each track segment to find stopover events**: within each track segment, continuous periods where successive locations remain within `r_stop_km` are identified. If that period lasts at least `min_stop_hours`, it is recorded as a stopover event with start/end time, duration, centroid location, and number of points.
2. **Assigns events to stopover sites using spatial clustering**: all detected stopover event centroids are clustered using DBSCAN with a clustering eps of `site_eps_km` and a minimum support of `min_samples`. This groups nearby events into shared stopover sites across individuals and segments.
3. **Handles isolated events as separate sites**: events that do not belong to any DBSCAN cluster are treated as their own single-event sites and marked as noise-derived sites in the site summary.

**Output**: An interactive stopover atlas and summaries of stopover sites and stopover events.

### Stage 4: Phenology Analysis
This stage estimates seasonal departure and arrival timing for each individual in a migration season.

The following process is followed:
1. **Defines wintering and breeding "regions" per individual-year**: for each individual and season-year, observed locations in the configured winter months and breeding months are used to estimate typical winter and summer locations. This step requires at least `min_points_winter` and `min_points_breeding` points, respectively.
2. **Builds season-specific boundaries from stopover structure**: for each season-year, the stopover events associated with the individual (and nearby connected events within `link_radius_km`) are used to create a spatial boundary for the wintering and breeding ranges. A margin of `hull_margin_km` is added so the boundary is not overly tight.
3. **Detects migration timing from boundary entry/exit**:
    - Spring migration: estimates winter departure as the first confirmed exit from the winter boundary, and summer arrival as the first confirmed entry into the summer boundary.
    - Fall migration: estimates summer departure as the first confirmed exit from the summer boundary, and winter arrival as the first confirmed entry into the winter boundary.
  To avoid false triggers from brief excursions or missing points, departures are only accepted once the individual stays outside the boundary for `departure_confirm_days`.

**Output**: Individual and population level migration summaries, overall migration summary, phenology plots (timings and duration distributions)

### Stage 5: Migration Corridor and Utilization Distribution Mapping using dynamic Brownian Bridge Movement Model (dBMM)
This stage maps migration corridors by estimating a utilization distribution (UD) from the migration tracks using a dynamic Brownian Bridge Movement Model (dBBMM), and then extracting core (50%) and broad (95%) corridor polygons.

This stage performs the following steps:
1. **Loads migration-only GPS fixes**: the irregularly sampled data that containing only migration-period locations is read.
2. **Segments tracks at large temporal gaps**: for each individual, the track is split into segments whenever the gap between fixes exceeds `gap_split_days`. Segments with fewer than `min_points` fixes are discarded. This avoids fitting a corridor model across unrealistic gaps.
3. **Projects data and builds an analysis grid**: all points are projected to a planar coordinate system (`proj_crs`). A raster grid is created over the full movement extent. This grid is the base for estimating the UD.
4. **Fits dBBMM per segment**: each kept segment is fit with a dynamic Brownian bridge using the configured parameters (`location_error_m`, `window_size`, `margin`, `time_step_mins`). This produces a UD raster for that segment. Segments that fail to fit or produce unusable rasters are skipped.
5. **Builds individual and global UDs**:
      - Segment UDs are summed to form an individual-level UD, then normalized.
      - Individual UDs are summed to form a global population-level UD, then normalized.
6. **Extracts 50% and 95% corridors**: the global UD is converted into a cumulative "volume" surface. The pipeline then thresholds this surface to extract:
      - 50% UD polygon (core corridor)
      - 95% UD polygon (broad corridor)
  
**Output**: Interactive corridor map, 95% and 50% corridor polygons as geojson.

## Environment Setup

### Python Environment
Conda Installation
```
conda create env -f environment.yml
conda activate cranes
```
PyPI Installation
```
pip install -r requirements.txt
```
### R Environment
```
Rscript setup.R
```

## How to run
```
python src/main.py
```
