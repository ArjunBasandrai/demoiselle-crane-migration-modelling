from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Callable, Any

import matplotlib
# matplotlib.use("Agg")

from shapely.geometry import MultiPoint, Point
from shapely.ops import transform
from shapely.geometry.base import BaseGeometry
import pyproj

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

XYTransform = Callable[[Any, Any], Tuple[Any, Any]]

EARTH_RADIUS_KM = 6371.0088

def _to_utc(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True, errors="coerce")

def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

def _dist_to_point_km(lat: pd.Series, lon: pd.Series, clat: float, clon: float) -> np.ndarray:
    return _haversine_km(lat.to_numpy(float), lon.to_numpy(float), clat, clon)

def _calculate_overlap_hours(a_start, a_end, b_start, b_end) -> float:
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    if pd.isna(s) or pd.isna(e) or e <= s:
        return 0.0
    return (e - s).total_seconds() / 3600.0

def _expand_year_season_events_connected_to_bird(
    year_season_events: pd.DataFrame,
    pid: int,
    link_radius_km: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if link_radius_km <= 0:
        raise ValueError("link_radius_km must be > 0")

    df = year_season_events.copy()

    n = len(df)
    if n == 0:
        return df, df

    seed_idx = np.flatnonzero(df["id"] == pid)
    if seed_idx.size == 0:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy()

    coords_rad = np.radians(df[["centroid_lat", "centroid_lon"]].to_numpy(dtype=float))
    tree = BallTree(coords_rad, metric="haversine")
    r_rad = float(link_radius_km) / EARTH_RADIUS_KM

    visited = np.zeros(n, dtype=bool)
    visited[seed_idx] = True

    queue = seed_idx.tolist()
    while queue:
        i = queue.pop()
        neigh = tree.query_radius(coords_rad[i : i + 1], r=r_rad, return_distance=False)[0]
        if neigh.size == 0:
            continue
        new = neigh[~visited[neigh]]
        if new.size:
            visited[new] = True
            queue.extend(new.tolist())

    return df.loc[seed_idx].reset_index(drop=True), df.loc[visited].reset_index(drop=True)

def _create_hull(
    lats: np.ndarray | pd.Series,
    lons: np.ndarray | pd.Series,
    margin_km: float = 0.0
) -> Tuple[XYTransform, BaseGeometry, pyproj.CRS]:
    _EPS = 1e-10

    def _dist(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    def _is_in_circle(c, p):
        cx, cy, r = c
        dx = p[0] - cx
        dy = p[1] - cy
        return (dx * dx + dy * dy) <= (r * r + _EPS)

    def _circle_two_points(a, b):
        cx = (a[0] + b[0]) / 2.0
        cy = (a[1] + b[1]) / 2.0
        r = _dist(a, b) / 2.0
        return (cx, cy, r)

    def _circle_three_points(a, b, c):
        ax, ay = a
        bx, by = b
        cx, cy = c
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < _EPS:
            return None
        a2 = ax * ax + ay * ay
        b2 = bx * bx + by * by
        c2 = cx * cx + cy * cy
        ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
        uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
        r = ((ux - ax) ** 2 + (uy - ay) ** 2) ** 0.5
        return (ux, uy, r)

    def _min_enclosing_circle(points):
        pts = points.copy()
        rng = np.random.default_rng(0)
        rng.shuffle(pts)

        c = None
        for i, p in enumerate(pts):
            if c is not None and _is_in_circle(c, p):
                continue
            c = (p[0], p[1], 0.0)
            for j in range(i):
                q = pts[j]
                if _is_in_circle(c, q):
                    continue
                c = _circle_two_points(p, q)
                for k in range(j):
                    r = pts[k]
                    if _is_in_circle(c, r):
                        continue
                    cc = _circle_three_points(p, q, r)
                    if cc is None:
                        dists = [(_dist(p, q), (p, q)), (_dist(p, r), (p, r)), (_dist(q, r), (q, r))]
                        _, (u, v) = max(dists, key=lambda x: x[0])
                        c = _circle_two_points(u, v)
                    else:
                        c = cc
        return c  # (cx, cy, radius)

    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)

    lat0 = float(lats.mean())
    lon0 = float(lons.mean())

    crs_src = pyproj.CRS("EPSG:4326")
    crs_dst = pyproj.CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    transformer = pyproj.Transformer.from_crs(crs_src, crs_dst, always_xy=True)
    to_xy = transformer.transform

    xs, ys = to_xy(lons, lats)
    pts_xy = np.column_stack([xs, ys])

    if len(pts_xy) == 0:
        circle_xy = Point(0, 0).buffer(0)
        return to_xy, circle_xy, crs_dst

    cx, cy, r = _min_enclosing_circle(pts_xy)
    r = r + (margin_km * 1000.0 if margin_km > 0 else 0.0)
    circle_xy = Point(cx, cy).buffer(r)

    return to_xy, circle_xy, crs_dst

def _inside_hull(
    to_xy: XYTransform,
    hull: BaseGeometry,
    lat: float,
    lon: float
) -> bool:
    p_xy = transform(to_xy, Point(lon, lat))
    return hull.covers(p_xy)

def read_rw_track(rw_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(rw_csv_path)
    needed = {"id", "seg_id", "date", "lon", "lat"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"rw_csv_path missing columns: {sorted(missing)}")

    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["seg_id"] = df["seg_id"].astype(str)
    df["date"] = _to_utc(df["date"])
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

    df = df.dropna(subset=["id", "seg_id", "date", "lon", "lat"]).sort_values(["id", "date"])
    df = df.drop_duplicates(subset=["id", "date"], keep="first")
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    return df.reset_index(drop=True)

def read_stopover_events(events_csv_path: str) -> pd.DataFrame:
    ev = pd.read_csv(events_csv_path)
    needed = {"id", "start_time", "end_time", "duration_hours"}
    missing = needed - set(ev.columns)
    if missing:
        raise ValueError(f"events_csv_path missing columns: {sorted(missing)}")

    ev = ev.copy()
    ev["id"] = ev["id"].astype(str)
    ev["start_time"] = _to_utc(ev["start_time"])
    ev["end_time"] = _to_utc(ev["end_time"])
    ev["duration_hours"] = pd.to_numeric(ev["duration_hours"], errors="coerce")
    if "site_id" in ev.columns:
        ev["site_id"] = pd.to_numeric(ev["site_id"], errors="coerce")

    ev = ev.dropna(subset=["id", "start_time", "end_time", "duration_hours"])
    ev = ev[ev["end_time"] > ev["start_time"]].sort_values(["id", "start_time"]).reset_index(drop=True)
    return ev

def compute_seasonal_centroids(
    track: pd.DataFrame,
    *,
    winter_months: Tuple[int, ...],
    breeding_months: Tuple[int, ...],
    min_points_winter: int,
    min_points_breeding: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t = track.copy()

    winter = t[t["month"].isin(winter_months)].copy()
    winter["season_year"] = np.where(winter["month"] == 12, winter["year"] + 1, winter["year"]).astype(int)
    winter_centroids = (
        winter.groupby(["id", "season_year"], as_index=False)
        .agg(
            winter_lon=("lon", "mean"),
            winter_lat=("lat", "mean"),
            n_winter=("lon", "size"),
        )
        .query("n_winter >= @min_points_winter")
        .reset_index(drop=True)
    )

    breed = t[t["month"].isin(breeding_months)].copy()
    breed["season_year"] = breed["year"].astype(int)
    breeding_centroids = (
        breed.groupby(["id", "season_year"], as_index=False)
        .agg(
            breed_lon=("lon", "mean"),
            breed_lat=("lat", "mean"),
            n_breeding=("lon", "size"),
        )
        .query("n_breeding >= @min_points_breeding")
        .reset_index(drop=True)
    )

    return winter_centroids, breeding_centroids

def _ts(y: int, md: Tuple[int, int]) -> pd.Timestamp:
    m, d = md
    return pd.Timestamp(year=y, month=m, day=d, tz="UTC")

def _last_within(
    df: pd.DataFrame,
    clat: float,
    clon: float,
    radius_km: float,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    w = df[(df["date"] >= t_start) & (df["date"] <= t_end)]
    if w.empty:
        return None
    dist = _dist_to_point_km(w["lat"], w["lon"], clat, clon)
    w2 = w.loc[dist <= radius_km]
    if w2.empty:
        return None
    return pd.Timestamp(w2["date"].iloc[-1])

def _first_within(
    df: pd.DataFrame,
    clat: float,
    clon: float,
    radius_km: float,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    w = df[(df["date"] >= t_start) & (df["date"] <= t_end)]
    if w.empty:
        return None
    dist = _dist_to_point_km(w["lat"], w["lon"], clat, clon)
    w2 = w.loc[dist <= radius_km]
    if w2.empty:
        return None
    return pd.Timestamp(w2["date"].iloc[0])

def _migration_step_stats(points: pd.DataFrame, *, gap_split_days: float) -> dict:
    if len(points) < 2:
        return dict(
            n_points=int(len(points)),
            route_distance_km=np.nan,
            net_displacement_km=np.nan,
            straightness=np.nan,
            max_gap_hours=np.nan,
            n_big_gaps=0,
            median_step_speed_kmph=np.nan,
            mean_step_speed_kmph=np.nan,
        )

    lat = points["lat"].to_numpy(float)
    lon = points["lon"].to_numpy(float)
    t = points["date"].to_numpy(dtype="datetime64[ns]")

    dt_h = (t[1:] - t[:-1]) / np.timedelta64(1, "h")
    dt_h = np.asarray(dt_h, dtype=float)

    step_km = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    step_km = np.asarray(step_km, dtype=float)

    big_gap = dt_h > (gap_split_days * 24.0)
    max_gap = float(np.nanmax(dt_h)) if dt_h.size else np.nan
    n_big = int(np.sum(big_gap))

    use = (~big_gap) & np.isfinite(dt_h) & (dt_h > 0) & np.isfinite(step_km)
    dist_km = float(np.sum(step_km[use])) if np.any(use) else np.nan

    net_km = float(_haversine_km(lat[0], lon[0], lat[-1], lon[-1]))
    straight = float(net_km / dist_km) if (np.isfinite(net_km) and np.isfinite(dist_km) and dist_km > 0 and net_km > 0) else np.nan

    speed = np.full_like(step_km, np.nan, dtype=float)
    ok = np.isfinite(step_km) & np.isfinite(dt_h) & (dt_h > 0)
    speed[ok] = step_km[ok] / dt_h[ok]

    return dict(
        n_points=int(len(points)),
        route_distance_km=dist_km,
        net_displacement_km=net_km,
        straightness=straight,
        max_gap_hours=max_gap,
        n_big_gaps=n_big,
        median_step_speed_kmph=float(np.nanmedian(speed)) if np.any(np.isfinite(speed)) else np.nan,
        mean_step_speed_kmph=float(np.nanmean(speed)) if np.any(np.isfinite(speed)) else np.nan,
    )

def first_confirmed_entry_day(
    track_df: pd.DataFrame,
    cutoff_timestamp: pd.Timestamp,
    is_inside: pd.Series
) -> None | int:
    df = track_df.copy()
    is_inside = pd.Series(is_inside, index=df.index).astype(bool)

    # Cutoff Trimming
    cutoff_mask = df['date'] >= cutoff_timestamp
    df, timestamps, is_inside = df.loc[cutoff_mask], df['date'].loc[cutoff_mask], is_inside.loc[cutoff_mask]
    if df.empty:
        return None
        
    # Group by day
    day = timestamps.dt.floor('D')
    day_inside = is_inside.groupby(day).any()
    if not day_inside.any():
        return None
    
    # Entry detection
    entry_day = day_inside.index[day_inside.values.argmax()]
    return entry_day
    
def first_confirmed_exit_day(
    track_df: pd.DataFrame,
    cutoff_timestamp: pd.Timestamp,
    departure_confirm_days: int, 
    is_inside: pd.Series
) -> None | int:
    df = track_df.copy()
    is_inside = pd.Series(is_inside, index=df.index).astype(bool)

    # Cutoff Trimming
    cutoff_mask = df['date'] >= cutoff_timestamp
    df, timestamps, is_inside = df.loc[cutoff_mask], df['date'].loc[cutoff_mask], is_inside.loc[cutoff_mask]
    if df.empty:
        return None
        
    # Group by day
    day = timestamps.dt.floor('D')
    day_inside = is_inside.groupby(day).any()
    if not day_inside.any():
        return None
    
    # Entry detection
    entry_day = day_inside.index[day_inside.values.argmax()]
    post_entry_days = day_inside.loc[entry_day:].index
    day_inside = day_inside.loc[post_entry_days]
    day_outside = ~day_inside

    # Exit detection
    run = 0
    exit_day = None
    for d, out in day_outside.items():
        if out:
            run += 1
            if run == 1:
                exit_day = d
            if run >= departure_confirm_days:
                break
        else:
            run = 0
            exit_day = None
    
    if run < departure_confirm_days:
        return None

    return exit_day

def plot_hull_xy(hull_xy, crs_dst, ax=None):
    from_xy = pyproj.Transformer.from_crs(crs_dst, "EPSG:4326", always_xy=True).transform
    hull_ll = transform(from_xy, hull_xy)
    if ax is None:
        _, ax = plt.subplots()

    geoms = getattr(hull_ll, "geoms", [hull_ll])

    for g in geoms:
        if g.geom_type != "Polygon":
            continue

        c = np.asarray(g.exterior.coords)  # (lon, lat)
        ax.fill(c[:, 0], c[:, 1], color="orange", alpha=0.5, edgecolor="orange")

        for hole in g.interiors:
            h = np.asarray(hole.coords)
            ax.fill(h[:, 0], h[:, 1], color="white", alpha=1.0, edgecolor="white")

    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.grid(True)
    return ax

def _plot_bird_hull_observations(
    hull_xy_summer: BaseGeometry,
    hull_xy_winter: BaseGeometry,
    crs_dst_summer: XYTransform,
    crs_dst_winter: XYTransform,
    bird_year_summer_events: pd.DataFrame,
    bird_year_winter_events: pd.DataFrame,
    bird_year_observations: pd.DataFrame,
    bird_id: str,
    year: int
):
    fig_w, fig_h = 12, 12
    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)

    plot_hull_xy(hull_xy_summer, crs_dst_summer, ax)
    plot_hull_xy(hull_xy_winter, crs_dst_winter, ax)

    sns.scatterplot(
        data=bird_year_observations,
        x='lon',
        y='lat',
        s=8,
        ax=ax,
        transform=ccrs.PlateCarree()
    )
    sns.scatterplot(
        data=bird_year_winter_events,
        x='centroid_lon',
        y='centroid_lat',
        s=48,
        ax=ax,
        transform=ccrs.PlateCarree()
    )
    sns.scatterplot(
        data=bird_year_summer_events,
        x='centroid_lon',
        y='centroid_lat',
        s=48,
        ax=ax,
        transform=ccrs.PlateCarree()
    )
    xmin, xmax = bird_year_observations["lon"].min(), bird_year_observations["lon"].max()
    ymin, ymax = bird_year_observations["lat"].min(), bird_year_observations["lat"].max()
    lat_margin, lon_margin = 5, 5

    lat_span = (ymax - ymin) + 2 * lat_margin
    lon_span = (xmax - xmin) + 2 * lon_margin
    target_ratio = fig_w / fig_h
    if lon_span < lat_span * target_ratio:
        extra = (lat_span * target_ratio - lon_span) / 2
        lon_margin += extra

    ax.set_extent(
        [xmin - lon_margin, xmax + lon_margin, ymin - lat_margin, ymax + lat_margin],
        crs=ccrs.PlateCarree(),
    )

    ax.set_title(f"Hull and Observations for Bird {bird_id} in the year {year}")

    plt.show()

def compute_phenology(
    track: pd.DataFrame,
    events: pd.DataFrame,
    *,
    hull_margin_km: float,
    winter_months: Tuple[int, ...],
    breeding_months: Tuple[int, ...],
    min_points_winter: int,
    min_points_breeding: int,
    major_stopover_hours: float,
    gap_split_days: float,
    winter_start_month_day: Tuple[int, int],
    winter_end_month_day: Tuple[int, int],
    summer_start_month_day: Tuple[int, int],
    summer_end_month_day: Tuple[int, int],
    departure_confirm_days: int,
    link_radius_km: int,
) -> pd.DataFrame:
    winter_c, breed_c = compute_seasonal_centroids(
        track,
        winter_months=winter_months,
        breeding_months=breeding_months,
        min_points_winter=min_points_winter,
        min_points_breeding=min_points_breeding,
    )

    out_rows = []

    for pid, df_id in track.groupby("id", sort=False):
        df_id = df_id.sort_values("date").reset_index(drop=True)
        years = sorted(
            set(winter_c.loc[winter_c["id"] == pid, "season_year"].tolist())
            | set(breed_c.loc[breed_c["id"] == pid, "season_year"].tolist())
        )
        if not years:
            continue

        ev_id = events[events["id"] == pid].copy()

        for y in years:
            bird_year_summer_events, bird_year_summer_events_expanded = _expand_year_season_events_connected_to_bird(
                events.loc[
                    (
                        (events['start_time'].dt.year == y) | 
                        (events['end_time'].dt.year == y)
                    ) &
                    (events['start_time'].between(_ts(y, summer_start_month_day), _ts(y, summer_end_month_day))) &
                    (events['end_time'].between(_ts(y, summer_start_month_day), _ts(y, summer_end_month_day)))
                ].reset_index(drop=True),
                pid,
                link_radius_km
            )
            bird_year_winter_events, bird_year_winter_events_expanded = _expand_year_season_events_connected_to_bird(
                events.loc[
                    (events['id'] == pid) &
                    (
                        (events['start_time'].dt.year.between(y - 1, y)) | 
                        (events['end_time'].dt.year.between(y - 1, y))
                    ) &
                    (events['start_time'].between(_ts(y - 1, winter_start_month_day), _ts(y, winter_end_month_day))) &
                    (events['end_time'].between(_ts(y - 1, winter_start_month_day), _ts(y, winter_end_month_day)))
                ].reset_index(drop=True),
                pid,
                link_radius_km
            )
            bird_next_year_winter_events, bird_next_year_winter_events_expanded  = _expand_year_season_events_connected_to_bird(
                events.loc[
                    (events['id'] == pid) &
                    (
                        (events['start_time'].dt.year.between(y, y + 1)) | 
                        (events['end_time'].dt.year.between(y, y + 1))
                    ) &
                    (events['start_time'].between(_ts(y, winter_start_month_day), _ts(y + 1, winter_end_month_day))) &
                    (events['end_time'].between(_ts(y, winter_start_month_day), _ts(y + 1, winter_end_month_day)))
                ].reset_index(drop=True),
                pid,
                link_radius_km
            )

            # Spring Migration
            if len(bird_year_summer_events_expanded) > 0 and len(bird_year_winter_events_expanded) > 0:
                bird_year_observations = df_id[df_id['date'].dt.year == y]
                to_xy_summer, hull_xy_summer, crs_dst_summer = _create_hull(
                    bird_year_summer_events_expanded['centroid_lat'],
                    bird_year_summer_events_expanded['centroid_lon'],
                    margin_km=hull_margin_km
                )
                to_xy_winter, hull_xy_winter, crs_dst_winter = _create_hull(
                    bird_year_winter_events_expanded['centroid_lat'],
                    bird_year_winter_events_expanded['centroid_lon'],
                    margin_km=hull_margin_km
                )
                is_inside_summer = bird_year_observations.apply(lambda r: _inside_hull(to_xy_summer, hull_xy_summer, r['lat'], r['lon']), axis=1)
                is_inside_winter = bird_year_observations.apply(lambda r: _inside_hull(to_xy_winter, hull_xy_winter, r['lat'], r['lon']), axis=1)

                winter_departure_day = first_confirmed_exit_day(bird_year_observations, _ts(y - 1, winter_start_month_day), departure_confirm_days, is_inside_winter)
                summer_arrival_day = first_confirmed_entry_day(bird_year_observations, _ts(y, summer_start_month_day), is_inside_summer)

                if winter_departure_day is not None and summer_arrival_day is not None and summer_arrival_day > winter_departure_day:
                    pts = df_id[(df_id["date"] >= winter_departure_day) & (df_id["date"] <= summer_arrival_day)].copy()
                    dur_h = (summer_arrival_day - winter_departure_day).total_seconds() / 3600.0

                    ev_win = ev_id[(ev_id["end_time"] >= winter_departure_day) & (ev_id["start_time"] <= summer_arrival_day)].copy()
                    overlap_h = np.array(
                        [_calculate_overlap_hours(winter_departure_day, summer_arrival_day, s, e) for s, e in zip(ev_win["start_time"], ev_win["end_time"])],
                        dtype=float,
                    )
                    stop_h = float(np.nansum(overlap_h)) if overlap_h.size else 0.0
                    n_stop = int(len(ev_win))
                    n_major = int(np.sum(ev_win["duration_hours"].to_numpy(float) >= major_stopover_hours))

                    n_sites = int(pd.Series(ev_win["site_id"]).dropna().nunique())

                    step_stats = _migration_step_stats(pts, gap_split_days=gap_split_days)

                    travel_h = max(0.0, dur_h - stop_h)
                    speed_kmph = (
                        float(step_stats["route_distance_km"] / travel_h)
                        if np.isfinite(step_stats["route_distance_km"]) and travel_h > 0
                        else np.nan
                    )

                    out_rows.append(
                        dict(
                            id=pid,
                            season="spring",
                            season_year=y,
                            departure_time=winter_departure_day,
                            arrival_time=summer_arrival_day,
                            duration_days=float(dur_h / 24.0),
                            stopover_hours=stop_h,
                            travel_hours=travel_h,
                            n_stopovers=n_stop,
                            n_major_stopovers=n_major,
                            n_sites_used=n_sites,
                            travel_speed_kmph=speed_kmph,
                            **step_stats,
                        )
                    )

            # Fall Migration
            if len(bird_year_summer_events_expanded) > 0 and len(bird_next_year_winter_events_expanded) > 0:
                bird_year_observations = df_id[df_id['date'].dt.year == y]
                to_xy_summer, hull_xy_summer, crs_dst_summer = _create_hull(
                    bird_year_summer_events_expanded['centroid_lat'],
                    bird_year_summer_events_expanded['centroid_lon'],
                    margin_km=hull_margin_km
                )
                to_xy_winter, hull_xy_winter, crs_dst_winter = _create_hull(
                    bird_next_year_winter_events_expanded['centroid_lat'],
                    bird_next_year_winter_events_expanded['centroid_lon'],
                    margin_km=hull_margin_km
                )
                is_inside_summer = bird_year_observations.apply(lambda r: _inside_hull(to_xy_summer, hull_xy_summer, r['lat'], r['lon']), axis=1)
                is_inside_winter = bird_year_observations.apply(lambda r: _inside_hull(to_xy_winter, hull_xy_winter, r['lat'], r['lon']), axis=1)

                summer_departure_day = first_confirmed_exit_day(bird_year_observations, _ts(y, summer_start_month_day), departure_confirm_days, is_inside_summer)
                winter_arrival_day = first_confirmed_entry_day(bird_year_observations, _ts(y, winter_start_month_day), is_inside_winter)

                if summer_departure_day is not None and winter_arrival_day is not None and winter_arrival_day > summer_departure_day:
                    pts = df_id[(df_id["date"] >= summer_departure_day) & (df_id["date"] <= winter_arrival_day)].copy()
                    dur_h = (winter_arrival_day - summer_departure_day).total_seconds() / 3600.0

                    ev_win = ev_id[(ev_id["end_time"] >= summer_departure_day) & (ev_id["start_time"] <= winter_arrival_day)].copy()
                    overlap_h = np.array(
                        [_calculate_overlap_hours(summer_departure_day, winter_arrival_day, s, e) for s, e in zip(ev_win["start_time"], ev_win["end_time"])],
                        dtype=float,
                    )
                    stop_h = float(np.nansum(overlap_h)) if overlap_h.size else 0.0
                    n_stop = int(len(ev_win))
                    n_major = int(np.sum(ev_win["duration_hours"].to_numpy(float) >= major_stopover_hours))

                    n_sites = int(pd.Series(ev_win["site_id"]).dropna().nunique())

                    step_stats = _migration_step_stats(pts, gap_split_days=gap_split_days)

                    travel_h = max(0.0, dur_h - stop_h)
                    speed_kmph = (
                        float(step_stats["route_distance_km"] / travel_h)
                        if np.isfinite(step_stats["route_distance_km"]) and travel_h > 0
                        else np.nan
                    )

                    out_rows.append(
                        dict(
                            id=pid,
                            season="fall",
                            season_year=y,
                            departure_time=summer_departure_day,
                            arrival_time=winter_arrival_day,
                            duration_days=float(dur_h / 24.0),
                            stopover_hours=stop_h,
                            travel_hours=travel_h,
                            n_stopovers=n_stop,
                            n_major_stopovers=n_major,
                            n_sites_used=n_sites,
                            travel_speed_kmph=speed_kmph,
                            **step_stats,
                        )
                    )

    if not out_rows:
        return pd.DataFrame()

    mig = pd.DataFrame(out_rows).sort_values(["id", "season", "season_year"]).reset_index(drop=True)
    mig["departure_doy"] = mig["departure_time"].dt.dayofyear.astype(int)
    mig["arrival_doy"] = mig["arrival_time"].dt.dayofyear.astype(int)
    # print(mig)
    return mig

def summarize_individuals(mig: pd.DataFrame) -> pd.DataFrame:
    if mig.empty:
        return pd.DataFrame()

    g = mig.groupby(["id", "season"], as_index=False).agg(
        n_migrations=("season_year", "size"),
        mean_duration_days=("duration_days", "mean"),
        mean_stopover_hours=("stopover_hours", "mean"),
        mean_travel_speed_kmph=("travel_speed_kmph", "mean"),
        mean_n_stopovers=("n_stopovers", "mean"),
        mean_n_major_stopovers=("n_major_stopovers", "mean"),
        mean_n_sites_used=("n_sites_used", "mean"),
    )
    return g.sort_values(["season", "n_migrations"], ascending=[True, False]).reset_index(drop=True)

def summarize_population(mig: pd.DataFrame) -> pd.DataFrame:
    if mig.empty:
        return pd.DataFrame()

    g = mig.groupby(["season"], as_index=False).agg(
        n_migrations=("id", "size"),
        n_individuals=("id", "nunique"),
        mean_departure_doy=("departure_doy", "mean"),
        sd_departure_doy=("departure_doy", "std"),
        mean_arrival_doy=("arrival_doy", "mean"),
        sd_arrival_doy=("arrival_doy", "std"),
        mean_duration_days=("duration_days", "mean"),
        sd_duration_days=("duration_days", "std"),
        mean_stopover_hours=("stopover_hours", "mean"),
        sd_stopover_hours=("stopover_hours", "std"),
        mean_travel_speed_kmph=("travel_speed_kmph", "mean"),
        sd_travel_speed_kmph=("travel_speed_kmph", "std"),
    )
    return g.sort_values(["season"]).reset_index(drop=True)

def run_phenology(
    *,
    rw_csv_path: str,
    stopover_events_csv_path: str,
    out_migrations_csv_path: str,
    out_individuals_csv_path: str,
    out_population_csv_path: str,
    hull_margin_km: float,
    winter_months: Tuple[int, ...],
    breeding_months: Tuple[int, ...],
    min_points_winter: int,
    min_points_breeding: int,
    major_stopover_hours: float,
    gap_split_days: float,
    winter_start_month_day: Tuple[int, int],
    winter_end_month_day: Tuple[int, int],
    summer_start_month_day: Tuple[int, int],
    summer_end_month_day: Tuple[int, int],
    departure_confirm_days: int,
    link_radius_km: int,
) -> None:
    track = read_rw_track(rw_csv_path)
    events = read_stopover_events(stopover_events_csv_path)

    mig = compute_phenology(
        track,
        events,
        hull_margin_km=hull_margin_km,
        winter_months=winter_months,
        breeding_months=breeding_months,
        min_points_winter=min_points_winter,
        min_points_breeding=min_points_breeding,
        major_stopover_hours=major_stopover_hours,
        gap_split_days=gap_split_days,
        winter_start_month_day=winter_start_month_day,
        winter_end_month_day=winter_end_month_day,
        summer_start_month_day=summer_start_month_day,
        summer_end_month_day=summer_end_month_day,
        departure_confirm_days=departure_confirm_days,
        link_radius_km=link_radius_km
    )

    ind = summarize_individuals(mig)
    pop = summarize_population(mig)

    Path(out_migrations_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_individuals_csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_population_csv_path).parent.mkdir(parents=True, exist_ok=True)

    mig.to_csv(out_migrations_csv_path, index=False)
    ind.to_csv(out_individuals_csv_path, index=False)
    pop.to_csv(out_population_csv_path, index=False)

def _read_phenology_outputs(
    migrations_csv_path: str,
    individuals_csv_path: str,
    population_csv_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mig = pd.read_csv(migrations_csv_path)
    ind = pd.read_csv(individuals_csv_path)
    pop = pd.read_csv(population_csv_path)

    if "departure_time" in mig.columns:
        mig["departure_time"] = pd.to_datetime(mig["departure_time"], utc=True, errors="coerce")
    if "arrival_time" in mig.columns:
        mig["arrival_time"] = pd.to_datetime(mig["arrival_time"], utc=True, errors="coerce")

    for c in ["departure_doy", "arrival_doy", "duration_days", "stopover_hours", "travel_speed_kmph", "route_distance_km"]:
        if c in mig.columns:
            mig[c] = pd.to_numeric(mig[c], errors="coerce")

    if "season_year" in mig.columns:
        mig["season_year"] = pd.to_numeric(mig["season_year"], errors="coerce")

    if "season" in mig.columns:
        mig["season"] = mig["season"].astype(str)

    return mig, ind, pop

def _save_fig(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def visualize_phenology_outputs(
    *,
    migrations_csv_path: str,
    individuals_csv_path: str,
    population_csv_path: str,
    out_dir: str = "outputs/phenology",
    dpi: int = 180,
    max_points_scatter: int = 20000,
) -> None:
    out = Path(out_dir)
    mig, ind, pop = _read_phenology_outputs(migrations_csv_path, individuals_csv_path, population_csv_path)

    sns.set_theme(style="whitegrid")

    if mig.empty:
        out.mkdir(parents=True, exist_ok=True)
        (out / "README.txt").write_text("No migration episodes found in migrations.csv\n", encoding="utf-8")
        return
    
    mig = mig.sort_values(["season"])

    mig_scatter = mig.copy()
    if len(mig_scatter) > max_points_scatter:
        mig_scatter = mig_scatter.sample(n=max_points_scatter, random_state=0)

    # 1) Departure timing distribution
    if "departure_doy" in mig.columns and "season" in mig.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(data=mig, x="departure_doy", hue="season", bins=35, element="step", stat="density", common_norm=False)
        plt.xlabel("Departure day-of-year")
        plt.ylabel("Density")
        plt.xticks(ticks=range(0, 367, 10), rotation=45)
        plt.title("Departure timing (DOY)")
        _save_fig(out / "departure_doy.png", dpi)

    # 2) Arrival timing distribution
    if "arrival_doy" in mig.columns and "season" in mig.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(data=mig, x="arrival_doy", hue="season", bins=35, element="step", stat="density", common_norm=False)
        plt.xlabel("Arrival day-of-year")
        plt.ylabel("Density")
        plt.xticks(ticks=range(0, 367, 10), rotation=45)
        plt.title("Arrival timing (DOY)")
        _save_fig(out / "arrival_doy.png", dpi)

    # 3) Duration by season
    if "duration_days" in mig.columns and "season" in mig.columns:
        plt.figure(figsize=(8.5, 5))
        sns.boxplot(data=mig, x="season", y="duration_days", hue="season", palette="mako_r")
        sns.stripplot(data=mig_scatter, x="season", y="duration_days", size=2, alpha=0.35)
        plt.xlabel("")
        plt.ylabel("Migration duration (days)")
        plt.title("Migration duration by season")
        _save_fig(out / "duration_by_season.png", dpi)

    # 4) Year-to-year mean timing (if enough data available)
    if {"season_year", "departure_doy", "season"}.issubset(mig.columns) and mig['season_year'].nunique() > 4:
        by_year = (
            mig.groupby(["season", "season_year"], as_index=False)
            .agg(mean_departure_doy=("departure_doy", "mean"))
            .dropna(subset=["season_year", "mean_departure_doy"])
            .sort_values(["season", "season_year"])
        )
        if not by_year.empty:
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=by_year, x="season_year", y="mean_departure_doy", hue="season", marker="o")
            plt.xlabel("Season year")
            plt.ylabel("Mean departure DOY")
            plt.title("Year-to-year departure timing")
            _save_fig(out / "departure_trend_by_year.png", dpi)