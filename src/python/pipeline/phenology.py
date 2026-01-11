from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib
# matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

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

def compute_phenology(
    track: pd.DataFrame,
    events: pd.DataFrame,
    *,
    core_radius_km: float,
    winter_months: Tuple[int, ...],
    breeding_months: Tuple[int, ...],
    min_points_winter: int,
    min_points_breeding: int,
    major_stopover_hours: float,
    gap_split_days: float,
    spring_arrival_start_month_day: Tuple[int, int],
    spring_arrival_end_month_day: Tuple[int, int],
    spring_departure_start_month_day: Tuple[int, int],
    spring_departure_end_month_day: Tuple[int, int],
    fall_departure_start_month_day: Tuple[int, int],
    fall_departure_end_month_day: Tuple[int, int],
    fall_arrival_start_month_day: Tuple[int, int],
    fall_arrival_end_month_day: Tuple[int, int],
) -> pd.DataFrame:
    winter_c, breed_c = compute_seasonal_centroids(
        track,
        winter_months=winter_months,
        breeding_months=breeding_months,
        min_points_winter=min_points_winter,
        min_points_breeding=min_points_breeding,
    )

    winter_map = winter_c.set_index(["id", "season_year"])
    breed_map = breed_c.set_index(["id", "season_year"])

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
            if (pid, y) in winter_map.index and (pid, y) in breed_map.index:
                wlat = float(winter_map.loc[(pid, y), "winter_lat"])
                wlon = float(winter_map.loc[(pid, y), "winter_lon"])
                blat = float(breed_map.loc[(pid, y), "breed_lat"])
                blon = float(breed_map.loc[(pid, y), "breed_lon"])

                dep = _last_within(
                    df_id,
                    wlat,
                    wlon,
                    core_radius_km,
                    _ts(y - 1, fall_departure_start_month_day),
                    _ts(y, fall_departure_end_month_day),
                )
                arr = None
                if dep is not None:
                    arr = _first_within(
                        df_id,
                        blat,
                        blon,
                        core_radius_km,
                        max(dep, _ts(y, spring_arrival_start_month_day)),
                        _ts(y, spring_arrival_end_month_day),
                    )

                if dep is not None and arr is not None and arr > dep:
                    pts = df_id[(df_id["date"] >= dep) & (df_id["date"] <= arr)].copy()
                    dur_h = (arr - dep).total_seconds() / 3600.0

                    ev_win = ev_id[(ev_id["end_time"] >= dep) & (ev_id["start_time"] <= arr)].copy()
                    overlap_h = np.array(
                        [_calculate_overlap_hours(dep, arr, s, e) for s, e in zip(ev_win["start_time"], ev_win["end_time"])],
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
                            departure_time=dep,
                            arrival_time=arr,
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

            if (pid, y) in breed_map.index and (pid, y + 1) in winter_map.index:
                blat = float(breed_map.loc[(pid, y), "breed_lat"])
                blon = float(breed_map.loc[(pid, y), "breed_lon"])
                wlat = float(winter_map.loc[(pid, y + 1), "winter_lat"])
                wlon = float(winter_map.loc[(pid, y + 1), "winter_lon"])

                dep = _last_within(
                    df_id,
                    blat,
                    blon,
                    core_radius_km,
                    _ts(y, spring_departure_start_month_day),
                    _ts(y, spring_departure_end_month_day),
                )
                arr = None
                if dep is not None:
                    arr = _first_within(
                        df_id,
                        wlat,
                        wlon,
                        core_radius_km,
                        max(dep, _ts(y, fall_arrival_start_month_day)),
                        _ts(y + 1, fall_arrival_end_month_day),
                    )

                if dep is not None and arr is not None and arr > dep:
                    pts = df_id[(df_id["date"] >= dep) & (df_id["date"] <= arr)].copy()
                    dur_h = (arr - dep).total_seconds() / 3600.0

                    ev_win = ev_id[(ev_id["end_time"] >= dep) & (ev_id["start_time"] <= arr)].copy()
                    overlap_h = np.array(
                        [_calculate_overlap_hours(dep, arr, s, e) for s, e in zip(ev_win["start_time"], ev_win["end_time"])],
                        dtype=float,
                    )
                    stop_h = float(np.nansum(overlap_h)) if overlap_h.size else 0.0
                    n_stop = int(len(ev_win))
                    n_major = int(np.sum(ev_win["duration_hours"].to_numpy(float) >= major_stopover_hours))

                    n_sites = np.nan
                    if "site_id" in ev_win.columns:
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
                            departure_time=dep,
                            arrival_time=arr,
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
    core_radius_km: float,
    winter_months: Tuple[int, ...],
    breeding_months: Tuple[int, ...],
    min_points_winter: int,
    min_points_breeding: int,
    major_stopover_hours: float,
    gap_split_days: float,
    spring_arrival_start_month_day: Tuple[int, int],
    spring_arrival_end_month_day: Tuple[int, int],
    spring_departure_start_month_day: Tuple[int, int],
    spring_departure_end_month_day: Tuple[int, int],
    fall_departure_start_month_day: Tuple[int, int],
    fall_departure_end_month_day: Tuple[int, int],
    fall_arrival_start_month_day: Tuple[int, int],
    fall_arrival_end_month_day: Tuple[int, int],
) -> None:
    track = read_rw_track(rw_csv_path)
    events = read_stopover_events(stopover_events_csv_path)

    mig = compute_phenology(
        track,
        events,
        core_radius_km=core_radius_km,
        winter_months=winter_months,
        breeding_months=breeding_months,
        min_points_winter=min_points_winter,
        min_points_breeding=min_points_breeding,
        major_stopover_hours=major_stopover_hours,
        gap_split_days=gap_split_days,
        spring_arrival_start_month_day=spring_arrival_start_month_day,
        spring_arrival_end_month_day=spring_arrival_end_month_day,
        spring_departure_start_month_day=spring_departure_start_month_day,
        spring_departure_end_month_day=spring_departure_end_month_day,
        fall_departure_start_month_day=fall_departure_start_month_day,
        fall_departure_end_month_day=fall_departure_end_month_day,
        fall_arrival_start_month_day=fall_arrival_start_month_day,
        fall_arrival_end_month_day=fall_arrival_end_month_day,
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