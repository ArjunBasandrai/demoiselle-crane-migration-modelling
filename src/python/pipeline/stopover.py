from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Any, Dict, List, Tuple, Union

EARTH_RADIUS_KM = 6371.0088
ArrayLike = Union[float, np.ndarray]

def haversine_km(lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike) -> ArrayLike:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

def detect_stopovers_one_segment(
    seg_df: pd.DataFrame,
    r_stop_km: float,
    min_stop_hours: float,
) -> List[Dict[str, Any]]:
    seg_df = seg_df.sort_values("date")

    lat_deg = seg_df["lat"].to_numpy(dtype=float)
    lon_deg = seg_df["lon"].to_numpy(dtype=float)
    t = seg_df["date"].to_numpy(dtype="datetime64[ns]")

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    n = len(seg_df)
    i = 0
    out: List[Dict[str, Any]] = []

    min_stop_ns = np.int64(min_stop_hours * 3600 * 1e9)

    while i < n - 1:
        d = haversine_km(lat[i], lon[i], lat[i + 1 :], lon[i + 1 :])
        idx = np.flatnonzero(d > r_stop_km)
        if idx.size == 0:
            j = n
        else:
            j = i + 1 + int(idx[0])

        start_t = t[i]
        end_t = t[j - 1]
        dur_ns = (end_t - start_t) / np.timedelta64(1, "ns")

        if dur_ns >= min_stop_ns:
            sl = slice(i, j)
            out.append(
                {
                    "id": seg_df["id"].iloc[0],
                    "seg_id": seg_df["seg_id"].iloc[0],
                    "start_time": pd.Timestamp(start_t, tz="UTC"),
                    "end_time": pd.Timestamp(end_t, tz="UTC"),
                    "duration_hours": float(dur_ns) / (3600 * 1e9),
                    "centroid_lon": float(np.mean(lon_deg[sl])),
                    "centroid_lat": float(np.mean(lat_deg[sl])),
                    "n_points": int(j - i),
                }
            )
            i = j
        else:
            i += 1

    return out

def build_stopover_events(
    final_csv_path: str,
    r_stop_km: float = 10.0,
    min_stop_hours: float = 24.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(final_csv_path)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

    df = df.dropna(subset=["id", "seg_id", "date", "lon", "lat"]).sort_values(["id", "seg_id", "date"])

    events: List[Dict[str, Any]] = []
    # Group by (id, seg_id) to avoid mixing individuals if seg_id is not globally unique
    for _, seg_df in df.groupby(["id", "seg_id"], sort=False):
        events.extend(detect_stopovers_one_segment(seg_df, r_stop_km=r_stop_km, min_stop_hours=min_stop_hours))

    events_df = pd.DataFrame(events)
    if len(events_df) == 0:
        return df, events_df

    events_df = events_df.sort_values(["id", "start_time"]).reset_index(drop=True)
    events_df.insert(0, "event_id", np.arange(len(events_df), dtype=int))
    return df, events_df

def cluster_stopover_sites(
    events_df: pd.DataFrame,
    site_eps_km: float = 20.0,
    min_samples: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(events_df) == 0:
        sites_df = pd.DataFrame(
            columns=[
                "site_id",
                "site_lon",
                "site_lat",
                "n_events",
                "n_individuals",
                "median_duration_hours",
                "is_noise_site",
            ]
        )
        return events_df.copy(), sites_df

    X = np.radians(events_df[["centroid_lat", "centroid_lon"]].to_numpy())
    eps_rad = site_eps_km / EARTH_RADIUS_KM

    labels = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(X)

    ev = events_df.copy()
    ev["dbscan_label"] = labels
    ev["is_noise"] = ev["dbscan_label"] == -1
    ev["site_cluster"] = ev["dbscan_label"]

    max_label = ev["site_cluster"].max()
    if max_label < 0:
        max_label = -1

    noise_mask = ev["site_cluster"] == -1
    if noise_mask.any():
        noise_ids = np.arange(max_label + 1, max_label + 1 + noise_mask.sum(), dtype=int)
        ev.loc[noise_mask, "site_cluster"] = noise_ids

    ev["site_id"] = ev["site_cluster"].astype(int)

    sites_df = (
        ev.groupby("site_id", as_index=False)
        .agg(
            site_lon=("centroid_lon", "median"),
            site_lat=("centroid_lat", "median"),
            n_events=("event_id", "size"),
            n_individuals=("id", "nunique"),
            median_duration_hours=("duration_hours", "median"),
            is_noise_site=("is_noise", "any"),
        )
        .sort_values(["n_individuals", "n_events", "median_duration_hours", "is_noise_site"], ascending=False)
        .reset_index(drop=True)
    )

    return ev.drop(columns=["site_cluster"]), sites_df

def build_sites_contributors_df(
    df: pd.DataFrame,
    events_with_sites_df: pd.DataFrame,
    sites_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per *raw observation* (from df) that falls within a detected stopover event.

    Output columns:
      site_id, site_lat, site_lon, is_noise_site, id, seg_id, date, lon, lat
    """
    out_cols = ["site_id", "site_lat", "site_lon", "is_noise_site", "id", "seg_id", "date", "lon", "lat"]

    if df.empty or events_with_sites_df.empty or sites_df.empty:
        return pd.DataFrame(columns=out_cols)

    df2 = df[["id", "seg_id", "date", "lon", "lat"]].copy()
    df2["date"] = pd.to_datetime(df2["date"], utc=True, errors="coerce")
    df2["lon"] = pd.to_numeric(df2["lon"], errors="coerce")
    df2["lat"] = pd.to_numeric(df2["lat"], errors="coerce")
    df2 = df2.dropna(subset=["id", "seg_id", "date", "lon", "lat"]).sort_values(["id", "seg_id", "date"])

    ev = events_with_sites_df[["id", "seg_id", "start_time", "end_time", "site_id"]].copy()
    ev["start_time"] = pd.to_datetime(ev["start_time"], utc=True, errors="coerce")
    ev["end_time"] = pd.to_datetime(ev["end_time"], utc=True, errors="coerce")
    ev = ev.dropna(subset=["id", "seg_id", "start_time", "end_time", "site_id"]).sort_values(
        ["id", "seg_id", "start_time"]
    )

    site_meta = sites_df.set_index("site_id")[["site_lat", "site_lon", "is_noise_site"]]

    ev_groups: Dict[Tuple[Any, Any], pd.DataFrame] = {
        key: g.reset_index(drop=True) for key, g in ev.groupby(["id", "seg_id"], sort=False)
    }

    parts: List[pd.DataFrame] = []

    for (pid, sid), obs_g in df2.groupby(["id", "seg_id"], sort=False):
        ev_g = ev_groups.get((pid, sid))
        if ev_g is None or ev_g.empty:
            continue

        obs_t = obs_g["date"].to_numpy(dtype="datetime64[ns]")
        ev_start = ev_g["start_time"].to_numpy(dtype="datetime64[ns]")
        ev_end = ev_g["end_time"].to_numpy(dtype="datetime64[ns]")
        ev_site = ev_g["site_id"].to_numpy(dtype=int)

        keep_idx: List[int] = []
        site_ids: List[int] = []

        j = 0
        m = len(ev_g)

        for i, t in enumerate(obs_t):
            while j < m and t > ev_end[j]:
                j += 1
            if j >= m:
                break
            if ev_start[j] <= t <= ev_end[j]:
                keep_idx.append(i)
                site_ids.append(int(ev_site[j]))

        if not keep_idx:
            continue

        kept = obs_g.iloc[keep_idx].copy()
        kept.insert(0, "site_id", np.array(site_ids, dtype=int))

        kept = kept.merge(
            site_meta.reset_index(),
            on="site_id",
            how="left",
            validate="many_to_one",
        )

        kept = kept[out_cols]
        parts.append(kept)

    if not parts:
        return pd.DataFrame(columns=out_cols)

    return pd.concat(parts, ignore_index=True)

def extract_stopover_sites(
    *,
    data_path: str,
    stopover_events_path: str,
    stopover_sites_path: str,
    stopover_sites_contributors_path: str,
    r_stop_km: float,
    min_stop_hours: float,
    site_eps_km: float,
    min_samples: int,
) -> None:
    df, events_df = build_stopover_events(
        final_csv_path=data_path,
        r_stop_km=r_stop_km,
        min_stop_hours=min_stop_hours,
    )

    events_with_sites_df, sites_df = cluster_stopover_sites(
        events_df,
        site_eps_km=site_eps_km,
        min_samples=min_samples,
    )

    sites_contributors_df = build_sites_contributors_df(df, events_with_sites_df, sites_df)

    events_with_sites_df.to_csv(stopover_events_path, index=False)
    sites_df.to_csv(stopover_sites_path, index=False)
    sites_contributors_df.to_csv(stopover_sites_contributors_path, index=False)
