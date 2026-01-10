from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

@dataclass(frozen=True)
class PlotConfig:
    title: str = "Stopover sites and contributing observations"
    height: int = 720

    # Sites (drawn ABOVE hulls and contributors)
    site_marker_size: int = 12
    site_halo_size: int = 24
    site_color: str = "#d62728"        # non-noise sites
    noise_site_color: str = "#1f77b4"  # noise sites

    # Hull styling (always shown for visible sites)
    hull_line_width: float = 1.2
    hull_opacity: float = 0.14  # fill opacity

    # Contributors (drawn BELOW sites but ABOVE hulls)
    contributor_marker_size: int = 8
    contributor_opacity: float = 0.90
    contributor_color: str = "#111111"
    contributors_default_on: bool = False

    # Noise toggle
    noise_sites_default_on: bool = False

    show_legend: bool = True

    # Safety for huge files. Set to None to disable sampling.
    max_contributors: int | None = 200_000

    # Satellite tiles (no Mapbox token needed)
    raster_tile_url: str = (
        "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )

def _estimate_zoom(lat: np.ndarray, lon: np.ndarray) -> float:
    lat = lat[np.isfinite(lat)]
    lon = lon[np.isfinite(lon)]
    if lat.size == 0 or lon.size == 0:
        return 2.0

    lat_rng = float(np.max(lat) - np.min(lat))
    lon_rng = float(np.max(lon) - np.min(lon))
    rng = max(lat_rng, lon_rng)

    if rng <= 0:
        return 9.0

    zoom = np.log2(360.0 / (rng * 1.5))
    return float(np.clip(zoom, 1.0, 15.0))

def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def _convex_hull_lonlat(points: np.ndarray) -> np.ndarray:
    """
    Monotonic chain convex hull.
    points: (N, 2) array of [lon, lat].
    returns: (H, 2) hull vertices in order (NOT closed).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (N, 2) [lon, lat]")

    # Unique points
    pts = np.unique(pts, axis=0)
    n = len(pts)

    if n == 0:
        return pts
    if n == 1:
        return pts
    if n == 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]  # sort by lon, then lat

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull

def _fallback_small_hull(lon: float, lat: float, eps_deg: float = 0.01) -> np.ndarray:
    """
    For sites with <3 unique points: create a tiny triangle around the point so it can be filled.
    eps_deg default ~1.1 km latitude; small enough to not distort site footprint much.
    """
    return np.array(
        [
            [lon - eps_deg, lat - eps_deg],
            [lon + eps_deg, lat - eps_deg],
            [lon, lat + eps_deg],
        ],
        dtype=float,
    )

def plot_rw_segments_and_stopover_sites_plotly(
    stopover_sites_csv_path: str,
    stopover_sites_contributors_path: str,
    out_path: str | None = None,
    config: PlotConfig = PlotConfig(),
) -> go.Figure:
    sites = pd.read_csv(stopover_sites_csv_path)
    contrib_full = pd.read_csv(stopover_sites_contributors_path)

    # Required columns
    for col in ["site_id", "site_lat", "site_lon"]:
        if col not in sites.columns:
            raise ValueError(f"stopover_sites_csv_path must contain '{col}'")
    for col in ["site_id", "lat", "lon"]:
        if col not in contrib_full.columns:
            raise ValueError(f"stopover_sites_contributors_path must contain '{col}'")

    # Optional noise flag
    if "is_noise_site" not in sites.columns:
        sites = sites.assign(is_noise_site=False)
    if "is_noise_site" not in contrib_full.columns:
        contrib_full = contrib_full.assign(is_noise_site=False)

    # Coerce numeric
    sites["site_lat"] = pd.to_numeric(sites["site_lat"], errors="coerce")
    sites["site_lon"] = pd.to_numeric(sites["site_lon"], errors="coerce")
    sites["site_id"] = pd.to_numeric(sites["site_id"], errors="coerce")

    contrib_full["lat"] = pd.to_numeric(contrib_full["lat"], errors="coerce")
    contrib_full["lon"] = pd.to_numeric(contrib_full["lon"], errors="coerce")
    contrib_full["site_id"] = pd.to_numeric(contrib_full["site_id"], errors="coerce")

    sites = sites.dropna(subset=["site_id", "site_lat", "site_lon"]).copy()
    contrib_full = contrib_full.dropna(subset=["site_id", "lat", "lon"]).copy()

    sites["site_id"] = sites["site_id"].astype(int)
    contrib_full["site_id"] = contrib_full["site_id"].astype(int)

    sites["is_noise_site"] = sites["is_noise_site"].astype(bool)
    contrib_full["is_noise_site"] = contrib_full["is_noise_site"].astype(bool)

    # View bounds (stable framing)
    if len(contrib_full) > 0:
        all_lat = np.concatenate([sites["site_lat"].to_numpy(float), contrib_full["lat"].to_numpy(float)])
        all_lon = np.concatenate([sites["site_lon"].to_numpy(float), contrib_full["lon"].to_numpy(float)])
    else:
        all_lat = sites["site_lat"].to_numpy(float)
        all_lon = sites["site_lon"].to_numpy(float)

    center = {"lat": float(np.nanmean(all_lat)), "lon": float(np.nanmean(all_lon))}
    zoom = _estimate_zoom(all_lat, all_lon)

    # Split sites
    non_noise = sites[~sites["is_noise_site"]].copy()
    noise = sites[sites["is_noise_site"]].copy()
    noise_site_ids: set[int] = set(noise["site_id"].tolist())

    # Keep a copy for hull construction (full contribs, unsampled)
    contrib_for_hulls = contrib_full

    # Optional sampling for contributor markers only
    contrib = contrib_full
    if config.max_contributors is not None and len(contrib) > config.max_contributors:
        contrib = contrib.sample(n=config.max_contributors, random_state=0).copy()

    # Split contributors by whether they belong to a noise site_id
    is_noise_contrib = contrib["site_id"].isin(noise_site_ids)
    contrib_non_noise = contrib[~is_noise_contrib].copy()
    contrib_noise = contrib[is_noise_contrib].copy()

    fig = go.Figure()

    # ----------------------------
    # Hulls FIRST (beneath everything)
    # ----------------------------
    non_noise_hull_indices: list[int] = []
    noise_hull_indices: list[int] = []

    def _add_site_hull_traces(site_ids: list[int], color: str, visible: bool, store: list[int]) -> None:
        if len(site_ids) == 0 or contrib_for_hulls.empty:
            return

        # Group contributors once for speed
        g = contrib_for_hulls.groupby("site_id", sort=False)

        for sid in site_ids:
            if sid not in g.groups:
                continue

            pts_df = g.get_group(sid)[["lon", "lat"]].dropna()
            if pts_df.empty:
                continue

            pts = pts_df.to_numpy(dtype=float)
            pts = np.unique(pts, axis=0)

            if len(pts) >= 3:
                hull = _convex_hull_lonlat(pts)
                if len(hull) < 3:
                    hull = _fallback_small_hull(float(np.nanmedian(pts[:, 0])), float(np.nanmedian(pts[:, 1])))
            else:
                hull = _fallback_small_hull(float(np.nanmedian(pts[:, 0])), float(np.nanmedian(pts[:, 1])))

            # Close polygon
            hull_closed = np.vstack([hull, hull[0]])

            idx = len(fig.data)
            fig.add_trace(
                go.Scattermap(
                    lat=hull_closed[:, 1],
                    lon=hull_closed[:, 0],
                    mode="lines",
                    fill="toself",
                    fillcolor=f"rgba{tuple(int(color[i:i+2], 16) for i in (1, 3, 5)) + (config.hull_opacity,)}",
                    line=dict(color=color, width=config.hull_line_width),
                    name="Site hull",
                    showlegend=False,
                    hovertemplate=f"site_id={sid}<extra></extra>",
                    visible=visible,
                )
            )
            store.append(idx)

    # Add hulls for non-noise sites (always visible)
    _add_site_hull_traces(
        site_ids=non_noise["site_id"].tolist(),
        color=config.site_color,
        visible=True,
        store=non_noise_hull_indices,
    )

    # Add hulls for noise sites (toggleable)
    _add_site_hull_traces(
        site_ids=noise["site_id"].tolist(),
        color=config.noise_site_color,
        visible=bool(config.noise_sites_default_on),
        store=noise_hull_indices,
    )

    # ----------------------------
    # Contributors SECOND
    # ----------------------------
    contrib_non_noise_idx = len(fig.data)
    fig.add_trace(
        go.Scattermap(
            lat=contrib_non_noise["lat"],
            lon=contrib_non_noise["lon"],
            mode="markers",
            name="Contributors",
            visible=bool(config.contributors_default_on),
            marker=dict(
                size=config.contributor_marker_size,
                opacity=config.contributor_opacity,
                color=config.contributor_color,
            ),
            customdata=np.c_[contrib_non_noise["site_id"].to_numpy()],
            hovertemplate=(
                "site_id=%{customdata[0]}<br>"
                "lat=%{lat:.4f}<br>"
                "lon=%{lon:.4f}<extra></extra>"
            ),
        )
    )

    contrib_noise_idx = len(fig.data)
    fig.add_trace(
        go.Scattermap(
            lat=contrib_noise["lat"],
            lon=contrib_noise["lon"],
            mode="markers",
            name="Noise contributors",
            visible=bool(config.contributors_default_on) and bool(config.noise_sites_default_on),
            marker=dict(
                size=config.contributor_marker_size,
                opacity=config.contributor_opacity,
                color=config.contributor_color,
            ),
            customdata=np.c_[contrib_noise["site_id"].to_numpy()],
            hovertemplate=(
                "site_id=%{customdata[0]}<br>"
                "lat=%{lat:.4f}<br>"
                "lon=%{lon:.4f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # ----------------------------
    # Sites THIRD (always above)
    # ----------------------------
    def _site_customdata(df: pd.DataFrame) -> np.ndarray:
        has_all = all(c in df.columns for c in ["n_individuals", "n_events", "median_duration_hours"])
        if not has_all:
            return np.c_[df["site_id"].to_numpy()]
        return np.c_[
            df["site_id"].to_numpy(),
            df["n_individuals"].to_numpy(),
            df["n_events"].to_numpy(),
            df["median_duration_hours"].to_numpy(),
        ]

    def _site_hovertemplate(cd_cols: int) -> str:
        parts = ["site_id=%{customdata[0]}"]
        if cd_cols >= 4:
            parts += [
                "n_individuals=%{customdata[1]}",
                "n_events=%{customdata[2]}",
                "median_duration_h=%{customdata[3]:.1f}",
            ]
        parts += ["site_lat=%{lat:.4f}", "site_lon=%{lon:.4f}"]
        return "<br>".join(parts) + "<extra></extra>"

    def _add_site_layer(df: pd.DataFrame, name: str, color: str, visible: bool) -> tuple[int, int]:
        if df.empty:
            return (-1, -1)

        cd = _site_customdata(df)
        ht = _site_hovertemplate(cd.shape[1])

        halo_idx = len(fig.data)
        fig.add_trace(
            go.Scattermap(
                lat=df["site_lat"],
                lon=df["site_lon"],
                mode="markers",
                showlegend=False,
                hoverinfo="skip",
                visible=visible,
                marker=dict(size=config.site_halo_size, opacity=0.22, color=color),
            )
        )

        dot_idx = len(fig.data)
        fig.add_trace(
            go.Scattermap(
                lat=df["site_lat"],
                lon=df["site_lon"],
                mode="markers",
                name=name,
                visible=visible,
                marker=dict(size=config.site_marker_size, opacity=0.95, color=color),
                customdata=cd,
                hovertemplate=ht,
            )
        )
        return (halo_idx, dot_idx)

    # Non-noise sites always ON
    _add_site_layer(non_noise, "Stopover sites", config.site_color, visible=True)

    # Noise sites toggleable (default OFF)
    noise_halo_idx, noise_dot_idx = _add_site_layer(
        noise, "Noise sites", config.noise_site_color, visible=bool(config.noise_sites_default_on)
    )

    # ----------------------------
    # Dropdown states (no extra hull option)
    # Hull visibility follows site visibility.
    # ----------------------------
    n = len(fig.data)

    def _vis(contrib_on: bool, noise_on: bool) -> list[bool]:
        v = [True] * n

        # Contributors: noise contributors only when BOTH on
        v[contrib_non_noise_idx] = contrib_on
        v[contrib_noise_idx] = contrib_on and noise_on

        # Noise sites
        if noise_halo_idx != -1:
            v[noise_halo_idx] = noise_on
        if noise_dot_idx != -1:
            v[noise_dot_idx] = noise_on

        # Noise hulls follow noise toggle
        for idx in noise_hull_indices:
            v[idx] = noise_on

        # Non-noise hulls always on (already True)

        return v

    states = [
        ("Contrib: Off · Noise: Off", _vis(False, False)),
        ("Contrib: On · Noise: Off", _vis(True, False)),
        ("Contrib: Off · Noise: On", _vis(False, True)),
        ("Contrib: On · Noise: On", _vis(True, True)),
    ]

    default_idx = 0
    if config.contributors_default_on and config.noise_sites_default_on:
        default_idx = 3
    elif config.contributors_default_on and not config.noise_sites_default_on:
        default_idx = 1
    elif (not config.contributors_default_on) and config.noise_sites_default_on:
        default_idx = 2

    fig.update_layout(
        template="plotly_white",
        title=config.title,
        height=config.height,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=config.show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        map=dict(
            center=center,
            zoom=zoom,
            layers=[
                dict(
                    sourcetype="raster",
                    source=[config.raster_tile_url],
                    below="traces",
                )
            ],
        ),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
                showactive=True,
                active=default_idx,
                buttons=[dict(label=lbl, method="update", args=[{"visible": vis}]) for lbl, vis in states],
            )
        ],
    )

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() != ".html":
            out = out.with_suffix(".html")
        fig.write_html(
            str(out),
            include_plotlyjs="cdn",
            full_html=True,
            config={"scrollZoom": True, "displayModeBar": True},
        )

    return fig
