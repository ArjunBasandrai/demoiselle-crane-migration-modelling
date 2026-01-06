import os
import pandas as pd

def read_data(
    data_path: str,
    *,
    min_count: int,
    max_latitude: float,
    save_path: str,
) -> None:
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.sort_values(by=['individual-local-identifier', 'timestamp'])
    df = df.drop_duplicates(
        subset=["individual-local-identifier", "timestamp"],
        keep="first"
    )

    low_data_individuals = (
        df
        .groupby('individual-local-identifier')
        .agg(count=('event-id', 'size'))
        .query(f'count < {min_count}')
        .index
    )

    df = df.query(
        "`individual-local-identifier` not in @low_data_individuals",
    )

    winter = df[df['timestamp'].dt.month.isin([12, 1, 2])].copy()

    winter_centroids = (
        winter
        .groupby('individual-local-identifier', as_index=False)
        .agg(
            centroid_lon=('location-long', 'mean'),
            centroid_lat=('location-lat', 'mean'),
            n_winter=('event-id', 'size')
        )
    )

    keep_ids = winter_centroids.loc[winter_centroids['centroid_lat'] < max_latitude, 'individual-local-identifier']

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = df[df['individual-local-identifier'].isin(keep_ids)].to_csv(save_path)
    
    # winter_centroids_kept = winter_centroids[winter_centroids['individual-local-identifier'].isin(keep_ids)].copy()