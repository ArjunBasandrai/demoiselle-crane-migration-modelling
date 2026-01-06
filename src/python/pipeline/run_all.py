from .read import read_data
from .rw import run_random_walk
from .stopover import extract_stopover_sites
from .plot import plot_rw_segments_and_stopover_sites_plotly

import plotly.io as pio
pio.renderers.default = "browser"

import yaml

def run_pipeline():
    with open("configs/project.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Read raw data
    print('Reading raw data...')
    read_config = config['stage_read']
    read_data(
        read_config['raw_data_path'],
        min_count=read_config['min_count'],
        max_latitude=read_config['max_latitude'],
        save_path=read_config['stage_1_output_path']
    )

    # 2. Random Walk
    print('Running random walk...')
    run_random_walk()

    # 3. Stopover sites
    print('Extracting stopover sites...')
    stopover_config = config['stage_stopover']
    extract_stopover_sites(
        data_path=config['stage_rw']['stage_2_output_path'],
        stopover_events_path=stopover_config['stage_3_events_with_sites_output_path'],
        stopover_sites_path=stopover_config['stage_3_sites_output_path'],
        stopover_sites_contributors_path=stopover_config['stage_3_site_contributors_output_path'],
        r_stop_km=stopover_config['r_stop_km'],
        min_stop_hours=stopover_config['min_stop_hours'],
        site_eps_km=stopover_config['site_eps_km'],
        min_samples=stopover_config['min_samples']
    )

    plot_rw_segments_and_stopover_sites_plotly(
        stopover_sites_csv_path=stopover_config['stage_3_sites_output_path'],
        stopover_sites_contributors_path=stopover_config['stage_3_site_contributors_output_path'],
        out_path=stopover_config['stage_3_stopover_map_output_path']
    ).show(config={"scrollZoom": True, "displayModeBar": True})



