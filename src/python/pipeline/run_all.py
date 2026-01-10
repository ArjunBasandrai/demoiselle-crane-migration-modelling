from .read import read_data
from .rw import run_random_walk
from .stopover import extract_stopover_sites
from .plot import plot_rw_segments_and_stopover_sites_plotly
from .dbmm import filter_migration_months, run_dynamic_brownian_bridge_movement_model

import yaml

import plotly.io as pio
pio.renderers.default = "browser"

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

    # 4. Corridor mapping using dBMM
    print('Mapping migration corridors...')
    corridor_config=config['stage_corridor']
    filter_migration_months(
        input_csv_path=read_config['stage_1_output_path'],
        output_csv_path=corridor_config['stage_4_data_output_path'],
        spring_migration_start_month=corridor_config['spring_migration_start_month'],
        spring_migration_end_month=corridor_config['spring_migration_end_month'],
        fall_migration_start_month=corridor_config['fall_migration_start_month'],
        fall_migration_end_month=corridor_config['fall_migration_end_month']
    )

    run_dynamic_brownian_bridge_movement_model()