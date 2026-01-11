from .read import read_data
from .rw import run_random_walk
from .stopover import extract_stopover_sites
from .plot import plot_rw_segments_and_stopover_sites_plotly
from .dbmm import filter_migration_months, run_dynamic_brownian_bridge_movement_model
from .phenology import run_phenology, visualize_phenology_outputs

import yaml

import plotly.io as pio
pio.renderers.default = "browser"

def run_pipeline():
    with open("configs/project.yaml", "r") as f:
        config = yaml.safe_load(f)

    # # 1. Read raw data
    # print("Reading raw data...")
    # read_config = config["stage_read"]
    # read_data(
    #     read_config["raw_data_path"],
    #     min_count=read_config["min_count"],
    #     max_latitude=read_config["max_latitude"],
    #     save_path=read_config["stage_1_output_path"],
    # )

    # # 2. Random Walk
    # print("Running random walk...")
    # run_random_walk()

    # # 3. Stopover sites
    # print("Extracting stopover sites...")
    stopover_config = config["stage_stopover"]
    # extract_stopover_sites(
    #     data_path=config["stage_rw"]["stage_2_output_path"],
    #     stopover_events_path=stopover_config["stage_3_events_with_sites_output_path"],
    #     stopover_sites_path=stopover_config["stage_3_sites_output_path"],
    #     stopover_sites_contributors_path=stopover_config["stage_3_site_contributors_output_path"],
    #     r_stop_km=stopover_config["r_stop_km"],
    #     min_stop_hours=stopover_config["min_stop_hours"],
    #     site_eps_km=stopover_config["site_eps_km"],
    #     min_samples=stopover_config["min_samples"],
    # )

    # plot_rw_segments_and_stopover_sites_plotly(
    #     stopover_sites_csv_path=stopover_config["stage_3_sites_output_path"],
    #     stopover_sites_contributors_path=stopover_config["stage_3_site_contributors_output_path"],
    #     out_path=stopover_config["stage_3_stopover_map_output_path"],
    # ).show(config={"scrollZoom": True, "displayModeBar": True})

    # 4. Phenology
    print("Computing migration phenology...")
    phenology_config = config.get("stage_phenology") or config.get("stage_pheonology")

    run_phenology(
        rw_csv_path=config["stage_rw"]["stage_2_output_path"],
        stopover_events_csv_path=stopover_config["stage_3_events_with_sites_output_path"],
        out_migrations_csv_path=phenology_config["stage_4_migrations_output_path"],
        out_individuals_csv_path=phenology_config["stage_4_individuals_output_path"],
        out_population_csv_path=phenology_config["stage_4_population_output_path"],
        core_radius_km=float(phenology_config["core_radius_km"]),
        winter_months=tuple(phenology_config["winter_months"]),
        breeding_months=tuple(phenology_config["breeding_months"]),
        min_points_winter=int(phenology_config["min_points_winter"]),
        min_points_breeding=int(phenology_config["min_points_breeding"]),
        major_stopover_hours=float(phenology_config["major_stopover_hours"]),
        gap_split_days=float(phenology_config["gap_split_days"]),
        spring_arrival_start_month_day=tuple(phenology_config["spring_arrival_start_month_day"]),
        spring_arrival_end_month_day=tuple(phenology_config["spring_arrival_end_month_day"]),
        spring_departure_start_month_day=tuple(phenology_config["spring_departure_start_month_day"]),
        spring_departure_end_month_day=tuple(phenology_config["spring_departure_end_month_day"]),
        fall_departure_start_month_day=tuple(phenology_config["fall_departure_start_month_day"]),
        fall_departure_end_month_day=tuple(phenology_config["fall_departure_end_month_day"]),
        fall_arrival_start_month_day=tuple(phenology_config["fall_arrival_start_month_day"]),
        fall_arrival_end_month_day=tuple(phenology_config["fall_arrival_end_month_day"]),
    )

    visualize_phenology_outputs(
        migrations_csv_path=phenology_config["stage_4_migrations_output_path"],
        individuals_csv_path=phenology_config["stage_4_individuals_output_path"],
        population_csv_path=phenology_config["stage_4_population_output_path"],
        out_dir=phenology_config.get("plots_out_dir", "outputs/phenology"),
        dpi=int(phenology_config.get("plots_dpi", 180)),
        max_points_scatter=int(phenology_config.get("max_points_scatter", 20000)),
    )

    # 5. Corridor mapping using dBMM
    # print("Mapping migration corridors...")
    # corridor_config = config["stage_corridor"]
    # filter_migration_months(
    #     input_csv_path=read_config["stage_1_output_path"],
    #     output_csv_path=corridor_config["stage_4_migration_only_output_path"],
    #     spring_migration_start_month=corridor_config["spring_migration_start_month"],
    #     spring_migration_end_month=corridor_config["spring_migration_end_month"],
    #     fall_migration_start_month=corridor_config["fall_migration_start_month"],
    #     fall_migration_end_month=corridor_config["fall_migration_end_month"],
    # )

    # run_dynamic_brownian_bridge_movement_model()
