import pandas as pd
import subprocess

def filter_migration_months(
    input_csv_path: str,
    output_csv_path: str,
    spring_migration_start_month: int,
    spring_migration_end_month: int,
    fall_migration_start_month: int,
    fall_migration_end_month: int,
) -> None:
    """
    Reads a CSV and keeps only rows whose `timestamp` month falls within:
      - Spring migration window: [spring_migration_start_month, spring_migration_end_month] (inclusive, no wrap)
      - Fall migration window:   [fall_migration_start_month, fall_migration_end_month] (inclusive, wrap allowed)

    Sanity checks enforced:
      - spring_migration_end_month > spring_migration_start_month
      - fall_migration_end_month can be > or < fall_migration_start_month (wrap allowed)
      - fall_migration_end_month < spring_migration_start_month
      - fall_migration_start_month > spring_migration_end_month

    Writes the filtered rows to `output_csv_path`.
    """

    def validate_month_is_integer_between_1_and_12(parameter_name: str, month_value: int) -> int:
        if not isinstance(month_value, int):
            raise TypeError(
                f"{parameter_name} must be an int in [1..12]. Got: {type(month_value).__name__}"
            )
        if month_value < 1 or month_value > 12:
            raise ValueError(f"{parameter_name} must be in [1..12]. Got: {month_value}")
        return month_value

    spring_start_month_validated = validate_month_is_integer_between_1_and_12(
        "spring_migration_start_month",
        spring_migration_start_month,
    )
    spring_end_month_validated = validate_month_is_integer_between_1_and_12(
        "spring_migration_end_month",
        spring_migration_end_month,
    )
    fall_start_month_validated = validate_month_is_integer_between_1_and_12(
        "fall_migration_start_month",
        fall_migration_start_month,
    )
    fall_end_month_validated = validate_month_is_integer_between_1_and_12(
        "fall_migration_end_month",
        fall_migration_end_month,
    )

    if not (spring_end_month_validated > spring_start_month_validated):
        raise ValueError(
            "Sanity check failed: spring_migration_end_month "
            f"({spring_end_month_validated}) must be > spring_migration_start_month "
            f"({spring_start_month_validated})."
        )

    if (fall_end_month_validated < fall_start_month_validated and not fall_end_month_validated < spring_start_month_validated):
        raise ValueError(
            "Sanity check failed: fall_migration_end_month "
            f"({fall_end_month_validated}) must be < spring_migration_start_month "
            f"({spring_start_month_validated})."
        )

    if not (fall_start_month_validated > spring_end_month_validated):
        raise ValueError(
            "Sanity check failed: fall_migration_start_month "
            f"({fall_start_month_validated}) must be > spring_migration_end_month "
            f"({spring_end_month_validated})."
        )

    input_dataframe = pd.read_csv(input_csv_path)

    if "timestamp" not in input_dataframe.columns:
        raise KeyError("Input CSV must contain a 'timestamp' column.")

    parsed_timestamp_series = pd.to_datetime(input_dataframe["timestamp"], errors="coerce")

    if parsed_timestamp_series.isna().all():
        raise ValueError("All values in 'timestamp' failed to parse as datetimes.")

    valid_timestamp_mask = ~parsed_timestamp_series.isna()
    input_dataframe_with_valid_timestamps = input_dataframe.loc[valid_timestamp_mask].copy()

    month_extracted_from_timestamp = parsed_timestamp_series.loc[valid_timestamp_mask].dt.month
    input_dataframe_with_valid_timestamps["__timestamp_month__"] = month_extracted_from_timestamp

    spring_migration_month_mask = (
        (input_dataframe_with_valid_timestamps["__timestamp_month__"] >= spring_start_month_validated)
        & (input_dataframe_with_valid_timestamps["__timestamp_month__"] <= spring_end_month_validated)
    )

    if fall_start_month_validated <= fall_end_month_validated:
        fall_migration_month_mask = (
            (input_dataframe_with_valid_timestamps["__timestamp_month__"] >= fall_start_month_validated)
            & (input_dataframe_with_valid_timestamps["__timestamp_month__"] <= fall_end_month_validated)
        )
    else:
        # Wrap across year end (example: Oct to Feb)
        fall_migration_month_mask = (
            (input_dataframe_with_valid_timestamps["__timestamp_month__"] >= fall_start_month_validated)
            | (input_dataframe_with_valid_timestamps["__timestamp_month__"] <= fall_end_month_validated)
        )

    combined_migration_month_mask = spring_migration_month_mask | fall_migration_month_mask

    filtered_migration_dataframe = input_dataframe_with_valid_timestamps.loc[
        combined_migration_month_mask
    ].drop(columns=["__timestamp_month__"])

    filtered_migration_dataframe.to_csv(output_csv_path, index=False)

def run_dynamic_brownian_bridge_movement_model():
    result = subprocess.run(
        ["Rscript", "src/R/dbmm.R"],
        check=True
    )
