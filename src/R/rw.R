Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

RNGkind("Mersenne-Twister")

library(dplyr)
library(readr)
library(yaml)
library(foieGras)
library(ggplot2)

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0) y else x

# Config + data load
config_data  <- read_yaml("configs/project.yaml")
dataset_path <- config_data$stage_read$stage_1_output_path
save_path    <- config_data$stage_rw$stage_2_output_path

seed <- config_data$stage_rw$seed %||% 123
set.seed(seed)

df <- read_csv(
  dataset_path,
  show_col_types = FALSE,
  col_types = cols(timestamp = col_character())
)

# Parameters
time_step_hours <- config_data$stage_rw$time_step_hours
gap_split_days  <- config_data$stage_rw$gap_split_days
min_points      <- config_data$stage_rw$min_points

# FoieGras input prep
d <- df %>%
  transmute(
    id   = `individual-local-identifier`,
    date = as.POSIXct(timestamp, tz = "UTC"),
    lc   = "G",
    lon  = `location-long`,
    lat  = `location-lat`
  ) %>%
  filter(!is.na(date), !is.na(lon), !is.na(lat)) %>%
  arrange(id, date)

# Split tracks on gaps
d <- d %>%
  group_by(id) %>%
  mutate(
    dt_days = as.numeric(difftime(date, lag(date), units = "days")),
    seg     = cumsum(is.na(dt_days) | dt_days > gap_split_days) + 1L,
    seg_id  = paste(id, seg, sep = "__")
  ) %>%
  ungroup()

seg_ids <- d %>%
  count(seg_id, name = "n") %>%
  filter(n >= min_points) %>%
  arrange(seg_id) %>%
  pull(seg_id)

# RW fit (safe)
safe_fit <- function(di) {
  tryCatch(
    fit_ssm(
      di[, c("id", "date", "lc", "lon", "lat")],
      model = "rw",
      time.step = time_step_hours,
      spdf = FALSE,
      control = ssm_control(se = FALSE)
    ),
    error = function(e) e
  )
}

fits <- setNames(vector("list", length(seg_ids)), seg_ids)

for (i in seq_along(seg_ids)) {
  di <- d %>% filter(seg_id == seg_ids[i])
  fits[[i]] <- safe_fit(di)
}

ok <- !vapply(fits, inherits, logical(1), "error")
fits_ok <- fits[ok]

# Predictions
pred <- bind_rows(lapply(names(fits_ok), function(sid) {
  out <- grab(fits_ok[[sid]], what = "predicted", as_sf = FALSE)
  out$seg_id <- sid
  out
}))

# Consolidate segments
final_df <- pred %>%
  select(id, seg_id, date, lon, lat) %>%
  arrange(id, date)

# Save output
write_csv(final_df, save_path)

# Plot helper
plot_graph <- function(i) {
  seg_levels <- sort(unique(pred$seg_id))
  seg_choice <- seg_levels[i]
  
  pred_one <- pred %>%
    filter(seg_id == seg_choice) %>%
    arrange(date)
  
  orig_one <- d %>%
    filter(seg_id == seg_choice) %>%
    arrange(date)
  
  ggplot() +
    geom_path(
      data = pred_one,
      aes(x = lon, y = lat, color = "Predicted"),
      linewidth = 0.7
    ) +
    geom_point(
      data = pred_one,
      aes(x = lon, y = lat, color = "Predicted"),
      size = 1.2
    ) +
    geom_path(
      data = orig_one,
      aes(x = lon, y = lat, color = "Original"),
      linewidth = 0.5,
      alpha = 0.7
    ) +
    geom_point(
      data = orig_one,
      aes(x = lon, y = lat, color = "Original"),
      size = 0.9,
      alpha = 0.7
    ) +
    scale_color_manual(values = c("Predicted" = "#1f77b4", "Original" = "#ff7f0e")) +
    coord_equal() +
    labs(
      title = paste("RW predicted vs original track:", seg_choice),
      x = "Longitude",
      y = "Latitude",
      color = ""
    ) +
    theme_minimal()
}
