Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

RNGkind("Mersenne-Twister")

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(sf)
  library(sp)
  library(raster)
  library(move)
  library(plotly)
  library(htmlwidgets)
  library(yaml)
})

config_data  <- read_yaml("configs/project.yaml")
dbmm_config <- config_data$stage_corridor

r_tmp <- "outputs/tmp_rasters"
dir.create(r_tmp, recursive = TRUE, showWarnings = FALSE)
raster::rasterOptions(tmpdir = r_tmp)

write_tmp <- function(r) {
  raster::writeRaster(r, filename = raster::rasterTmpFile(), overwrite = TRUE)
}

cat('Segmenting data...\n')
input_path <- dbmm_config$stage_4_migration_only_output_path

gap_split_days <- config_data$stage_rw$gap_split_days
min_points <- config_data$stage_rw$min_points

parse_timestamp_utc <- function(x) {
  x <- as.character(x)
  t1 <- as.POSIXct(x, tz = "UTC")
  bad <- is.na(t1) & !is.na(x) & nzchar(x)
  if (any(bad)) {
    t2 <- as.POSIXct(strptime(x[bad], format = "%Y-%m-%d %H:%M:%OS", tz = "UTC"))
    t1[bad] <- t2
  }
  t1
}

raw <- read_csv(
  input_path,
  show_col_types = FALSE,
  col_types = cols(timestamp = col_character())
)

d <- raw %>%
  transmute(
    id = as.character(`individual-local-identifier`),
    date = parse_timestamp_utc(timestamp),
    lon = as.numeric(`location-long`),
    lat = as.numeric(`location-lat`)
  ) %>%
  filter(!is.na(id), !is.na(date), is.finite(lon), is.finite(lat)) %>%
  arrange(id, date) %>%
  distinct(id, date, .keep_all = TRUE)

d_seg <- d %>%
  group_by(id) %>%
  mutate(
    dt_days = as.numeric(difftime(date, lag(date), units = "days")),
    seg = cumsum(is.na(dt_days) | dt_days > gap_split_days) + 1L,
    seg_id = paste(id, seg, sep = "__")
  ) %>%
  ungroup()

seg_tbl <- d_seg %>%
  count(id, seg_id, name = "n_points") %>%
  arrange(id, seg_id)

seg_keep <- seg_tbl %>%
  filter(n_points >= min_points) %>%
  pull(seg_id)

d_seg_kept <- d_seg %>%
  filter(seg_id %in% seg_keep) %>%
  arrange(id, date)

cat('Creating raster...\n')

proj_crs <- dbmm_config$proj_crs

cellsize_m <- dbmm_config$cellsize_m
extent_buffer_m <- dbmm_config$extent_buffer_m

pts_sf <- st_as_sf(d_seg_kept, coords = c("lon", "lat"), crs = 4326, remove = FALSE)
pts_p  <- st_transform(pts_sf, proj_crs)
xy     <- st_coordinates(pts_p)

d_seg_kept <- d_seg_kept %>%
  mutate(
    x = as.numeric(xy[, 1]),
    y = as.numeric(xy[, 2])
  ) %>%
  filter(is.finite(x), is.finite(y))

xmin <- min(d_seg_kept$x)
xmax <- max(d_seg_kept$x)
ymin <- min(d_seg_kept$y)
ymax <- max(d_seg_kept$y)

e <- extent(
  xmin - extent_buffer_m,
  xmax + extent_buffer_m,
  ymin - extent_buffer_m,
  ymax + extent_buffer_m
)

base_r <- raster(e, res = cellsize_m, crs = CRS(proj_crs))
base_r <- raster::setValues(base_r, 1)
base_r <- write_tmp(base_r)

stopifnot(raster::hasValues(base_r))
stopifnot(any(is.finite(raster::getValues(base_r))))

cat('Fitting dBMM...\n')

location_error_m <- dbmm_config$location_error_m
window_size <- dbmm_config$window_size
margin <- dbmm_config$margin
time_step_mins <- dbmm_config$time_step_mins

fit_dbbmm_segment_raster <- function(seg_df, base_raster) {
  mv <- move::move(
    x = seg_df$x,
    y = seg_df$y,
    time = seg_df$date,
    data = seg_df,
    proj = sp::CRS(proj_crs),
    animal = as.character(seg_df$id[1])
  )
  
  bb <- move::brownian.bridge.dyn(
    object = mv,
    location.error = location_error_m,
    window.size = window_size,
    margin = margin,
    raster = base_raster,
    time.step = time_step_mins,
    verbose = FALSE
  )
  
  methods::as(bb, "RasterLayer")
}

seg_ids <- d_seg_kept %>%
  distinct(seg_id) %>%
  arrange(seg_id) %>%
  pull(seg_id)

ud_by_seg <- setNames(vector("list", length(seg_ids)), seg_ids)
fit_fail <- list()

for (sid in seg_ids) {
  seg_df <- d_seg_kept %>%
    filter(seg_id == sid) %>%
    arrange(date)
  
  r_seg <- tryCatch(
    fit_dbbmm_segment_raster(seg_df, base_r),
    error = function(e) e
  )
  
  if (inherits(r_seg, "error")) {
    fit_fail[[length(fit_fail) + 1]] <- list(
      seg_id = sid,
      id = seg_df$id[1],
      msg = conditionMessage(r_seg)
    )
    next
  }
  
  if (!raster::hasValues(r_seg)) {
    fit_fail[[length(fit_fail) + 1]] <- list(
      seg_id = sid, id = seg_df$id[1],
      msg = "UD raster has no values (did you extract topology instead of values?)"
    )
    next
  }
  
  s <- raster::cellStats(r_seg, stat = "sum", na.rm = TRUE)
  if (!is.finite(s) || s <= 0) {
    fit_fail[[length(fit_fail) + 1]] <- list(
      seg_id = sid, id = seg_df$id[1],
      msg = "UD raster sum is 0 or non-finite (segment too short for window/margin or no usable steps)"
    )
    next
  }
  
  ud_by_seg[[sid]] <- r_seg
}

ok_seg <- names(ud_by_seg)[!vapply(ud_by_seg, is.null, logical(1))]
ud_by_seg <- ud_by_seg[ok_seg]
ud_by_seg <- lapply(ud_by_seg, write_tmp)

cat('Creating global UD...\n')

sum_rasters_na0 <- function(a, b) {
  raster::overlay(
    a, b,
    fun = function(x, y) {
      x[is.na(x)] <- 0
      y[is.na(y)] <- 0
      x + y
    },
    filename = raster::rasterTmpFile(),
    overwrite = TRUE
  )
}

empty_raster <- function(template_r) {
  r <- raster::raster(template_r)
  r <- raster::setValues(r, 0)
  write_tmp(r)
}

normalize_ud <- function(r) {
  s <- raster::cellStats(r, stat = "sum", na.rm = TRUE)
  if (!is.finite(s) || s <= 0) return(r)
  write_tmp(r / s)
}

seg_info <- d_seg_kept %>%
  distinct(seg_id, id) %>%
  filter(seg_id %in% names(ud_by_seg)) %>%
  arrange(id, seg_id)

ids_ok <- seg_info %>%
  distinct(id) %>%
  pull(id)

ud_by_id <- setNames(vector("list", length(ids_ok)), ids_ok)

for (pid in ids_ok) {
  segs <- seg_info %>%
    filter(id == pid) %>%
    pull(seg_id)
  
  r_id <- empty_raster(base_r)
  
  for (sid in segs) {
    r_id <- sum_rasters_na0(r_id, ud_by_seg[[sid]])
  }
  
  ud_by_id[[pid]] <- normalize_ud(r_id)
}

ud_global <- empty_raster(base_r)

for (pid in names(ud_by_id)) {
  ud_global <- sum_rasters_na0(ud_global, ud_by_id[[pid]])
}

ud_global <- normalize_ud(ud_global)

cat('Creating corridor polygons...\n')

ud_to_volume <- function(ud) {
  v <- raster::getValues(ud)
  ok <- which(!is.na(v) & is.finite(v) & v > 0)
  
  vol <- rep(NA_real_, length(v))
  if (length(ok) == 0) {
    out <- ud
    raster::values(out) <- vol
    return(out)
  }
  
  vv <- v[ok]
  o <- order(vv, decreasing = TRUE)
  cs <- cumsum(vv[o])
  
  vol[ok[o]] <- cs
  
  out <- ud
  raster::values(out) <- vol
  out
}

mask_to_sf_polygons <- function(mask_r) {
  ok <- !is.na(raster::getValues(mask_r)) & is.finite(raster::getValues(mask_r))
  if (!any(ok)) {
    return(sf::st_sf(geometry = sf::st_sfc(crs = sf::st_crs(as.character(raster::crs(mask_r))))))
  }
  
  p <- raster::rasterToPolygons(mask_r, fun = function(x) !is.na(x), dissolve = TRUE)
  s <- sf::st_as_sf(p)
  
  if (is.na(sf::st_crs(s))) {
    sf::st_crs(s) <- sf::st_crs(as.character(raster::crs(mask_r)))
  }
  
  s
}

# 1) volume surface
vol_global <- ud_to_volume(ud_global)

# 2) masks for 95% and 50%
mask95 <- vol_global
mask50 <- vol_global

v <- raster::getValues(vol_global)

v95 <- v
v50 <- v

v95[!is.na(v95) & is.finite(v95) & v95 > 0.95] <- NA_real_
v50[!is.na(v50) & is.finite(v50) & v50 > 0.50] <- NA_real_

raster::values(mask95) <- v95
raster::values(mask50) <- v50

# 3) raster -> polygons in projected CRS
poly95 <- mask_to_sf_polygons(mask95)
poly50 <- mask_to_sf_polygons(mask50)

# 4) project back to lon/lat for plotting/export
poly95_ll <- sf::st_transform(poly95, 4326)
poly50_ll <- sf::st_transform(poly50, 4326)

make_file_path <- function(path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  file.path(path)
}

out_html  <- make_file_path(dbmm_config$out_html)
out_gj95  <- make_file_path(dbmm_config$out_gj95)
out_gj50  <- make_file_path(dbmm_config$out_gj50)

max_points_plot <- dbmm_config$max_points_plot

poly95_ll <- sf::st_make_valid(poly95_ll)
poly50_ll <- sf::st_make_valid(poly50_ll)

poly95_ll <- sf::st_cast(poly95_ll, "MULTIPOLYGON", warn = FALSE)
poly50_ll <- sf::st_cast(poly50_ll, "MULTIPOLYGON", warn = FALSE)

poly95_ll <- sf::st_sf(geometry = sf::st_union(poly95_ll))
poly50_ll <- sf::st_sf(geometry = sf::st_union(poly50_ll))

sf::st_write(poly95_ll, out_gj95, driver = "GeoJSON", delete_dsn = TRUE, quiet = TRUE)
sf::st_write(poly50_ll, out_gj50, driver = "GeoJSON", delete_dsn = TRUE, quiet = TRUE)

estimate_zoom <- function(lat, lon) {
  lat <- lat[is.finite(lat)]
  lon <- lon[is.finite(lon)]
  if (length(lat) == 0 || length(lon) == 0) return(2)
  rng <- max(diff(range(lat)), diff(range(lon)))
  if (!is.finite(rng) || rng <= 0) return(8)
  z <- log2(360 / (rng * 1.5))
  max(1, min(15, z))
}

sf_polygon_to_paths <- function(sfc) {
  sfc <- sf::st_cast(sfc, "MULTIPOLYGON", warn = FALSE)
  coords <- sf::st_coordinates(sfc)
  if (nrow(coords) == 0) return(data.frame(lon = numeric(), lat = numeric(), grp = character()))
  
  df <- data.frame(
    lon = coords[, "X"],
    lat = coords[, "Y"],
    grp = paste0("g", coords[, "L1"], "_", coords[, "L2"], "_", coords[, "L3"])
  )
  
  parts <- split(df, df$grp)
  out <- do.call(rbind, lapply(seq_along(parts), function(i) {
    di <- parts[[i]]
    di$grp <- names(parts)[i]
    rbind(di, data.frame(lon = NA_real_, lat = NA_real_, grp = names(parts)[i]))
  }))
  rownames(out) <- NULL
  out
}

points_plot <- d_seg_kept %>% select(lon, lat)
if (nrow(points_plot) > max_points_plot) {
  set.seed(0)
  points_plot <- points_plot[sample.int(nrow(points_plot), max_points_plot), ]
}

p95 <- sf_polygon_to_paths(sf::st_geometry(poly95_ll))
p50 <- sf_polygon_to_paths(sf::st_geometry(poly50_ll))

all_lat <- c(points_plot$lat, p95$lat, p50$lat)
all_lon <- c(points_plot$lon, p95$lon, p50$lon)
all_lat <- all_lat[is.finite(all_lat)]
all_lon <- all_lon[is.finite(all_lon)]

center <- list(lat = mean(all_lat), lon = mean(all_lon))
zoom <- estimate_zoom(all_lat, all_lon)

fig <- plot_ly(type = "scattermapbox")

if (nrow(p95) > 0) {
  fig <- fig %>% add_trace(
    data = p95,
    lat = ~lat, lon = ~lon,
    type = "scattermapbox",
    mode = "lines",
    fill = "toself",
    name = "Corridor 95%",
    line = list(width = 2),
    opacity = 0.20,
    hoverinfo = "skip"
  )
}

if (nrow(p50) > 0) {
  fig <- fig %>% add_trace(
    data = p50,
    lat = ~lat, lon = ~lon,
    type = "scattermapbox",
    mode = "lines",
    fill = "toself",
    name = "Core 50%",
    line = list(width = 2),
    opacity = 0.30,
    hoverinfo = "skip"
  )
}

if (nrow(points_plot) > 0) {
  fig <- fig %>% add_trace(
    data = points_plot,
    lat = ~lat, lon = ~lon,
    type = "scattermapbox",
    mode = "markers",
    name = "Fixes (sampled)",
    marker = list(size = 4, opacity = 0.35, color = "#111111"),
    hoverinfo = "skip"
  )
}

tile_url <- "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
fig <- fig %>% layout(
  template = "plotly_white",
  title = "dBBMM corridors (50% core, 95% corridor)",
  height = 780,
  margin = list(l = 10, r = 10, t = 60, b = 10),
  legend = list(orientation = "h", yanchor = "bottom", y = 0.01, xanchor = "left", x = 0.01),
  mapbox = list(
    style = "white-bg",
    center = center,
    zoom = zoom,
    layers = list(
      list(
        sourcetype = "raster",
        source = list(tile_url),
        below = "traces",
        opacity = 1,
        sourceattribution = "Esri World Imagery"
      )
    )
  )
)

out_dir <- dirname(out_html)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

old <- getwd()
setwd(out_dir)

htmlwidgets::saveWidget(fig, basename(out_html), selfcontained = TRUE)

setwd(old)

cat("Saved:\n")
cat("  HTML  :", out_html, "\n")
cat("  Geo95 :", out_gj95, "\n")
cat("  Geo50 :", out_gj50, "\n")

if (dir.exists(r_tmp)) {
  n_files <- length(list.files(r_tmp, recursive = TRUE, all.files = TRUE, no.. = TRUE))
  unlink(r_tmp, recursive = TRUE, force = TRUE)
  cat("Removed tmp raster dir:", r_tmp, " (files:", n_files, ")\n")
}

