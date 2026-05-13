# load required packages, helper functions, and set up environment variables for data source and syncing behavior
packages <- c(
  "foreach", "doParallel", "parsnip", "ranger",
  "dbarts", "SuperLearner", "xgboost", "magrittr", "purrr", "tidymodels"
)
ipak <- function(pkg) {
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) {
    install.packages(new.pkg, dependencies = TRUE, repos = "http://cran.rstudio.com/")
  }
  sapply(pkg, require, character.only = TRUE)
}
ipak(packages)


source(file.path("simulations", "complete_data", "scripts", "non_cf", "two_stage_model_helpers.R"))

# set environment variables to control data source and syncing behavior
data_source <- tolower(Sys.getenv("DATA_SOURCE", "archive"))
archive_dir <- Sys.getenv("ARCHIVE_DATASETS_DIR", "")
sync_archive <- tolower(Sys.getenv("SYNC_ARCHIVE_TO_LOCAL", "false")) == "true"
zenodo_url <- Sys.getenv("ZENODO_URL", "")
zenodo_doi <- Sys.getenv("ZENODO_DOI", "")

local_datasets_dir <- file.path("simulations", "complete_data", "datasets")
input_dir <- local_datasets_dir
output_dir <- file.path("simulations", "complete_data", "complete_data_results")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
procedure_name <- "two_stage"
scenario_output_dir <- file.path(output_dir, procedure_name)
if (!dir.exists(scenario_output_dir)) {
  dir.create(scenario_output_dir, recursive = TRUE)
}
log_dir <- file.path("logs", "complete_data", procedure_name)
if (!dir.exists(log_dir)) {
  dir.create(log_dir, recursive = TRUE)
}
log_path <- file.path(log_dir, paste0(procedure_name, "_logs.txt"))

## capture console output in a log file while keeping it in the console
sink(log_path, append = TRUE, split = TRUE)
on.exit(sink(), add = TRUE)

# function to resolve the Zenodo URL from either a direct URL or a DOI
resolve_zenodo_url <- function(doi, url) {
  if (url != "") {
    return(url)
  }
  if (doi == "") {
    return("")
  }
  doi <- sub("^https?://(dx\\.)?doi\\.org/", "", doi)
  doi <- sub("^doi:", "", doi)
  zenodo_id <- sub("^10\\.5281/zenodo\\.", "", doi)
  paste0("https://zenodo.org/record/", zenodo_id, "/files/complete_data_datasets.zip?download=1")
}

# function to download and unzip archived datasets from a given URL
download_and_unzip <- function(url, dest_dir) {
  if (url == "") {
    stop("ZENODO_URL or ZENODO_DOI must be set when ARCHIVE_DATASETS_DIR is empty.")
  }
  if (!dir.exists(dest_dir)) {
    dir.create(dest_dir, recursive = TRUE)
  }
  zip_path <- file.path(dest_dir, "zenodo_datasets.zip")
  utils::download.file(url, zip_path, mode = "wb", quiet = FALSE)
  utils::unzip(zip_path, exdir = dest_dir)
  dest_dir
}

# function to find the directory containing the archived datasets after unzipping
find_archive_dir <- function(base_dir) {
  archive_files <- list.files(
    base_dir,
    pattern = "^complete_n\\d+_r2_\\d+p\\d+\\.RData$",
    full.names = TRUE,
    recursive = TRUE
  )
  if (length(archive_files) == 0) {
    stop(paste("No scenario files found after download in", base_dir))
  }
  dirname(archive_files[1])
}

# main execution flow to load datasets, run two-stage models, and save results
if (data_source == "regenerate") {
  ## regenerate datasets using the controlled seeds in the generator
  existing_files <- list.files(
    local_datasets_dir,
    pattern = "^complete_n\\d+_r2_\\d+p\\d+\\.RData$",
    full.names = TRUE
  )
  if (length(existing_files) > 0) {
    warning("Regenerating datasets into existing folder: ", local_datasets_dir)
  }
  source(file.path("simulations", "complete_data", "scripts", "generate_complete_data.R"))
} else if (data_source == "archive") {
  # allow ARCHIVE_DATASETS_DIR=local to use the repo datasets folder directly
  if (archive_dir %in% c("local", "local_datasets")) {
    archive_dir <- local_datasets_dir
  }
  if (archive_dir == "") {
    zenodo_cache <- file.path("simulations", "complete_data", "archives", "zenodo")
    archive_url <- resolve_zenodo_url(zenodo_doi, zenodo_url)
    download_and_unzip(archive_url, zenodo_cache)
    archive_dir <- find_archive_dir(zenodo_cache)
  }
  input_dir <- archive_dir
  if (sync_archive) {
    cat("Syncing archived datasets into", local_datasets_dir, "\n")
    if (!dir.exists(local_datasets_dir)) {
      dir.create(local_datasets_dir, recursive = TRUE)
    }
    archive_files <- list.files(
      archive_dir,
      pattern = "^complete_n\\d+_r2_\\d+p\\d+\\.RData$",
      full.names = TRUE
    )
    file.copy(archive_files, local_datasets_dir, overwrite = TRUE)
    input_dir <- local_datasets_dir
  }
} else {
  stop("DATA_SOURCE must be either 'archive' or 'regenerate'.")
}

# -- main execution flow to load datasets, run two-stage models, and save results --
load_scenario <- function(file_path) {
  env <- new.env(parent = emptyenv())
  load(file_path, envir = env)
  if (!exists("datasets", envir = env) || !exists("metadata", envir = env)) {
    stop(paste("Missing datasets or metadata in", file_path))
  }
  list(
    datasets = get("datasets", envir = env),
    metadata = get("metadata", envir = env)
  )
}

input_files <- list.files(
  input_dir,
  pattern = "^complete_n\\d+_r2_\\d+p\\d+\\.RData$",
  full.names = TRUE
)
if (length(input_files) == 0) {
  stop(paste("No scenario files found in", input_dir))
}

cat("Using datasets from:", input_dir, "\n")

# checkpoint to resume processing after interruption
checkpoint_path <- file.path(scenario_output_dir, paste0(procedure_name, "_checkpoint.csv"))
reset_checkpoint <- tolower(Sys.getenv("RESET_CHECKPOINT", "false")) %in% c("true", "t", "1")
if (reset_checkpoint && file.exists(checkpoint_path)) {
  file.remove(checkpoint_path)
  cat("RESET_CHECKPOINT enabled: cleared", checkpoint_path, "\n")
}
if (reset_checkpoint && dir.exists(scenario_output_dir)) {
  scenario_csvs <- list.files(scenario_output_dir, pattern = "\\.csv$", full.names = TRUE)
  if (length(scenario_csvs) > 0) {
    file.remove(scenario_csvs)
  }
  cat("RESET_CHECKPOINT enabled: cleared per-scenario outputs in", scenario_output_dir, "\n")
}
processed_files <- character(0)
if (file.exists(checkpoint_path)) {
  processed <- read.csv(checkpoint_path, stringsAsFactors = FALSE)
  processed_files <- processed$file
}
pending_files <- input_files[!basename(input_files) %in% processed_files]

cat("Running two-stage models on", length(pending_files), "scenario files.\n")

input_manifest <- data.frame(
  data_source = data_source,
  input_dir = normalizePath(input_dir, winslash = "/", mustWork = TRUE),
  file = basename(input_files),
  md5 = as.character(tools::md5sum(input_files)),
  stringsAsFactors = FALSE
)
write.csv(
  input_manifest,
  file = file.path(scenario_output_dir, paste0(procedure_name, "_input_manifest.csv")),
  row.names = FALSE
)

two_stage_results <- do.call(rbind, lapply(pending_files, function(file_path) {
  cat("Processing scenario:", basename(file_path), "\n")
  scenario <- load_scenario(file_path)
  timing <- system.time({
    results <- run_two_stage(
      datasets = scenario$datasets,
      metadata = scenario$metadata,
      methods = get_two_stage_registry(),
      use_parallel = TRUE,
      cores = parallel::detectCores(logical = TRUE),
      cl = NULL
    )
  })
  cat("Completed scenario:", basename(file_path), "in", round(timing[["elapsed"]], 2), "seconds.\n")
  # write per-scenario results before checkpointing
  scenario_name <- tools::file_path_sans_ext(basename(file_path))
  scenario_csv <- file.path(scenario_output_dir, paste0(scenario_name, ".csv"))
  write.csv(results, file = scenario_csv, row.names = FALSE)
  write.table(
    data.frame(file = basename(file_path), stringsAsFactors = FALSE),
    file = checkpoint_path,
    append = TRUE,
    sep = ",",
    col.names = !file.exists(checkpoint_path),
    row.names = FALSE
  )
  cat("Checkpointed scenario:", basename(file_path), "\n")
  results
}))

# aggregate all per-scenario results into a single output
scenario_files <- list.files(
  scenario_output_dir,
  pattern = "^complete_n\\d+_r2_\\d+p\\d+\\.csv$",
  full.names = TRUE
)
if (length(scenario_files) > 0) {
  aggregated <- do.call(rbind, lapply(scenario_files, read.csv, stringsAsFactors = FALSE))
  write.csv(
    aggregated,
    file = file.path(scenario_output_dir, paste0(procedure_name, "_results.csv")),
    row.names = FALSE
  )
}


# Example run (R terminal):
# DATA_SOURCE=archive ARCHIVE_DATASETS_DIR=local \
# Rscript simulations/complete_data/scripts/non_cf/two_stage_results.R
# ARCHIVE_DATASETS_DIR=path/to/zenodo_datasets if a path to the unzipped datasets is available locally or url to the zip file is available directly
