packages <- c(
  "foreach", "doParallel", "parsnip", "ranger",
  "dbarts", "SuperLearner", "xgboost", "bartMachine"
)

ipak <- function(pkg) {
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) {
    install.packages(new.pkg, dependencies = TRUE, repos = "http://cran.rstudio.com/")
  }
  sapply(pkg, require, character.only = TRUE)
}

ipak(packages)

source(file.path("simulations", "complete_data", "scripts", "two_stage_model_helpers.R"))

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

cat("Running two-stage models on", length(input_files), "scenario files.\n")

input_manifest <- data.frame(
  data_source = data_source,
  input_dir = normalizePath(input_dir, winslash = "/", mustWork = TRUE),
  file = basename(input_files),
  md5 = as.character(tools::md5sum(input_files)),
  stringsAsFactors = FALSE
)
write.csv(
  input_manifest,
  file = file.path(output_dir, "two_stage_input_manifest.csv"),
  row.names = FALSE
)

two_stage_results <- do.call(rbind, lapply(input_files, function(file_path) {
  cat("Processing scenario:", basename(file_path), "\n")
  scenario <- load_scenario(file_path)
  results <- run_two_stage(
    datasets = scenario$datasets,
    metadata = scenario$metadata,
    methods = get_two_stage_registry(),
    use_parallel = TRUE,
    cores = parallel::detectCores(logical = TRUE),
    cl = NULL
  )
  cat("Completed scenario:", basename(file_path), "\n")
  results
}))

write.csv(
  two_stage_results,
  file = file.path(output_dir, "two_stage_results.csv"),
  row.names = FALSE
)

# edit and do prelim runs for two-stage methods in a separate script, then combine results into a single dataframe and save as csv for plotting and comparison with single-stage methods

# Example run (R terminal):
# DATA_SOURCE=archive ARCHIVE_DATASETS_DIR=/path/to/zenodo/datasets \
# Rscript simulations/complete_data/scripts/two_stage_results.R
