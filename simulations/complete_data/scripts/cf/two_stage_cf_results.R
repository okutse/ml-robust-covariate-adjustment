library(data.table)
library(foreach)
library(doParallel)
library(doSNOW)
library(parallel)
library(tidyverse)
library(purrr)
library(magrittr)
library(parsnip)
library(dbarts)
library(rsample)
library(SuperLearner)
library(xgboost)

source("simulations/complete_data/scripts/cf/cf_two_stage_helpers.R")

# Cross-fitted two-stage simulation (one file per scenario)
procedure_name <- "two_stage_cf"

# Keep bootstrap settings explicit for reproducibility and easy audit in logs.
bootstrap_reps <- BOOTSTRAP_REPS_DEFAULT
bootstrap_seed <- 20260417

# Allow dynamic control over how many replicates are processed per scenario.
replicates_to_run <- as.integer(Sys.getenv("REPLICATES_TO_RUN", REPLICATES_DEFAULT))

# Default to sequential execution so per-replicate progress logs stay ordered.
use_parallel <- tolower(Sys.getenv("USE_PARALLEL", "false")) == "true"

load_data <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  list(datasets = env$datasets, metadata = as.data.frame(env$metadata))
}

# Inputs
input_dir <- "simulations/complete_data/datasets"
output_dir <- file.path("simulations/complete_data/complete_data_results", procedure_name)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

log_dir <- file.path("logs", "complete_data", procedure_name)
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(log_dir, sprintf("%s_%s.log", procedure_name, format(Sys.time(), "%Y%m%d_%H%M%S")))
cat("Logging to:", log_file, "\n")
# Mirror console output to the log file for reproducibility and auditability.
sink(log_file, append = TRUE, split = TRUE)
on.exit(sink(), add = TRUE)

scenario_checkpoint <- file.path(output_dir, "scenario_checkpoint.csv")
manifest_file <- file.path(output_dir, "input_manifest.csv")

all_files <- list.files(input_dir, pattern = "^complete_.*\\.RData$", full.names = TRUE)
if (length(all_files) == 0) {
  stop("No input files found in complete_data/datasets")
}

if (file.exists(manifest_file)) {
  file.remove(manifest_file)
}

manifest <- lapply(all_files, function(f) {
  ld <- load_data(f)
  md <- ld$metadata
  data.frame(
    file = basename(f),
    path = f,
    n = md$n[1],
    target_r2 = md$target_r2[1],
    dataset_count = length(ld$datasets),
    stringsAsFactors = FALSE
  )
}) %>% dplyr::bind_rows()
write.csv(manifest, manifest_file, row.names = FALSE)

if (file.exists(scenario_checkpoint)) {
  scenario_cp <- read.csv(scenario_checkpoint, stringsAsFactors = FALSE)
  processed <- scenario_cp$file
} else {
  scenario_cp <- data.frame(file = character(), status = character(), timestamp = character(), error = character(), stringsAsFactors = FALSE)
  processed <- character()
}

remaining <- setdiff(all_files, processed)
cat("Found", length(all_files), "files; remaining:", length(remaining), "\n")

method_registry <- get_two_stage_cf_registry()
method_names <- names(method_registry)

run_single_file <- function(file) {
  scenario_start <- Sys.time()
  ld <- load_data(file)
  scenario_name <- sub("\\.RData$", "", basename(file))
  scenario_dir <- file.path(output_dir, scenario_name)
  dir.create(scenario_dir, recursive = TRUE, showWarnings = FALSE)

  reps_requested <- if (is.na(replicates_to_run)) REPLICATES_DEFAULT else replicates_to_run
  reps_to_run <- min(length(ld$datasets), reps_requested)
  cat(
    "Processing scenario:",
    scenario_name,
    "with",
    length(ld$datasets),
    "datasets; running",
    reps_to_run,
    "replicates\n"
  )

  method_checkpoint <- file.path(scenario_dir, "method_checkpoint.csv")
  if (file.exists(method_checkpoint)) {
    method_cp <- read.csv(method_checkpoint, stringsAsFactors = FALSE)
    completed_methods <- method_cp$method[method_cp$status == "done"]
  } else {
    method_cp <- data.frame(method = character(), status = character(), timestamp = character(), error = character(), stringsAsFactors = FALSE)
    completed_methods <- character()
  }

  for (method_name in method_names) {
    if (method_name %in% completed_methods) {
      cat("Skipping completed method:", method_name, "for", scenario_name, "\n")
      next
    }

    cat("Running method:", method_name, "for", scenario_name, "\n")
    method_start <- Sys.time()
    msg <- ""
    status <- "done"
    tryCatch({
      res <- run_two_stage_cf(
        datasets = ld$datasets,
        metadata = ld$metadata,
        methods = list(method_registry[[method_name]]),
        use_parallel = use_parallel,
        n_reps = reps_to_run,
        bootstrap_reps = bootstrap_reps,
        bootstrap_seed = bootstrap_seed
      )

      # Write method-level results to a dedicated file before checkpointing.
      method_file <- file.path(scenario_dir, paste0(method_name, "_cf.csv"))
      write.csv(res, method_file, row.names = FALSE)

      # Persist replicate-level diagnostics for this method and scenario.
      diag <- attr(res, "replicate_diagnostics")
      if (!is.null(diag) && nrow(diag) > 0) {
        diag_file <- file.path(scenario_dir, paste0(method_name, "_replicate_diagnostics.csv"))
        write.csv(diag, diag_file, row.names = FALSE)
      }
    }, error = function(e) {
      status <<- "error"
      msg <<- conditionMessage(e)
    })

    method_elapsed <- as.numeric(difftime(Sys.time(), method_start, units = "secs"))
    cat("Running method:", method_name, "completed in", round(method_elapsed, 2), "sec\n")

    # Checkpoint after method outputs are saved so resuming is safe.
    method_cp <- rbind(method_cp, data.frame(method = method_name, status = status, timestamp = as.character(Sys.time()), error = msg, stringsAsFactors = FALSE))
    write.csv(method_cp, method_checkpoint, row.names = FALSE)
    cat("Method checkpointed:", method_name, "for", scenario_name, "\n")

    if (status == "error") {
      stop(msg)
    }
  }

  # Aggregate method-level files into a scenario-level summary after all methods complete.
  method_files <- list.files(scenario_dir, pattern = "_cf\\.csv$", full.names = TRUE)
  if (length(method_files) > 0) {
    scenario_agg <- lapply(method_files, read.csv, stringsAsFactors = FALSE) %>% dplyr::bind_rows()
    scenario_file <- file.path(scenario_dir, paste0(scenario_name, "_results.csv"))
    write.csv(scenario_agg, scenario_file, row.names = FALSE)
  }

  scenario_elapsed <- as.numeric(difftime(Sys.time(), scenario_start, units = "secs"))
  cat("Completed scenario:", scenario_name, "in", round(scenario_elapsed, 2), "sec\n")

  list(file = basename(file), status = "done", error = "")
}

for (f in remaining) {
  ts <- as.character(Sys.time())
  msg <- ""
  status <- "done"
  tryCatch({
    run_single_file(f)
  }, error = function(e) {
    status <<- "error"
    msg <<- conditionMessage(e)
    cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "ERROR", basename(f), ":", msg, "\n")
  })

  if (status == "done") {
    scenario_cp <- rbind(scenario_cp, data.frame(file = basename(f), status = status, timestamp = ts, error = msg, stringsAsFactors = FALSE))
    write.csv(scenario_cp, scenario_checkpoint, row.names = FALSE)
  }
}

# Aggregate all scenario-level files
scenario_files <- list.files(output_dir, pattern = "_results\\.csv$", full.names = TRUE, recursive = TRUE)
scenario_files <- setdiff(scenario_files, file.path(output_dir, "two_stage_cf_results.csv"))
if (length(scenario_files) > 0) {
  all_results <- lapply(scenario_files, read.csv, stringsAsFactors = FALSE) %>% dplyr::bind_rows()
  write.csv(all_results, file.path(output_dir, "two_stage_cf_results.csv"), row.names = FALSE)
  cat("Wrote aggregate:", file.path(output_dir, "two_stage_cf_results.csv"), "\n")
}
