# load required packages, helper functions, and set up environment variables for data source and syncing behavior
packages <- c(
  "foreach", "doParallel", "parsnip", "ranger",
  "dbarts", "SuperLearner", "xgboost", "magrittr", "purrr", "tidymodels", "jsonlite"
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
source(file.path("helpers", "data_source_helpers.R"))

# set environment variables to control data source and syncing behavior
data_source <- normalize_data_source(get_runtime_option(c("data_source", "DATA_SOURCE"), "local"), default = "local")
archive_dir <- get_runtime_option(c("archive_datasets_dir", "ARCHIVE_DATASETS_DIR"), "")
sync_archive <- normalize_data_source(get_runtime_option(c("sync_archive_to_local", "SYNC_ARCHIVE_TO_LOCAL"), "false"), default = "false") == "true"
zenodo_url <- get_runtime_option(c("zenodo_data_url", "ZENODO_DATA_URL", "ZENODO_URL"), "")
zenodo_doi <- get_runtime_option(c("zenodo_data_doi", "ZENODO_DATA_DOI", "ZENODO_DOI"), "")

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
  reference <- if (nzchar(url)) url else doi
  resolve_zenodo_download_url(reference, preferred_file = "Data.zip")
}

# function to download and unzip archived datasets from a given URL
download_and_unzip <- function(url, dest_dir) {
  if (url == "") {
    stop("ZENODO data DOI or URL must be set when a local dataset directory is unavailable.")
  }
  if (!dir.exists(dest_dir)) {
    dir.create(dest_dir, recursive = TRUE)
  }
  zip_path <- file.path(dest_dir, "zenodo_datasets.zip")
  if (file.exists(url) && grepl("\\.zip$", url, ignore.case = TRUE)) {
    safe_unzip_archive(url, dest_dir)
  } else {
    download_archive_file(url, zip_path)
    safe_unzip_archive(zip_path, dest_dir)
  }
  dest_dir
}

# function to find the directory containing the archived datasets after unzipping
find_archive_dir <- function(base_dir) {
  find_archive_root(
    base_dir = base_dir,
    file_pattern = "^complete_n\\d+_r2_\\d+p\\d+\\.RData$",
    parent_levels = 0
  )
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
} else if (data_source %in% c("archive", "local")) {
  local_available <- local_dataset_available(local_datasets_dir, "^complete_n\\d+_r2_\\d+p\\d+\\.RData$")
  if (data_source == "local" && local_available) {
    input_dir <- local_datasets_dir
  } else {
    # Prefer a project-local archive folder if one was supplied; otherwise fall back to Zenodo.
    archive_source <- archive_dir
    if (archive_source %in% c("local", "local_datasets")) {
      archive_source <- local_datasets_dir
    } else if (!nzchar(archive_source)) {
      archive_source <- resolve_zenodo_url(doi = if (nzchar(zenodo_url)) zenodo_url else zenodo_doi, url = zenodo_url)
    }

    if (!nzchar(archive_source)) {
      if (local_available) {
        input_dir <- local_datasets_dir
      } else {
        stop("No local complete-data datasets were found and no Zenodo DOI/URL was provided.")
      }
    } else if (dir.exists(archive_source)) {
      input_dir <- find_archive_dir(archive_source)
    } else {
      zenodo_cache <- file.path("simulations", "complete_data", "archives", "zenodo")
      download_and_unzip(archive_source, zenodo_cache)
      input_dir <- find_archive_dir(zenodo_cache)
    }
  }

  if (sync_archive && input_dir != local_datasets_dir) {
    cat("Syncing archived datasets into", local_datasets_dir, "\n")
    copy_matching_files(input_dir, local_datasets_dir, "^complete_n\\d+_r2_\\d+p\\d+\\.RData$")
    input_dir <- local_datasets_dir
  }
} else {
  stop("DATA_SOURCE must be either 'local', 'archive', or 'regenerate'.")
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

merge_replicate_checkpoint <- function(existing, incoming) {
  # Keep the latest status row per replicate when resuming.
  if (is.null(existing) || nrow(existing) == 0) {
    return(incoming)
  }
  combined <- rbind(existing, incoming)
  combined <- combined[!duplicated(combined$replicate, fromLast = TRUE), , drop = FALSE]
  combined
}

merge_replicate_diagnostics <- function(existing, incoming) {
  # De-duplicate by replicate, preferring the most recent diagnostics.
  if (is.null(existing) || nrow(existing) == 0) {
    return(incoming)
  }
  combined <- rbind(existing, incoming)
  combined <- combined[!duplicated(combined$replicate, fromLast = TRUE), , drop = FALSE]
  combined
}

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
  scenario_name <- tools::file_path_sans_ext(basename(file_path))
  scenario_dir <- file.path(scenario_output_dir, scenario_name)
  if (!dir.exists(scenario_dir)) {
    dir.create(scenario_dir, recursive = TRUE)
  }

  method_registry <- get_two_stage_registry()
  method_names <- names(method_registry)
  method_checkpoint <- file.path(scenario_dir, "method_checkpoint.csv")
  if (file.exists(method_checkpoint)) {
    method_cp <- read.csv(method_checkpoint, stringsAsFactors = FALSE)
    completed_methods <- method_cp$method[method_cp$status == "done"]
  } else {
    method_cp <- data.frame(method = character(), status = character(), timestamp = character(), error = character(), stringsAsFactors = FALSE)
    completed_methods <- character()
  }

  timing <- system.time({
    for (method_name in method_names) {
      if (method_name %in% completed_methods) {
        cat("Skipping completed method:", method_name, "for", scenario_name, "\n")
        next
      }

      method_rep_checkpoint <- file.path(scenario_dir, paste0(method_name, "_replicate_checkpoint.csv"))
      method_diag_file <- file.path(scenario_dir, paste0(method_name, "_replicate_diagnostics.csv"))
      method_file <- file.path(scenario_dir, paste0(method_name, ".csv"))

      completed_reps <- integer(0)
      rep_checkpoint <- NULL
      if (file.exists(method_rep_checkpoint)) {
        rep_checkpoint <- read.csv(method_rep_checkpoint, stringsAsFactors = FALSE)
        completed_reps <- rep_checkpoint$replicate[rep_checkpoint$status == "done"]
      }
      replicate_ids <- seq_len(length(scenario$datasets))
      pending_reps <- setdiff(replicate_ids, completed_reps)
      if (length(pending_reps) == 0) {
        method_cp <- rbind(
          method_cp,
          data.frame(method = method_name, status = "done", timestamp = as.character(Sys.time()), error = "", stringsAsFactors = FALSE)
        )
        write.csv(method_cp, method_checkpoint, row.names = FALSE)
        cat("Skipping completed method:", method_name, "for", scenario_name, "\n")
        next
      }

      cat("Running method:", method_name, "for", scenario_name, "\n")
      # Track total wall time per method for the scenario.
      method_start <- Sys.time()
      msg <- ""
      status <- "done"
      tryCatch({
        res <- run_two_stage(
          datasets = scenario$datasets,
          metadata = scenario$metadata,
          methods = list(method_registry[[method_name]]),
          use_parallel = TRUE,
          cores = parallel::detectCores(logical = TRUE),
          cl = NULL,
          replicate_ids = pending_reps
        )

        diag_new <- attr(res, "replicate_diagnostics")
        diag_existing <- NULL
        if (file.exists(method_diag_file)) {
          diag_existing <- read.csv(method_diag_file, stringsAsFactors = FALSE)
        }
        diag_all <- merge_replicate_diagnostics(diag_existing, diag_new)
        summary_res <- summarize_two_stage_metrics(diag_all)
        # Persist diagnostics before marking checkpoints so resume has data to aggregate.
        write.csv(diag_all, method_diag_file, row.names = FALSE)
        write.csv(summary_res, method_file, row.names = FALSE)

        rep_rows <- data.frame(
          replicate = pending_reps,
          status = "done",
          timestamp = as.character(Sys.time()),
          error = "",
          stringsAsFactors = FALSE
        )
        rep_checkpoint <- merge_replicate_checkpoint(rep_checkpoint, rep_rows)
        # Replicate checkpoint advances only after diagnostics + summary are saved.
        write.csv(rep_checkpoint, method_rep_checkpoint, row.names = FALSE)
      }, error = function(e) {
        status <<- "error"
        msg <<- conditionMessage(e)
      })

      if (status == "error") {
        method_cp <- rbind(
          method_cp,
          data.frame(method = method_name, status = status, timestamp = as.character(Sys.time()), error = msg, stringsAsFactors = FALSE)
        )
        write.csv(method_cp, method_checkpoint, row.names = FALSE)
        stop(msg)
      }

      method_elapsed <- as.numeric(difftime(Sys.time(), method_start, units = "secs"))
      cat("Running method:", method_name, "completed in", round(method_elapsed, 2), "sec\n")

      rep_checkpoint <- read.csv(method_rep_checkpoint, stringsAsFactors = FALSE)
      completed_reps <- rep_checkpoint$replicate[rep_checkpoint$status == "done"]
      if (length(completed_reps) == length(replicate_ids)) {
        method_cp <- rbind(
          method_cp,
          data.frame(method = method_name, status = "done", timestamp = as.character(Sys.time()), error = msg, stringsAsFactors = FALSE)
        )
        write.csv(method_cp, method_checkpoint, row.names = FALSE)
        cat("Method checkpointed:", method_name, "for", scenario_name, "\n")
      }
    }
  })

  cat("Completed scenario:", basename(file_path), "in", round(timing[["elapsed"]], 2), "seconds.\n")

  method_files <- file.path(scenario_dir, paste0(method_names, ".csv"))
  method_files <- method_files[file.exists(method_files)]
  scenario_csv <- file.path(scenario_output_dir, paste0(scenario_name, ".csv"))
  if (length(method_files) > 0) {
    scenario_results <- do.call(rbind, lapply(method_files, read.csv, stringsAsFactors = FALSE))
    scenario_results$n <- scenario$metadata$n[1]
    scenario_results$target_r2 <- scenario$metadata$target_r2[1]
    scenario_results$achieved_r2_raw <- mean(scenario$metadata$achieved_r2_raw)
    scenario_results$ss <- mean(scenario$metadata$ss)
    # Relative efficiency uses the Monte Carlo variance relative to lm.
    lm_var <- scenario_results$var_estimate[scenario_results$Estimator == "lm"]
    scenario_results$relative_efficiency <- ifelse(
      scenario_results$var_estimate > 0,
      lm_var / scenario_results$var_estimate,
      NA
    )
    write.csv(scenario_results, file = scenario_csv, row.names = FALSE)
  }

  write.table(
    data.frame(file = basename(file_path), stringsAsFactors = FALSE),
    file = checkpoint_path,
    append = TRUE,
    sep = ",",
    col.names = !file.exists(checkpoint_path),
    row.names = FALSE
  )
  cat("Checkpointed scenario:", basename(file_path), "\n")
  if (length(method_files) > 0) {
    return(scenario_results)
  }
  data.frame()
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
# Rscript simulations/complete_data/scripts/non_cf/two_stage_results.R
#
# Use an archive if local datasets are absent:
# zenodo_data_doi="https://doi.org/10.5281/zenodo.19393092" \
# Rscript simulations/complete_data/scripts/non_cf/two_stage_results.R
#
# Use a project-local archive copy if you already unpacked Data.zip:
# DATA_SOURCE=archive ARCHIVE_DATASETS_DIR=simulations/complete_data/datasets \
# Rscript simulations/complete_data/scripts/non_cf/two_stage_results.R
