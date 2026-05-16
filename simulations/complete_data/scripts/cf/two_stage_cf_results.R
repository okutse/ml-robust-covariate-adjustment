# 2. Cross-fitted two-stage covariate adjustment procedure of Cohen and Fogarty (2023) with per-replicate checkpointing and diagnostics.

# Required packages for missing-outcome workflows (models, CF helpers, and parallel orchestration).
required_packages <- c(
  "parsnip","ranger", "dbarts", "SuperLearner", "xgboost", "magrittr", "purrr",
  "rsample", "dplyr", "foreach", "doParallel", "nnls", "gam", "data.table",
  "doSNOW", "renv", "parallel", "magrittr", "parsnip", "dbarts", "SuperLearner", "xgboost", "jsonlite"
) 
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  renv_lock <- file.path(getwd(), "renv.lock")
  if (requireNamespace("renv", quietly = TRUE) && file.exists(renv_lock)) {
    renv::install(missing_packages, prompt = FALSE)
  } else {
    install.packages(missing_packages, repos = "https://cloud.r-project.org")
  }
}

invisible(lapply(required_packages, function(pkg) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}))

# source helper functions for the two-stage CF procedure of Cohen and Fogarty (2023).
source("simulations/complete_data/scripts/cf/cf_two_stage_helpers.R")
source(file.path("helpers", "data_source_helpers.R"))

# Cross-fitted two-stage simulation (one file per scenario)
procedure_name <- "two_stage_cf"

# Keep bootstrap settings explicit for reproducibility and easy audit in logs.
bootstrap_reps <- BOOTSTRAP_REPS_DEFAULT
bootstrap_seed <- 20260417

# Allow dynamic control over how many replicates are processed per scenario.
replicates_to_run <- as.integer(Sys.getenv("REPLICATES_TO_RUN", REPLICATES_DEFAULT))

# Default to parallel execution; can be disabled by setting USE_PARALLEL=false.
use_parallel <- tolower(Sys.getenv("USE_PARALLEL", "true")) == "true"

load_data <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  list(datasets = env$datasets, metadata = as.data.frame(env$metadata))
}

# Inputs
input_dir <- "simulations/complete_data/datasets"
local_datasets_dir <- input_dir
output_dir <- file.path("simulations/complete_data/complete_data_results", procedure_name)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

log_dir <- file.path("logs", "complete_data", procedure_name)
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(log_dir, sprintf("%s_%s.log", procedure_name, format(Sys.time(), "%Y%m%d_%H%M%S")))
log_file <- normalizePath(log_file, mustWork = FALSE)
cat("Logging to:", log_file, "\n")
# Mirror console output to the log file for reproducibility and auditability.
sink(log_file, append = TRUE, split = TRUE)
on.exit(sink(), add = TRUE)

scenario_checkpoint <- file.path(output_dir, "scenario_checkpoint.csv")
manifest_file <- file.path(output_dir, "input_manifest.csv")

# Keep the data-source switches explicit so users can choose local, archive, or regenerate runs.
data_source <- normalize_data_source(get_runtime_option(c("data_source", "DATA_SOURCE"), "local"), default = "local")
archive_dir <- get_runtime_option(c("archive_datasets_dir", "ARCHIVE_DATASETS_DIR"), "")
sync_archive <- normalize_data_source(get_runtime_option(c("sync_archive_to_local", "SYNC_ARCHIVE_TO_LOCAL"), "false"), default = "false") == "true"
zenodo_url <- get_runtime_option(c("zenodo_data_url", "ZENODO_DATA_URL", "ZENODO_URL"), "")
zenodo_doi <- get_runtime_option(c("zenodo_data_doi", "ZENODO_DATA_DOI", "ZENODO_DOI"), "")

resolve_zenodo_url <- function(doi, url) {
  reference <- if (nzchar(url)) url else doi
  resolve_zenodo_download_url(reference, preferred_file = "Data.zip")
}

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

find_archive_dir <- function(base_dir) {
  find_archive_root(
    base_dir = base_dir,
    file_pattern = "^complete_n\\d+_r2_\\d+p\\d+\\.RData$",
    parent_levels = 0
  )
}

if (data_source == "regenerate") {
  input_dir <- local_datasets_dir
  # Regenerating complete-data scenarios keeps the same seed-controlled generator behavior.
  source(file.path("simulations", "complete_data", "scripts", "generate_complete_data.R"))
} else if (data_source %in% c("archive", "local")) {
  local_available <- local_dataset_available(local_datasets_dir, "^complete_n\\d+_r2_\\d+p\\d+\\.RData$")
  if (data_source == "local" && local_available) {
    input_dir <- local_datasets_dir
  } else {
    # Prefer an explicitly supplied local archive folder; otherwise fall back to the Zenodo DOI/URL.
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

merge_replicate_checkpoint <- function(existing, incoming) {
  # Keep the latest status row per replicate when resuming.
  if (is.null(existing) || nrow(existing) == 0) {
    return(incoming)
  }
  combined <- dplyr::bind_rows(existing, incoming)
  combined <- combined %>%
    dplyr::group_by(replicate) %>%
    dplyr::slice_tail(n = 1) %>%
    dplyr::ungroup()
  combined
}

merge_replicate_diagnostics <- function(existing, incoming) {
  # De-duplicate by replicate, preferring the most recent diagnostics.
  if (is.null(existing) || nrow(existing) == 0) {
    return(incoming)
  }
  combined <- dplyr::bind_rows(existing, incoming)
  combined <- combined %>%
    dplyr::group_by(replicate) %>%
    dplyr::slice_tail(n = 1) %>%
    dplyr::ungroup()
  combined
}

read_cached_replicate_diagnostics <- function(method_progress_dir) {
  if (is.null(method_progress_dir) || !dir.exists(method_progress_dir)) {
    return(NULL)
  }
  files <- list.files(method_progress_dir, pattern = "^replicate_\\d+\\.csv$", full.names = TRUE)
  if (length(files) == 0) {
    return(NULL)
  }
  dplyr::bind_rows(lapply(files, read.csv, stringsAsFactors = FALSE))
}

recompute_relative_efficiency <- function(results) {
  # Recompute relative efficiency from Monte Carlo variance in per-scenario aggregates.
  if (!is.data.frame(results) || nrow(results) == 0) {
    return(results)
  }
  if (!"relative_efficiency" %in% names(results)) {
    results$relative_efficiency <- NA_real_
  }
  var_col <- if ("mc_var_estimate" %in% names(results)) {
    "mc_var_estimate"
  } else if ("var_estimate" %in% names(results)) {
    "var_estimate"
  } else {
    return(results)
  }
  lm_var <- results[results$Estimator == "lm", var_col]
  if (length(lm_var) == 1 && !is.na(lm_var)) {
    results$relative_efficiency <- ifelse(
      results[[var_col]] > 0,
      lm_var / results[[var_col]],
      NA_real_
    )
  }
  results
}

run_single_file <- function(file) {
  scenario_start <- Sys.time()
  ld <- load_data(file)
  scenario_name <- sub("\\.RData$", "", basename(file))
  scenario_dir <- file.path(output_dir, scenario_name)
  dir.create(scenario_dir, recursive = TRUE, showWarnings = FALSE)
  scenario_progress_dir <- normalizePath(file.path(scenario_dir, "replicate_cache"), mustWork = FALSE)

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

    method_rep_checkpoint <- file.path(scenario_dir, paste0(method_name, "_replicate_checkpoint.csv"))
    method_diag_file <- file.path(scenario_dir, paste0(method_name, "_replicate_diagnostics.csv"))
    method_file <- file.path(scenario_dir, paste0(method_name, "_cf.csv"))

    cached_diag <- read_cached_replicate_diagnostics(file.path(scenario_progress_dir, method_name))
    if (!is.null(cached_diag)) {
      diag_existing <- NULL
      if (file.exists(method_diag_file)) {
        diag_existing <- read.csv(method_diag_file, stringsAsFactors = FALSE)
      }
      diag_all_cached <- merge_replicate_diagnostics(diag_existing, cached_diag)
      write.csv(diag_all_cached, method_diag_file, row.names = FALSE)
    }

    completed_reps <- integer(0)
    rep_checkpoint <- NULL
    # Consolidate per-replicate checkpoint files written by parallel workers
    rep_checkpoint <- consolidate_replicate_checkpoints(file.path(scenario_progress_dir, method_name))
    if (!is.null(rep_checkpoint)) {
      completed_reps <- rep_checkpoint$replicate[rep_checkpoint$status == "done"]
    }
    if (!is.null(cached_diag) && "replicate" %in% names(cached_diag)) {
      cached_reps_unique <- unique(cached_diag$replicate)
      completed_reps <- unique(c(completed_reps, cached_reps_unique))
    }
    replicate_ids <- seq_len(reps_to_run)
    pending_reps <- setdiff(replicate_ids, completed_reps)
    if (length(pending_reps) == 0) {
      # All replicates reported done. Ensure method-level aggregates exist; if not,
      # attempt to reconstruct them from per-replicate cache before marking done.
      method_cache_dir <- file.path(scenario_progress_dir, method_name)
      need_aggregation <- !(file.exists(method_diag_file) && file.exists(method_file))
      if (need_aggregation) {
        cached_rows <- read_cached_replicate_diagnostics(method_cache_dir)
        if (!is.null(cached_rows) && nrow(cached_rows) > 0) {
          diag_existing <- NULL
          if (file.exists(method_diag_file)) {
            diag_existing <- read.csv(method_diag_file, stringsAsFactors = FALSE)
          }
          diag_all_cached <- merge_replicate_diagnostics(diag_existing, cached_rows)
          summary_res <- tryCatch(summarize_two_stage_cf_metrics(diag_all_cached), error = function(e) NULL)
          if (!is.null(summary_res)) {
            diag_all <- summary_res$diagnostics
            write.csv(diag_all, method_diag_file, row.names = FALSE)
            write.csv(summary_res$summary, method_file, row.names = FALSE)
            # Delete per-replicate cache files after successful aggregation
            if (dir.exists(method_cache_dir)) {
              cache_files <- list.files(method_cache_dir, pattern = "^replicate_\\d+\\.csv$", full.names = TRUE)
              if (length(cache_files) > 0) file.remove(cache_files)
            }
          } else {
            err_msg <- paste0("aggregation_failed_for_", method_name)
            method_cp <- rbind(method_cp, data.frame(method = method_name, status = "error", timestamp = as.character(Sys.time()), error = err_msg, stringsAsFactors = FALSE))
            write.csv(method_cp, method_checkpoint, row.names = FALSE)
            stop(err_msg)
          }
        } else {
          err_msg <- paste0("no_cached_replicates_for_", method_name)
          method_cp <- rbind(method_cp, data.frame(method = method_name, status = "error", timestamp = as.character(Sys.time()), error = err_msg, stringsAsFactors = FALSE))
          write.csv(method_cp, method_checkpoint, row.names = FALSE)
          stop(err_msg)
        }
      }

      # Now safe to mark method done and clean up cache directory.
      method_cp <- rbind(
        method_cp,
        data.frame(method = method_name, status = "done", timestamp = as.character(Sys.time()), error = "", stringsAsFactors = FALSE)
      )
      write.csv(method_cp, method_checkpoint, row.names = FALSE)
      if (dir.exists(method_cache_dir)) {
        unlink(method_cache_dir, recursive = TRUE, force = TRUE)
      }
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
        bootstrap_seed = bootstrap_seed,
        replicate_ids = pending_reps,
        progress_dir = scenario_progress_dir,
        log_file = log_file
      )

      diag_new <- attr(res, "replicate_diagnostics")
      diag_existing <- NULL
      if (file.exists(method_diag_file)) {
        diag_existing <- read.csv(method_diag_file, stringsAsFactors = FALSE)
      }
      diag_all <- merge_replicate_diagnostics(diag_existing, diag_new)
      summary_res <- summarize_two_stage_cf_metrics(diag_all)
      diag_all <- summary_res$diagnostics
      # Persist diagnostics before marking checkpoints so resume has data to aggregate.
      write.csv(diag_all, method_diag_file, row.names = FALSE)
      write.csv(summary_res$summary, method_file, row.names = FALSE)

      # Delete per-replicate cache files immediately after aggregation to avoid stale data on resume.
      method_cache_dir <- file.path(scenario_progress_dir, method_name)
      if (dir.exists(method_cache_dir)) {
        cache_files <- list.files(method_cache_dir, pattern = "^replicate_\\d+\\.csv$", full.names = TRUE)
        if (length(cache_files) > 0) {
          file.remove(cache_files)
        }
      }

      # Per-replicate checkpoints are already written atomically by workers; consolidate them for tracking.
      rep_checkpoint <- consolidate_replicate_checkpoints(method_cache_dir)
    }, error = function(e) {
      status <<- "error"
      msg <<- conditionMessage(e)
    })

    method_elapsed <- as.numeric(difftime(Sys.time(), method_start, units = "secs"))
    cat("Running method:", method_name, "completed in", round(method_elapsed, 2), "sec\n")

    if (status == "error") {
      method_cp <- rbind(
        method_cp,
        data.frame(method = method_name, status = status, timestamp = as.character(Sys.time()), error = msg, stringsAsFactors = FALSE)
      )
      write.csv(method_cp, method_checkpoint, row.names = FALSE)
      cat("Method checkpointed:", method_name, "for", scenario_name, "\n")
      stop(msg)
    }

    rep_checkpoint <- consolidate_replicate_checkpoints(file.path(scenario_progress_dir, method_name))
    completed_reps <- if (!is.null(rep_checkpoint)) rep_checkpoint$replicate[rep_checkpoint$status == "done"] else integer(0)
    if (length(completed_reps) == length(replicate_ids)) {
      method_cp <- rbind(
        method_cp,
        data.frame(method = method_name, status = "done", timestamp = as.character(Sys.time()), error = msg, stringsAsFactors = FALSE)
      )
      write.csv(method_cp, method_checkpoint, row.names = FALSE)
      method_cache_dir <- file.path(scenario_progress_dir, method_name)
      if (dir.exists(method_cache_dir)) {
        # Clean up per-replicate checkpoint files and remaining cache files
        unlink(method_cache_dir, recursive = TRUE, force = TRUE)
      }
      cat("Method checkpointed:", method_name, "for", scenario_name, "\n")
    }
  }

  # Aggregate method-level files into a scenario-level summary after all methods complete.
  method_files <- list.files(scenario_dir, pattern = "_cf\\.csv$", full.names = TRUE)
  scenario_file <- file.path(scenario_dir, paste0(scenario_name, "_results.csv"))
  if (length(method_files) > 0) {
    scenario_agg <- lapply(method_files, read.csv, stringsAsFactors = FALSE) %>% dplyr::bind_rows()
    scenario_agg <- recompute_relative_efficiency(scenario_agg)
    write.csv(scenario_agg, scenario_file, row.names = FALSE)
  } else if (file.exists(scenario_file)) {
    scenario_agg <- read.csv(scenario_file, stringsAsFactors = FALSE)
    scenario_agg <- recompute_relative_efficiency(scenario_agg)
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
