# Runner for missing-outcome scenarios across settings

source(file.path("simulations", "missing_outcomes", "miss_scripts", "miss_registry.R"))

assert_renv_active <- function() {
  if (!requireNamespace("renv", quietly = TRUE)) {
    stop("renv is not available in the current R session. Restore the project library before running this analysis.")
  }

  project_root <- tryCatch(renv::project(), error = function(e) "")
  if (!nzchar(project_root)) {
    stop("renv is not active for this session. Run the job from the repository root after restoring renv.")
  }

  project_library <- tryCatch(renv::paths$library(project_root), error = function(e) "")
  if (!nzchar(project_library)) {
    stop("Unable to resolve the renv project library for this session.")
  }

  active_libraries <- normalizePath(.libPaths(), winslash = "/", mustWork = FALSE)
  project_library <- normalizePath(project_library, winslash = "/", mustWork = FALSE)
  if (!project_library %in% active_libraries) {
    stop(
      "renv is not active for this session. The project library is not on .libPaths(). ",
      "Activate renv before running the analysis.")
  }

  invisible(TRUE)
}

required_packages <- c(
  "parsnip",
  "ranger",
  "dbarts",
  "SuperLearner",
  "xgboost",
  "magrittr",
  "purrr",
  "rsample",
  "dplyr",
  "foreach",
  "doParallel",
  "nnls",
  "gam",
  "jsonlite"
) # Required packages for missing-outcome workflows (models, CF helpers, and parallel orchestration).

install_missing_required_packages <- function(required_packages) {
  cat("[env-check] Validating required packages before processing...\n")

  renv_active <- FALSE
  if (requireNamespace("renv", quietly = TRUE)) {
    renv_active <- isTRUE(tryCatch({
      assert_renv_active()
      TRUE
    }, error = function(e) {
      FALSE
    }))
  }

  missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_packages) == 0) {
    cat("[env-check] All required packages are available.\n")
    return(invisible(NULL))
  }

  cat(
    "[env-check] Missing required packages detected: ",
    paste(missing_packages, collapse = ", "),
    "\n",
    sep = ""
  )
  cat("[env-check] Installing only required packages for this submitted job and their dependencies...\n")

  if (renv_active) {
    renv::install(missing_packages, prompt = FALSE)
  } else {
    install.packages(missing_packages, repos = "https://cloud.r-project.org", dependencies = TRUE)
  }

  unresolved <- missing_packages[!vapply(missing_packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(unresolved) > 0) {
    stop(
      "[env-check] Package bootstrap completed with failures. Unresolved packages: ",
      paste(unresolved, collapse = ", "),
      "."
    )
  }

  cat("[env-check] Required packages have been loaded successfully. Proceeding with processing.\n")
}

install_missing_required_packages(required_packages)

invisible(lapply(required_packages, function(pkg) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}))




##-- DATA LOADING AND HELPERS --##

# source the main shared helper for resolving path and loading datasets
source(file.path("helpers", "data_source_helpers.R"))

# Global defaults (override via environment variables as needed).
CF_FOLDS <- as.integer(Sys.getenv("CF_FOLDS", CF_FOLDS))
BOOTSTRAP_REPS_DEFAULT <- as.integer(Sys.getenv("BOOTSTRAP_REPS", 250))
REPLICATES_DEFAULT <- as.integer(Sys.getenv("REPLICATES_TO_RUN", 500))
BOOTSTRAP_SEED_DEFAULT <- as.integer(Sys.getenv("BOOTSTRAP_SEED", 20260417))
TRUE_EFFECT_DEFAULT <- as.numeric(Sys.getenv("TRUE_EFFECT", 50))
NULL_EFFECT_DEFAULT <- as.numeric(Sys.getenv("NULL_EFFECT", 0))
CI_LEVEL_DEFAULT <- as.numeric(Sys.getenv("CI_LEVEL", 0.95))

log_progress_line <- function(log_file, line) {
  cat(line, "\n")
  if (sink.number() > 0) {
    return(invisible(NULL))
  }
  if (!is.null(log_file) && nzchar(log_file)) {
    cat(line, "\n", file = log_file, append = TRUE)
  }
}

safe_read_csv <- function(path, ...) {
  if (is.null(path) || !nzchar(path) || !file.exists(path)) {
    return(NULL)
  }

  size <- suppressWarnings(file.info(path)$size)
  if (is.na(size) || size <= 0) {
    cat("[checkpoint] Skipping empty CSV file:", path, "\n")
    return(NULL)
  }

  out <- tryCatch(
    read.csv(path, stringsAsFactors = FALSE, ...),
    error = function(e) {
      msg <- conditionMessage(e)
      if (grepl("no lines available in input|line 1 did not have", msg, ignore.case = TRUE)) {
        cat("[checkpoint] Skipping unreadable/truncated CSV file:", path, "-", msg, "\n")
        return(NULL)
      }
      stop(e)
    }
  )

  if (is.null(out) || !is.data.frame(out) || nrow(out) == 0) {
    return(NULL)
  }
  out
}

write_replicate_cache <- function(method_dir, rep_row) {
  if (is.null(method_dir) || !nzchar(method_dir)) {
    return(invisible(NULL))
  }
  if (!dir.exists(method_dir)) {
    dir.create(method_dir, recursive = TRUE, showWarnings = FALSE)
  }
  cache_file <- file.path(method_dir, sprintf("replicate_%s.csv", rep_row$replicate))
  write.csv(rep_row, cache_file, row.names = FALSE)

  # Write per-replicate checkpoint atomically to avoid race conditions in parallel execution.
  # Multiple workers write their own checkpoint files without interfering.
  checkpoint_file <- file.path(method_dir, sprintf("replicate_%s_checkpoint.csv", rep_row$replicate))
  checkpoint_row <- data.frame(
    replicate = rep_row$replicate,
    status = "done",
    timestamp = as.character(Sys.time()),
    error = "",
    stringsAsFactors = FALSE
  )
  write.csv(checkpoint_row, checkpoint_file, row.names = FALSE)
  # Update combined checkpoint view for quick resume checks.
  combined <- consolidate_replicate_checkpoints(method_dir)
  if (!is.null(combined) && nrow(combined) > 0) {
    combined_file <- file.path(method_dir, "replicate_checkpoint.csv")
    write.csv(combined, combined_file, row.names = FALSE)
  }
}

consolidate_replicate_checkpoints <- function(method_dir) {
  # Merge per-replicate checkpoint files into a single consolidated checkpoint.
  # Called at read time to avoid race conditions from concurrent writes.
  if (is.null(method_dir) || !nzchar(method_dir) || !dir.exists(method_dir)) {
    return(NULL)
  }
  combined_list <- list()
  combined_file <- file.path(method_dir, "replicate_checkpoint.csv")
  if (file.exists(combined_file)) {
    combined_cp <- safe_read_csv(combined_file)
    if (!is.null(combined_cp)) {
      combined_list[[length(combined_list) + 1]] <- combined_cp
    }
  }
  checkpoint_files <- list.files(method_dir, pattern = "^replicate_\\d+_checkpoint\\.csv$", full.names = TRUE)
  if (length(checkpoint_files) > 0) {
    checkpoint_rows <- lapply(checkpoint_files, safe_read_csv)
    checkpoint_rows <- checkpoint_rows[!vapply(checkpoint_rows, is.null, logical(1))]
    if (length(checkpoint_rows) > 0) {
      combined_list <- c(combined_list, checkpoint_rows)
    }
  }
  if (length(combined_list) == 0) return(NULL)
  combined <- do.call(rbind, combined_list)
  if (nrow(combined) > 0) {
    combined <- combined[!duplicated(combined$replicate, fromLast = TRUE), , drop = FALSE]
    rownames(combined) <- NULL
  }
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
  rows <- lapply(files, safe_read_csv)
  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (length(rows) == 0) {
    return(NULL)
  }
  do.call(rbind, rows)
}

normalize_missing_metadata <- function(datasets, metadata, file_path = NULL) {
  n_datasets <- length(datasets)
  if (nrow(metadata) == n_datasets) {
    return(metadata)
  }

  if ("replicate" %in% names(metadata)) {
    metadata <- metadata[!duplicated(metadata$replicate), , drop = FALSE]
    rownames(metadata) <- NULL
    if (nrow(metadata) == n_datasets) {
      if (!is.null(file_path)) {
        cat("Collapsed metadata rows to one per replicate for", basename(file_path), "\n")
      }
      return(metadata)
    }
  }

  stop(sprintf(
    "Mismatch between datasets and metadata: %d datasets vs %d metadata rows.",
    n_datasets,
    nrow(metadata)
  ))
}

summarize_missing_replicates <- function(results, metadata) {
  # Aggregate replicate-level diagnostics into a single method summary.
  results <- as.data.frame(results)
  if (nrow(results) == 0) {
    stop("No results were produced; check model registry configuration.")
  }
  results$covered_mc_mean_95 <- NA_integer_
  safe_mean <- function(x) {
    if (all(is.na(x))) {
      return(NA_real_)
    }
    mean(x, na.rm = TRUE)
  }

  groups <- unique(results[, c("Estimator", "Procedure", "ModelSpec")])
  summaries <- lapply(seq_len(nrow(groups)), function(idx) {
    g <- groups[idx, ]
    rows <- results$Estimator == g$Estimator &
      results$Procedure == g$Procedure &
      results$ModelSpec == g$ModelSpec

    mc_mean_estimate <- mean(results$estimate[rows])
    results$covered_mc_mean_95[rows] <- as.integer(
      results$ci_lower[rows] <= mc_mean_estimate & mc_mean_estimate <= results$ci_upper[rows]
    )

    mc_sd_estimate <- stats::sd(results$estimate[rows])
    mc_var_estimate <- stats::var(results$estimate[rows])
    mc_se_estimate <- mc_sd_estimate / sqrt(sum(rows))

    data.frame(
      Estimator = g$Estimator,
      Procedure = g$Procedure,
      ModelSpec = g$ModelSpec,
      mean_estimate = mc_mean_estimate,
      mean_bias = mean(results$bias[rows]),
      mean_n_obs = mean(results$n_obs[rows]),
      mc_sd_estimate = mc_sd_estimate,
      mc_var_estimate = mc_var_estimate,
      mc_se_estimate = mc_se_estimate,
      sd_estimate = mc_sd_estimate,
      var_estimate = mc_var_estimate,
      mean_bootstrap_se = safe_mean(results$bootstrap_se[rows]),
      sd_bootstrap_se = stats::sd(results$bootstrap_se[rows], na.rm = TRUE),
      mean_bootstrap_bias = safe_mean(results$bootstrap_bias[rows]),
      mean_eif_se = safe_mean(results$eif_se[rows]),
      mean_eif_var = safe_mean(results$eif_var[rows]),
      coverage_95 = mean(results$covered_true_95[rows]),
      coverage_95_mc_mean = mean(results$covered_mc_mean_95[rows]),
      coverage_95_bc = safe_mean(results$covered_true_95_bc[rows]),
      reject_h0_effect_prob = mean(results$reject_h0_effect[rows]),
      reject_h0_null_prob = mean(results$reject_h0_null[rows]),
      mean_ci_lower = mean(results$ci_lower[rows]),
      mean_ci_upper = mean(results$ci_upper[rows]),
      mean_bc_ci_lower = safe_mean(results$bc_ci_lower[rows]),
      mean_bc_ci_upper = safe_mean(results$bc_ci_upper[rows]),
      mean_bootstrap_time_sec = safe_mean(results$bootstrap_time_sec[rows]),
      mean_replicate_time_sec = mean(results$replicate_time_sec[rows]),
      stringsAsFactors = FALSE
    )
  })

  merged <- do.call(rbind, summaries)
  merged$std_bias <- ifelse(merged$mc_sd_estimate > 0, merged$mean_bias / merged$mc_sd_estimate, NA)

  # Relative efficiency uses the Monte Carlo variance relative to lm.
  lm_var <- merged$mc_var_estimate[merged$Estimator == "lm"]
  merged$relative_efficiency <- ifelse(
    merged$mc_var_estimate > 0 & length(lm_var) == 1 && !is.na(lm_var),
    lm_var / merged$mc_var_estimate,
    NA_real_
  )

  meta <- data.frame(
    n = metadata$n[1],
    target_r2 = metadata$target_r2[1],
    achieved_r2_raw = mean(metadata$achieved_r2_raw),
    ss = mean(metadata$ss),
    setting = metadata$setting[1],
    stringsAsFactors = FALSE
  )

  output <- cbind(meta, merged)
  list(summary = output, diagnostics = results)
}

run_all_models_for_dataset <- function(
  datasets,
  metadata,
  procedure_name,
  model_spec,
  covariate_registry,
  procedure_registry,
  model_registry,
  n_reps = REPLICATES_DEFAULT,
  bootstrap_reps = BOOTSTRAP_REPS_DEFAULT,
  bootstrap_seed = BOOTSTRAP_SEED_DEFAULT,
  cf_folds = CF_FOLDS,
  true_effect = TRUE_EFFECT_DEFAULT,
  null_effect = NULL_EFFECT_DEFAULT,
  ci_level = CI_LEVEL_DEFAULT,
  use_parallel = TRUE,
  cores = NULL,
  replicate_ids = NULL,
  progress_dir = NULL,
  log_file = NULL
) {
  if (length(datasets) == 0) {
    stop("No datasets supplied.")
  }
  metadata <- normalize_missing_metadata(datasets, metadata)

  # Support resuming on a subset of replicates when checkpoints exist.
  if (!is.null(replicate_ids)) {
    if (!is.numeric(replicate_ids) || any(is.na(replicate_ids))) {
      stop("replicate_ids must be a numeric vector of replicate indices.")
    }
    if (any(replicate_ids < 1) || any(replicate_ids > length(datasets))) {
      stop("replicate_ids contain indices outside the available dataset range.")
    }
    datasets <- datasets[replicate_ids]
    metadata <- metadata[replicate_ids, , drop = FALSE]
  } else if (!is.null(n_reps)) {
    if (!is.numeric(n_reps) || length(n_reps) != 1 || is.na(n_reps) || n_reps < 1) {
      stop("n_reps must be a single positive number.")
    }
    n_keep <- min(length(datasets), as.integer(n_reps))
    datasets <- datasets[seq_len(n_keep)]
    metadata <- metadata[seq_len(n_keep), , drop = FALSE]
    replicate_ids <- seq_len(n_keep)
  } else {
    replicate_ids <- seq_len(length(datasets))
  }

  procedure <- procedure_registry[[procedure_name]]
  if (is.null(procedure)) {
    stop(paste("Unknown procedure:", procedure_name))
  }

  if (is.null(cores)) {
    cores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", parallel::detectCores(logical = TRUE)))
  }
  if (is.na(cores) || cores < 1) {
    cores <- 1L
  }

  if (use_parallel) {
    Sys.setenv(OMP_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
    slurm_cpus <- Sys.getenv("SLURM_CPUS_PER_TASK", "")
    workers <- max(1, cores - 1)
    cat(
      "[parallel] Cores available: ", cores,
      ", Registered workers: ", workers,
      ", SLURM_CPUS_PER_TASK: ", if (nzchar(slurm_cpus)) slurm_cpus else "(not set)",
      ", OMP_NUM_THREADS=1, MKL_NUM_THREADS=1\n",
      sep = ""
    )
    doParallel::registerDoParallel(workers)
    `%op%` <- foreach::`%dopar%`
  } else {
    foreach::registerDoSEQ()
    `%op%` <- foreach::`%do%`
  }

  model_names <- names(model_registry)
  cat(
    "Starting",
    procedure$name,
    "for model spec",
    model_spec,
    "with estimators:",
    paste(model_names, collapse = ", "),
    "\n"
  )

  z_crit <- stats::qnorm((1 + ci_level) / 2)

  results <- foreach::foreach(
    i = seq_along(datasets),
    .combine = rbind,
    .packages = c("parsnip", "ranger", "dbarts", "SuperLearner", "xgboost", "magrittr", "purrr", "rsample", "dplyr")
  ) %op% {
    df <- as.data.frame(datasets[[i]])
    df[] <- lapply(df, function(x) if (is.factor(x)) as.numeric(as.character(x)) else x)

    rep_id <- replicate_ids[i]
    per_model <- lapply(model_names, function(model_name) {
      log_line <- sprintf(
        "Running estimator %s for procedure %s and model spec %s on scenario %s of %s",
        model_name,
        procedure$name,
        model_spec,
        rep_id,
        length(datasets)
      )
      if (!use_parallel) {
        cat(log_line, "\n")
      } else {
        log_progress_line(log_file, log_line)
      }

      method_progress_dir <- NULL
      if (!is.null(progress_dir) && nzchar(progress_dir)) {
        method_progress_dir <- file.path(progress_dir, procedure$name, model_spec, model_name)
      }

      model <- model_registry[[model_name]]
      covariates <- resolve_covariates(model_spec, covariate_registry, df)
      if (model_name == "correct_model") {
        covariates <- resolve_covariates("correct", covariate_registry, df)
      }

      # Capture per-replicate timing for sequential runs.
      rep_time_start <- Sys.time()
      res <- procedure$run(
        df = df,
        outcome = "y",
        covariates = covariates,
        model = model,
        folds = cf_folds
      )

      estimate_from_df <- function(dat) {
        out <- procedure$run(
          df = dat,
          outcome = "y",
          covariates = covariates,
          model = model,
          folds = cf_folds
        )
        as.numeric(out$estimate)
      }

      bootstrap_se <- NA_real_
      bootstrap_bias <- NA_real_
      bc_estimate <- NA_real_
      bc_ci_lower <- NA_real_
      bc_ci_upper <- NA_real_
      bootstrap_time_sec <- NA_real_
      se_used <- NA_real_

      if (isTRUE(procedure$use_eif_var)) {
        se_used <- res$eif_se
      } else {
        boot_start <- Sys.time()
        set.seed(bootstrap_seed + rep_id)
        boot_estimates <- vapply(seq_len(bootstrap_reps), function(b) {
          idx <- sample.int(nrow(df), size = nrow(df), replace = TRUE)
          df_boot <- df[idx, , drop = FALSE]
          estimate_from_df(df_boot)
        }, numeric(1))
        bootstrap_time_sec <- as.numeric(difftime(Sys.time(), boot_start, units = "secs"))
        bootstrap_se <- stats::sd(boot_estimates)
        bootstrap_bias <- mean(boot_estimates) - res$estimate
        bc_estimate <- res$estimate - bootstrap_bias
        bc_ci_lower <- bc_estimate - z_crit * bootstrap_se
        bc_ci_upper <- bc_estimate + z_crit * bootstrap_se
        se_used <- bootstrap_se
      }

      ci_lower <- res$estimate - z_crit * se_used
      ci_upper <- res$estimate + z_crit * se_used

      covered_true_95 <- as.integer(ci_lower <= true_effect && true_effect <= ci_upper)
      covered_true_95_bc <- as.integer(bc_ci_lower <= true_effect && true_effect <= bc_ci_upper)
      reject_h0_effect <- as.integer(ci_lower > true_effect || ci_upper < true_effect)
      reject_h0_null <- as.integer(ci_lower > null_effect || ci_upper < null_effect)

      rep_time_sec <- as.numeric(difftime(Sys.time(), rep_time_start, units = "secs"))
      rep_row <- data.frame(
        replicate = rep_id,
        Estimator = model$name,
        Procedure = procedure$name,
        ModelSpec = model_spec,
        n_obs = res$n_obs,
        estimate = res$estimate,
        bias = res$bias,
        eif_var = res$eif_var,
        eif_se = res$eif_se,
        bootstrap_se = bootstrap_se,
        bootstrap_bias = bootstrap_bias,
        bc_estimate = bc_estimate,
        ci_lower = ci_lower,
        ci_upper = ci_upper,
        bc_ci_lower = bc_ci_lower,
        bc_ci_upper = bc_ci_upper,
        covered_true_95 = covered_true_95,
        covered_true_95_bc = covered_true_95_bc,
        reject_h0_effect = reject_h0_effect,
        reject_h0_null = reject_h0_null,
        bootstrap_time_sec = bootstrap_time_sec,
        replicate_time_sec = rep_time_sec,
        stringsAsFactors = FALSE
      )

      write_replicate_cache(method_progress_dir, rep_row)

      if (!use_parallel) {
        cat(
          "Running estimator",
          model_name,
          "replicate",
          rep_id,
          "completed in",
          round(rep_time_sec, 3),
          "sec\n"
        )
      } else {
        log_progress_line(log_file, sprintf(
          "Running estimator %s replicate %s completed in %s sec",
          model_name,
          rep_id,
          round(rep_time_sec, 3)
        ))
      }

      rep_row
    })

    do.call(rbind, per_model)
  }

  summary_res <- summarize_missing_replicates(results, metadata)
  attr(summary_res$summary, "replicate_diagnostics") <- summary_res$diagnostics
  summary_res$summary
}

run_all_methods_for_dataset <- function(
  datasets,
  metadata,
  procedures,
  model_specs,
  covariate_registry,
  procedure_registry,
  model_registry,
  n_reps = REPLICATES_DEFAULT,
  bootstrap_reps = BOOTSTRAP_REPS_DEFAULT,
  bootstrap_seed = BOOTSTRAP_SEED_DEFAULT,
  cf_folds = CF_FOLDS,
  true_effect = TRUE_EFFECT_DEFAULT,
  null_effect = NULL_EFFECT_DEFAULT,
  ci_level = CI_LEVEL_DEFAULT,
  use_parallel = TRUE,
  cores = NULL
) {
  do.call(rbind, lapply(procedures, function(proc_name) {
    do.call(rbind, lapply(model_specs, function(model_spec) {
      run_all_models_for_dataset(
        datasets = datasets,
        metadata = metadata,
        procedure_name = proc_name,
        model_spec = model_spec,
        covariate_registry = covariate_registry,
        procedure_registry = procedure_registry,
        model_registry = model_registry,
        n_reps = n_reps,
        bootstrap_reps = bootstrap_reps,
        bootstrap_seed = bootstrap_seed,
        cf_folds = cf_folds,
        true_effect = true_effect,
        null_effect = null_effect,
        ci_level = ci_level,
        use_parallel = use_parallel,
        cores = cores
      )
    }))
  }))
}

run_missing_outcomes <- function(
  input_root,
  output_file,
  procedures,
  model_specs,
  n_reps = REPLICATES_DEFAULT,
  bootstrap_reps = BOOTSTRAP_REPS_DEFAULT,
  bootstrap_seed = BOOTSTRAP_SEED_DEFAULT,
  cf_folds = CF_FOLDS,
  true_effect = TRUE_EFFECT_DEFAULT,
  null_effect = NULL_EFFECT_DEFAULT,
  ci_level = CI_LEVEL_DEFAULT,
  use_parallel = TRUE,
  cores = NULL
) {
  covariate_registry <- get_covariate_registry()
  procedure_registry <- get_procedure_registry()
  model_registry <- get_model_registry()

  input_files <- list.files(input_root, pattern = "\\.RData$", full.names = TRUE, recursive = TRUE)
  if (length(input_files) == 0) {
    stop(paste("No scenario files found in", input_root))
  }

  all_results <- do.call(rbind, lapply(input_files, function(file_path) {
    env <- new.env(parent = emptyenv())
    load(file_path, envir = env)
    if (!exists("datasets", envir = env) || !exists("metadata", envir = env)) {
      stop(paste("Missing datasets or metadata in", file_path))
    }
    datasets <- get("datasets", envir = env)
    metadata <- get("metadata", envir = env)

    run_all_methods_for_dataset(
      datasets = datasets,
      metadata = metadata,
      procedures = procedures,
      model_specs = model_specs,
      covariate_registry = covariate_registry,
      procedure_registry = procedure_registry,
      model_registry = model_registry,
      n_reps = n_reps,
      bootstrap_reps = bootstrap_reps,
      bootstrap_seed = bootstrap_seed,
      cf_folds = cf_folds,
      true_effect = true_effect,
      null_effect = null_effect,
      ci_level = ci_level,
      use_parallel = use_parallel,
      cores = cores
    )
  }))

  write.csv(all_results, file = output_file, row.names = FALSE)
  all_results
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
}

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

recompute_missing_relative_efficiency <- function(results) {
  # Recompute relative efficiency from Monte Carlo variance when aggregating per-scenario results.
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

resolve_missing_input_root <- function() {
  data_source <- normalize_data_source(get_runtime_option(c("data_source", "DATA_SOURCE"), "local"), default = "local")
  archive_dir <- get_runtime_option(c("archive_datasets_dir", "ARCHIVE_DATASETS_DIR"), "")
  zenodo_url <- get_runtime_option(c("zenodo_data_url", "ZENODO_DATA_URL", "ZENODO_URL"), "")
  zenodo_doi <- get_runtime_option(c("zenodo_data_doi", "ZENODO_DATA_DOI", "ZENODO_DOI"), "")
  input_root <- file.path("simulations", "missing_outcomes", "miss_datasets")
  local_available <- local_dataset_available(input_root, "^setting_.*\\.RData$")

  if (data_source == "regenerate") {
    return(input_root)
  }

  if (data_source == "local" && local_available) {
    return(input_root)
  }

  if (archive_dir %in% c("local", "local_datasets")) {
    return(input_root)
  }

  if (nzchar(archive_dir) && dir.exists(archive_dir)) {
    return(archive_dir)
  }

  archive_reference <- if (nzchar(zenodo_url)) zenodo_url else zenodo_doi
  if (!nzchar(archive_reference)) {
    if (local_available) {
      return(input_root)
    }
    # Use default Zenodo DOI if no local data and no explicit DOI/URL provided.
    # This ensures jobs can proceed without requiring explicit user input for the archive.
    archive_reference <- "https://doi.org/10.5281/zenodo.19393092"
    cat("[data-source] No local missing-outcomes data found. Using default Zenodo DOI: ", archive_reference, "\n", sep = "")
  }

  zenodo_cache <- file.path("simulations", "missing_outcomes", "archives", "zenodo")
  archive_url <- resolve_zenodo_download_url(archive_reference, preferred_file = "Data.zip")
  if (!nzchar(archive_url)) {
    stop("Unable to resolve a Zenodo download URL for the missing-outcomes archive.")
  }
  if (!dir.exists(zenodo_cache)) {
    dir.create(zenodo_cache, recursive = TRUE)
  }
  zip_path <- file.path(zenodo_cache, "zenodo_datasets.zip")
  download_archive_file(archive_url, zip_path)
  safe_unzip_archive(zip_path, zenodo_cache)
  find_archive_root(
    base_dir = zenodo_cache,
    file_pattern = "^setting_.*\\.RData$",
    parent_levels = 1
  )
}

run_procedure_for_setting <- function(
  setting_name,
  procedure_name,
  model_specs,
  n_reps = REPLICATES_DEFAULT,
  bootstrap_reps = BOOTSTRAP_REPS_DEFAULT,
  bootstrap_seed = BOOTSTRAP_SEED_DEFAULT,
  cf_folds = CF_FOLDS,
  true_effect = TRUE_EFFECT_DEFAULT,
  null_effect = NULL_EFFECT_DEFAULT,
  ci_level = CI_LEVEL_DEFAULT,
  use_parallel = TRUE,
  cores = NULL
) {
  input_root <- resolve_missing_input_root()
  setting_dir <- file.path(input_root, setting_name)
  if (!dir.exists(setting_dir)) {
    stop(paste("Missing setting directory:", setting_dir))
  }

  covariate_registry <- get_covariate_registry()
  procedure_registry <- get_procedure_registry()
  model_registry <- get_model_registry()

  setting_root <- file.path("simulations", "missing_outcomes", "miss_intermediate_results", setting_name)
  ensure_dir(setting_root)
  output_root <- file.path(setting_root, procedure_name)
  ensure_dir(output_root)

  log_dir <- file.path("logs", "missing_outcomes", setting_name, procedure_name)
  ensure_dir(log_dir)
  log_path <- file.path(log_dir, sprintf("%s_%s.log", procedure_name, format(Sys.time(), "%Y%m%d_%H%M%S")))
  log_path <- normalizePath(log_path, mustWork = FALSE)
  sink(log_path, append = TRUE, split = TRUE)
  on.exit(sink(), add = TRUE)

  input_files <- list.files(setting_dir, pattern = "\\.RData$", full.names = TRUE)
  if (length(input_files) == 0) {
    stop(paste("No scenario files found in", setting_dir))
  }

  reset_checkpoint <- tolower(Sys.getenv("RESET_CHECKPOINT", "false")) %in% c("true", "t", "1")

  all_aggregated <- list()

  for (model_spec in model_specs) {
    model_dir <- file.path(output_root, model_spec)
    ensure_dir(model_dir)
    scenario_checkpoint <- file.path(model_dir, "scenario_checkpoint.csv")

    if (reset_checkpoint && file.exists(scenario_checkpoint)) {
      file.remove(scenario_checkpoint)
      cat("RESET_CHECKPOINT enabled: cleared", scenario_checkpoint, "\n")
    }
    if (reset_checkpoint) {
      scenario_dirs <- list.dirs(model_dir, recursive = FALSE, full.names = TRUE)
      if (length(scenario_dirs) > 0) {
        unlink(scenario_dirs, recursive = TRUE, force = TRUE)
      }
      scenario_csvs <- list.files(model_dir, pattern = "_aggregated\\.csv$", full.names = TRUE)
      if (length(scenario_csvs) > 0) {
        file.remove(scenario_csvs)
      }
      cat("RESET_CHECKPOINT enabled: cleared per-scenario outputs in", model_dir, "\n")
    }

    processed_files <- character(0)
    if (file.exists(scenario_checkpoint)) {
      processed <- safe_read_csv(scenario_checkpoint)
      if (!is.null(processed) && "file" %in% names(processed)) {
        processed_files <- processed$file
      }
    }
    pending_files <- input_files[!basename(input_files) %in% processed_files]

    cat("Running", procedure_name, "for", model_spec, "on", length(pending_files), "scenario files.\n")

    for (file_path in pending_files) {
      scenario_name <- tools::file_path_sans_ext(basename(file_path))
      scenario_dir <- file.path(model_dir, scenario_name)
      ensure_dir(scenario_dir)
      scenario_progress_dir <- normalizePath(file.path(scenario_dir, "replicate_cache"), mustWork = FALSE)

      cat("Processing scenario:", basename(file_path), "\n")
      env <- new.env(parent = emptyenv())
      load(file_path, envir = env)
      if (!exists("datasets", envir = env) || !exists("metadata", envir = env)) {
        stop(paste("Missing datasets or metadata in", file_path))
      }
      datasets <- get("datasets", envir = env)
      metadata <- get("metadata", envir = env)
      metadata <- normalize_missing_metadata(datasets, metadata, file_path)

      reps_to_run <- min(length(datasets), n_reps)
      cat("Running", reps_to_run, "replicates for", scenario_name, "\n")

      method_checkpoint <- file.path(scenario_dir, "method_checkpoint.csv")
      if (file.exists(method_checkpoint)) {
        method_cp <- safe_read_csv(method_checkpoint)
        if (is.null(method_cp)) {
          method_cp <- data.frame(method = character(), status = character(), timestamp = character(), error = character(), stringsAsFactors = FALSE)
        }
        completed_methods <- method_cp$method[method_cp$status == "done"]
      } else {
        method_cp <- data.frame(method = character(), status = character(), timestamp = character(), error = character(), stringsAsFactors = FALSE)
        completed_methods <- character()
      }

      for (method_name in names(model_registry)) {
        if (method_name %in% completed_methods) {
          cat("Skipping completed method:", method_name, "for", scenario_name, "\n")
          next
        }

        method_rep_checkpoint <- file.path(scenario_dir, paste0(method_name, "_replicate_checkpoint.csv"))
        method_diag_file <- file.path(scenario_dir, paste0(method_name, "_replicate_diagnostics.csv"))
        method_file <- file.path(scenario_dir, paste0(method_name, "_results.csv"))

        method_progress_dir <- file.path(scenario_progress_dir, procedure_name, model_spec, method_name)
        # DO NOT re-merge cached diagnostics here; this causes duplicates on resume.
        # Diagnostics are merged once after the run completes.
        
        completed_reps <- integer(0)
        rep_checkpoint <- NULL
        # Consolidate per-replicate checkpoint files written by parallel workers
        rep_checkpoint <- consolidate_replicate_checkpoints(method_progress_dir)
        if (!is.null(rep_checkpoint)) {
          completed_reps <- rep_checkpoint$replicate[rep_checkpoint$status == "done"]
        }
        replicate_ids <- seq_len(reps_to_run)
        pending_reps <- setdiff(replicate_ids, completed_reps)
        if (length(pending_reps) == 0) {
          # All replicates reported done. Ensure method-level aggregates exist; if not,
          # attempt to reconstruct them from per-replicate cache before marking done.
          method_cache_dir <- file.path(scenario_progress_dir, procedure_name, model_spec, method_name)
          need_aggregation <- !(file.exists(method_diag_file) && file.exists(method_file))
          if (need_aggregation) {
            cached_rows <- read_cached_replicate_diagnostics(method_progress_dir)
            if (!is.null(cached_rows) && nrow(cached_rows) > 0) {
              diag_existing <- NULL
              if (file.exists(method_diag_file)) {
                diag_existing <- safe_read_csv(method_diag_file)
              }
              diag_all_cached <- merge_replicate_diagnostics(diag_existing, cached_rows)
              summary_res <- tryCatch(summarize_missing_replicates(diag_all_cached, metadata), error = function(e) NULL)
              if (!is.null(summary_res)) {
                diag_all <- summary_res$diagnostics
                write.csv(summary_res$summary, file = method_file, row.names = FALSE)
                write.csv(diag_all, method_diag_file, row.names = FALSE)
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
          method_cp <- rbind(method_cp, data.frame(method = method_name, status = "done", timestamp = as.character(Sys.time()), error = "", stringsAsFactors = FALSE))
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
          results <- run_all_models_for_dataset(
            datasets = datasets,
            metadata = metadata,
            procedure_name = procedure_name,
            model_spec = model_spec,
            covariate_registry = covariate_registry,
            procedure_registry = procedure_registry,
            model_registry = setNames(list(model_registry[[method_name]]), method_name),
            n_reps = reps_to_run,
            bootstrap_reps = bootstrap_reps,
            bootstrap_seed = bootstrap_seed,
            cf_folds = cf_folds,
            true_effect = true_effect,
            null_effect = null_effect,
            ci_level = ci_level,
            use_parallel = use_parallel,
            cores = cores,
            replicate_ids = pending_reps,
            progress_dir = scenario_progress_dir,
            log_file = log_path
          )

          diag_new <- attr(results, "replicate_diagnostics")
          diag_existing <- NULL
          if (file.exists(method_diag_file)) {
            diag_existing <- safe_read_csv(method_diag_file)
          }
          diag_all <- merge_replicate_diagnostics(diag_existing, diag_new)
          summary_res <- summarize_missing_replicates(diag_all, metadata)
          diag_all <- summary_res$diagnostics
          # Persist diagnostics before marking checkpoints so resume has data to aggregate.
          write.csv(summary_res$summary, file = method_file, row.names = FALSE)
          write.csv(diag_all, method_diag_file, row.names = FALSE)

          # Delete per-replicate cache files immediately after aggregation to avoid stale data on resume.
          if (dir.exists(method_progress_dir)) {
            cache_files <- list.files(method_progress_dir, pattern = "^replicate_\\d+\\.csv$", full.names = TRUE)
            if (length(cache_files) > 0) {
              file.remove(cache_files)
            }
          }

          # Per-replicate checkpoints are already written atomically by workers; consolidate them for tracking.
          rep_checkpoint <- consolidate_replicate_checkpoints(method_progress_dir)
        }, error = function(e) {
          status <<- "error"
          msg <<- conditionMessage(e)
        })

        method_elapsed <- as.numeric(difftime(Sys.time(), method_start, units = "secs"))
        cat("Running method:", method_name, "completed in", round(method_elapsed, 2), "sec\n")

        if (status == "error") {
          method_cp <- rbind(method_cp, data.frame(method = method_name, status = status, timestamp = as.character(Sys.time()), error = msg, stringsAsFactors = FALSE))
          write.csv(method_cp, method_checkpoint, row.names = FALSE)
          cat("Method checkpointed:", method_name, "for", scenario_name, "\n")
          stop(msg)
        }

        rep_checkpoint <- consolidate_replicate_checkpoints(method_progress_dir)
        completed_reps <- if (!is.null(rep_checkpoint)) rep_checkpoint$replicate[rep_checkpoint$status == "done"] else integer(0)
        if (length(completed_reps) == length(replicate_ids)) {
          method_cp <- rbind(method_cp, data.frame(method = method_name, status = "done", timestamp = as.character(Sys.time()), error = msg, stringsAsFactors = FALSE))
          write.csv(method_cp, method_checkpoint, row.names = FALSE)
          method_cache_dir <- file.path(scenario_progress_dir, procedure_name, model_spec, method_name)
          if (dir.exists(method_cache_dir)) {
            unlink(method_cache_dir, recursive = TRUE, force = TRUE)
          }
          cat("Method checkpointed:", method_name, "for", scenario_name, "\n")
        }
      }

      method_files <- list.files(scenario_dir, pattern = "_results\\.csv$", full.names = TRUE)
      scenario_file <- file.path(scenario_dir, paste0(scenario_name, "_results.csv"))
      if (length(method_files) > 0) {
        method_rows <- lapply(method_files, safe_read_csv)
        method_rows <- method_rows[!vapply(method_rows, is.null, logical(1))]
        if (length(method_rows) == 0) {
          stop("All method-level result files were empty or unreadable while aggregating scenario results.")
        }
        scenario_agg <- do.call(rbind, method_rows)
        scenario_agg <- recompute_missing_relative_efficiency(scenario_agg)
        write.csv(scenario_agg, file = scenario_file, row.names = FALSE)
      } else if (file.exists(scenario_file)) {
        scenario_agg <- safe_read_csv(scenario_file)
        if (is.null(scenario_agg)) {
          stop("Scenario result file exists but is empty or unreadable: ", scenario_file)
        }
        scenario_agg <- recompute_missing_relative_efficiency(scenario_agg)
        write.csv(scenario_agg, file = scenario_file, row.names = FALSE)
      }

      scenario_row <- data.frame(file = basename(file_path), status = "done", timestamp = as.character(Sys.time()), error = "", stringsAsFactors = FALSE)
      # Use atomic read-modify-write to avoid race conditions with concurrent writes
      if (file.exists(scenario_checkpoint)) {
        scenario_cp <- safe_read_csv(scenario_checkpoint)
        if (is.null(scenario_cp)) {
          scenario_cp <- data.frame(file = character(), status = character(), timestamp = character(), error = character(), stringsAsFactors = FALSE)
        }
        scenario_cp <- rbind(scenario_cp, scenario_row)
        scenario_cp <- scenario_cp[!duplicated(scenario_cp$file, fromLast = TRUE), , drop = FALSE]
        write.csv(scenario_cp, scenario_checkpoint, row.names = FALSE)
      } else {
        write.csv(scenario_row, scenario_checkpoint, row.names = FALSE)
      }
      cat("Checkpointed scenario:", basename(file_path), "\n")
    }

    scenario_files <- list.files(model_dir, pattern = "_results\\.csv$", full.names = TRUE, recursive = TRUE)
    scenario_files <- scenario_files[!grepl("_replicate_diagnostics\\.csv$", scenario_files)]
    if (length(scenario_files) > 0) {
      scenario_rows <- lapply(scenario_files, safe_read_csv)
      scenario_rows <- scenario_rows[!vapply(scenario_rows, is.null, logical(1))]
      if (length(scenario_rows) == 0) {
        stop("All scenario result files were empty or unreadable while building model-level aggregate.")
      }
      aggregated <- do.call(rbind, scenario_rows)
      aggregated_file <- file.path(model_dir, paste0(setting_name, "_", procedure_name, "_", model_spec, "_aggregated.csv"))
      write.csv(aggregated, file = aggregated_file, row.names = FALSE)
      all_aggregated[[model_spec]] <- aggregated
    }
  }

  if (length(all_aggregated) > 0) {
    combined <- do.call(rbind, all_aggregated)
    combined_file <- file.path(output_root, paste0(setting_name, "_", procedure_name, "_aggregated.csv"))
    write.csv(combined, file = combined_file, row.names = FALSE)
  }

  procedure_dirs <- list.dirs(setting_root, recursive = FALSE, full.names = TRUE)
  procedure_agg_files <- list.files(setting_root, pattern = paste0("^", setting_name, "_.*_aggregated\\.csv$"), recursive = TRUE, full.names = TRUE)
  procedure_agg_files <- procedure_agg_files[dirname(procedure_agg_files) %in% procedure_dirs]
  if (length(procedure_agg_files) > 0) {
    setting_agg <- do.call(rbind, lapply(procedure_agg_files, read.csv, stringsAsFactors = FALSE))
    setting_file <- file.path(setting_root, paste0(setting_name, "_aggregated.csv"))
    write.csv(setting_agg, file = setting_file, row.names = FALSE)
  }
}
