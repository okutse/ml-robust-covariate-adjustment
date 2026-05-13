# Runner for missing-outcome scenarios across settings

source(file.path("simulations", "missing_outcomes", "miss_scripts", "miss_registry.R"))

# Global defaults (override via environment variables as needed).
CF_FOLDS <- as.integer(Sys.getenv("CF_FOLDS", CF_FOLDS))
BOOTSTRAP_REPS_DEFAULT <- as.integer(Sys.getenv("BOOTSTRAP_REPS", 250))
REPLICATES_DEFAULT <- as.integer(Sys.getenv("REPLICATES_TO_RUN", 500))
BOOTSTRAP_SEED_DEFAULT <- as.integer(Sys.getenv("BOOTSTRAP_SEED", 20260417))
TRUE_EFFECT_DEFAULT <- as.numeric(Sys.getenv("TRUE_EFFECT", 50))
NULL_EFFECT_DEFAULT <- as.numeric(Sys.getenv("NULL_EFFECT", 0))
CI_LEVEL_DEFAULT <- as.numeric(Sys.getenv("CI_LEVEL", 0.95))

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
  cores = NULL
) {
  if (length(datasets) == 0) {
    stop("No datasets supplied.")
  }
  metadata <- normalize_missing_metadata(datasets, metadata)

  if (!is.null(n_reps)) {
    if (!is.numeric(n_reps) || length(n_reps) != 1 || is.na(n_reps) || n_reps < 1) {
      stop("n_reps must be a single positive number.")
    }
    n_keep <- min(length(datasets), as.integer(n_reps))
    datasets <- datasets[seq_len(n_keep)]
    metadata <- metadata[seq_len(n_keep), , drop = FALSE]
  }

  procedure <- procedure_registry[[procedure_name]]
  if (is.null(procedure)) {
    stop(paste("Unknown procedure:", procedure_name))
  }

  if (is.null(cores)) {
    cores <- parallel::detectCores(logical = TRUE)
  }

  if (use_parallel) {
    doParallel::registerDoParallel(max(1, cores - 1))
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

    per_model <- lapply(model_names, function(model_name) {
      cat(
        "Running estimator",
        model_name,
        "for procedure",
        procedure$name,
        "and model spec",
        model_spec,
        "on scenario",
        i,
        "of",
        length(datasets),
        "\n"
      )

      model <- model_registry[[model_name]]
      covariates <- resolve_covariates(model_spec, covariate_registry, df)
      if (model_name == "correct_model") {
        covariates <- resolve_covariates("correct", covariate_registry, df)
      }

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
        set.seed(bootstrap_seed + i)
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

      data.frame(
        replicate = i,
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
    })

    do.call(rbind, per_model)
  }

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

  meta <- data.frame(
    n = metadata$n[1],
    target_r2 = metadata$target_r2[1],
    achieved_r2_raw = mean(metadata$achieved_r2_raw),
    ss = mean(metadata$ss),
    setting = metadata$setting[1],
    stringsAsFactors = FALSE
  )

  output <- cbind(meta, merged)
  attr(output, "replicate_diagnostics") <- results
  output
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

resolve_missing_input_root <- function() {
  data_source <- tolower(Sys.getenv("DATA_SOURCE", "local"))
  archive_dir <- Sys.getenv("ARCHIVE_DATASETS_DIR", "")
  input_root <- file.path("simulations", "missing_outcomes", "miss_datasets")

  if (data_source == "archive") {
    if (archive_dir == "") {
      stop("ARCHIVE_DATASETS_DIR must be set when DATA_SOURCE=archive.")
    }
    input_root <- archive_dir
  }

  input_root
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
      processed <- read.csv(scenario_checkpoint, stringsAsFactors = FALSE)
      processed_files <- processed$file
    }
    pending_files <- input_files[!basename(input_files) %in% processed_files]

    cat("Running", procedure_name, "for", model_spec, "on", length(pending_files), "scenario files.\n")

    for (file_path in pending_files) {
      scenario_name <- tools::file_path_sans_ext(basename(file_path))
      scenario_dir <- file.path(model_dir, scenario_name)
      ensure_dir(scenario_dir)

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
        method_cp <- read.csv(method_checkpoint, stringsAsFactors = FALSE)
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
            cores = cores
          )

          method_file <- file.path(scenario_dir, paste0(method_name, "_results.csv"))
          write.csv(results, file = method_file, row.names = FALSE)

          diag <- attr(results, "replicate_diagnostics")
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

        method_cp <- rbind(method_cp, data.frame(method = method_name, status = status, timestamp = as.character(Sys.time()), error = msg, stringsAsFactors = FALSE))
        write.csv(method_cp, method_checkpoint, row.names = FALSE)
        cat("Method checkpointed:", method_name, "for", scenario_name, "\n")

        if (status == "error") {
          stop(msg)
        }
      }

      method_files <- list.files(scenario_dir, pattern = "_results\\.csv$", full.names = TRUE)
      if (length(method_files) > 0) {
        scenario_agg <- do.call(rbind, lapply(method_files, read.csv, stringsAsFactors = FALSE))
        scenario_file <- file.path(scenario_dir, paste0(scenario_name, "_results.csv"))
        write.csv(scenario_agg, file = scenario_file, row.names = FALSE)
      }

      scenario_row <- data.frame(file = basename(file_path), status = "done", timestamp = as.character(Sys.time()), error = "", stringsAsFactors = FALSE)
      write.table(
        scenario_row,
        file = scenario_checkpoint,
        append = TRUE,
        sep = ",",
        col.names = !file.exists(scenario_checkpoint),
        row.names = FALSE
      )
      cat("Checkpointed scenario:", basename(file_path), "\n")
    }

    scenario_files <- list.files(model_dir, pattern = "_results\\.csv$", full.names = TRUE, recursive = TRUE)
    scenario_files <- scenario_files[!grepl("_replicate_diagnostics\\.csv$", scenario_files)]
    if (length(scenario_files) > 0) {
      aggregated <- do.call(rbind, lapply(scenario_files, read.csv, stringsAsFactors = FALSE))
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
