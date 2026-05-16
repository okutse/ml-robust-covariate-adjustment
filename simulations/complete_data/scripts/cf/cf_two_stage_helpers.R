# Cross-fitted two-stage helpers (K is configurable below)

# Global CF settings keep fold count and bootstrap size consistent across methods.
CF_FOLDS <- 2
BOOTSTRAP_REPS_DEFAULT <- 250
# Default number of Monte Carlo replicates to process unless overridden by the caller.
REPLICATES_DEFAULT <- 500

# Ensure the pipe operator is available when helpers are sourced directly.
if (!exists("%>%")) {
  `%>%` <- magrittr::`%>%`
}

ensure_dir <- function(path) {
  if (!is.null(path) && nzchar(path) && !dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
}

log_progress_line <- function(log_file, line) {
  cat(line, "\n")
  if (sink.number() > 0) {
    return(invisible(NULL))
  }
  if (!is.null(log_file) && nzchar(log_file)) {
    cat(line, "\n", file = log_file, append = TRUE)
  }
}

write_replicate_cache <- function(method_dir, rep_row) {
  if (is.null(method_dir) || !nzchar(method_dir)) {
    return(invisible(NULL))
  }
  ensure_dir(method_dir)
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
    combined_list[[length(combined_list) + 1]] <- read.csv(combined_file, stringsAsFactors = FALSE)
  }
  checkpoint_files <- list.files(method_dir, pattern = "^replicate_\\d+_checkpoint\\.csv$", full.names = TRUE)
  if (length(checkpoint_files) > 0) {
    combined_list <- c(combined_list, lapply(checkpoint_files, function(f) read.csv(f, stringsAsFactors = FALSE)))
  }
  if (length(combined_list) == 0) return(NULL)
  combined <- do.call(rbind, combined_list)
  if (nrow(combined) > 0) {
    combined <- combined[!duplicated(combined$replicate, fromLast = TRUE), , drop = FALSE]
    rownames(combined) <- NULL
  }
  combined
}

unadj_cf <- function(df = NULL) {
  est <- mean(df$y[df$A == 1]) - mean(df$y[df$A == 0])
  data.frame(full_unadjusted = est, full_bias_unadjusted = est - 50)
}

crossfit_two_stage <- function(df, fit_fun, pred_fun, v = CF_FOLDS) {
  df_cf <- as.data.frame(df)
  df_cf$.row_id <- seq_len(nrow(df_cf))

  folds <- rsample::vfold_cv(df_cf, v = v)
  y0 <- rep(NA_real_, nrow(df_cf))
  y1 <- rep(NA_real_, nrow(df_cf))

  for (split in folds$splits) {
    train_dat <- rsample::analysis(split)
    assess_dat <- rsample::assessment(split)

    train_no_id <- train_dat[, setdiff(names(train_dat), ".row_id"), drop = FALSE]
    assess_no_id <- assess_dat[, setdiff(names(assess_dat), ".row_id"), drop = FALSE]

    fit_obj <- fit_fun(train_no_id)

    assess_A0 <- assess_no_id
    assess_A0$A <- 0
    assess_A1 <- assess_no_id
    assess_A1$A <- 1

    idx <- assess_dat$.row_id
    y0[idx] <- pred_fun(fit_obj, assess_A0)
    y1[idx] <- pred_fun(fit_obj, assess_A1)
  }

  if (anyNA(y0) || anyNA(y1)) {
    stop("Cross-fitting failed: some out-of-fold predictions are missing.")
  }

  stage2_df <- df_cf[, setdiff(names(df_cf), ".row_id"), drop = FALSE]
  stage2_df$y0 <- y0
  stage2_df$y1 <- y1

  out_m <- stats::glm(y ~ A + y0 + y1, data = stage2_df)
  ate <- unname(stats::coef(out_m)["A"])
  data.frame(ATE_adjusted = ate, bias_adjusted = ate - 50)
}

lm_one_cf <- function(df) {
  crossfit_two_stage(
    df = df,
    fit_fun = function(train) {
      parsnip::linear_reg() %>%
        parsnip::set_mode("regression") %>%
        parsnip::set_engine("lm") %>%
        parsnip::fit(y ~ A + x1 + x2 + x3 + x4, data = train)
    },
    pred_fun = function(fit_obj, new_data) as.numeric(predict(fit_obj, new_data)$.pred),
    v = CF_FOLDS
  )
}

lm_interact_one_cf <- function(df) {
  crossfit_two_stage(
    df = df,
    fit_fun = function(train) {
      parsnip::linear_reg() %>%
        parsnip::set_mode("regression") %>%
        parsnip::set_engine("lm") %>%
        parsnip::fit(y ~ A + x1 + x2 + x3 + x4 + A * x1 + A * x2 + A * x3 + A * x4, data = train)
    },
    pred_fun = function(fit_obj, new_data) as.numeric(predict(fit_obj, new_data)$.pred),
    v = CF_FOLDS
  )
}

rf_one_cf <- function(df) {
  crossfit_two_stage(
    df = df,
    fit_fun = function(train) {
      # Use off-the-shelf random forest defaults (no tuning).
      parsnip::rand_forest() %>%
        parsnip::set_mode("regression") %>%
        parsnip::set_engine("ranger") %>%
        parsnip::fit(y ~ A + x1 + x2 + x3 + x4, data = train)
    },
    pred_fun = function(fit_obj, new_data) as.numeric(predict(fit_obj, new_data)$.pred),
    v = CF_FOLDS
  )
}

bart_one_cf <- function(df) {
  crossfit_two_stage(
    df = df,
    fit_fun = function(train) {
      parsnip::bart(mode = "regression") %>%
        parsnip::set_engine("dbarts") %>%
        parsnip::fit(y ~ A + x1 + x2 + x3 + x4, data = train)
    },
    pred_fun = function(fit_obj, new_data) as.numeric(predict(fit_obj, new_data)$.pred),
    v = CF_FOLDS
  )
}

super_fit_cf <- function(train, cl = NULL) {
  x_names <- c("A", "x1", "x2", "x3", "x4")
  X <- train[, x_names, drop = FALSE]
  X[] <- lapply(X, as.numeric)
  Y <- as.numeric(train$y)

  if (is.null(cl)) {
    fit <- SuperLearner::SuperLearner(
      Y = Y,
      X = X,
      family = gaussian(),
      SL.library = c("SL.ranger", "SL.lm", "SL.xgboost"),
      method = "method.NNLS",
      cvControl = list(V = 10)
    )
  } else {
    fit <- SuperLearner::snowSuperLearner(
      Y = Y,
      X = X,
      family = gaussian(),
      SL.library = c("SL.ranger", "SL.lm", "SL.xgboost"),
      method = "method.NNLS",
      cvControl = list(V = 10),
      cluster = cl
    )
  }

  fit$covariates <- x_names
  fit
}

super_pred_cf <- function(fit_obj, new_data) {
  required <- fit_obj$covariates
  X_new <- new_data[, required, drop = FALSE]
  X_new[] <- lapply(X_new, as.numeric)
  as.numeric(predict.SuperLearner(fit_obj, newdata = X_new, onlySL = TRUE)$pred)
}

super_one_cf <- function(df, cl = NULL) {
  crossfit_two_stage(
    df = df,
    fit_fun = function(train) super_fit_cf(train, cl = cl),
    pred_fun = super_pred_cf,
    v = CF_FOLDS
  )
}

xgboost_one_cf <- function(df, cl = NULL) {
  crossfit_two_stage(
    df = df,
    fit_fun = function(train) {
      # Use off-the-shelf gradient boosting defaults (no tuning).
      parsnip::boost_tree(mode = "regression") %>%
        parsnip::set_engine("xgboost", objective = "reg:squarederror") %>%
        parsnip::fit(y ~ A + x1 + x2 + x3 + x4, data = train)
    },
    pred_fun = function(fit_obj, new_data) as.numeric(predict(fit_obj, new_data)$.pred),
    v = CF_FOLDS
  )
}

correct_model_cf <- function(df) {
  crossfit_two_stage(
    df = df,
    fit_fun = function(train) {
      parsnip::linear_reg() %>%
        parsnip::set_mode("regression") %>%
        parsnip::set_engine("lm") %>%
        parsnip::fit(y ~ A + z1 + z2 + z3 + z4, data = train)
    },
    pred_fun = function(fit_obj, new_data) as.numeric(predict(fit_obj, new_data)$.pred),
    v = CF_FOLDS
  )
}

get_two_stage_cf_registry <- function() {
  list(
    unadj = list(name = "unadj", fit = function(df, ...) unadj_cf(df)),
    lm = list(name = "lm", fit = function(df, ...) lm_one_cf(df)),
    lm_interact = list(name = "lm_interact", fit = function(df, ...) lm_interact_one_cf(df)),
    rf = list(name = "rf", fit = function(df, ...) rf_one_cf(df)),
    bart = list(name = "bart", fit = function(df, ...) bart_one_cf(df)),
    super = list(name = "super", fit = function(df, ...) super_one_cf(df, ...)),
    xgboost = list(name = "xgboost", fit = function(df, ...) xgboost_one_cf(df, ...)),
    correct_model = list(name = "correct_model", fit = function(df, ...) correct_model_cf(df))
  )
}

extract_two_stage_metrics_cf <- function(res) {
  if (all(c("ATE_adjusted", "bias_adjusted") %in% names(res))) {
    return(list(estimate = res$ATE_adjusted, bias = res$bias_adjusted))
  }
  if (all(c("full_unadjusted", "full_bias_unadjusted") %in% names(res))) {
    return(list(estimate = res$full_unadjusted, bias = res$full_bias_unadjusted))
  }
  stop("Unknown result structure from estimator.")
}

summarize_two_stage_cf_metrics <- function(metrics) {
  # Aggregate replicate-level diagnostics into a single method summary.
  metrics <- as.data.frame(metrics)
  if (nrow(metrics) == 0) {
    stop("No replicate diagnostics available to summarize.")
  }

  mc_mean_estimate <- mean(metrics$estimate)
  metrics$covered_mc_mean_95 <- as.integer(
    metrics$ci_lower <= mc_mean_estimate & mc_mean_estimate <= metrics$ci_upper
  )

  mc_sd_estimate <- stats::sd(metrics$estimate)
  mc_var_estimate <- stats::var(metrics$estimate)
  mc_se_estimate <- mc_sd_estimate / sqrt(nrow(metrics))

  summary <- data.frame(
    Estimator = metrics$Estimator[1],
    mean_estimate = mean(metrics$estimate),
    mean_bias = mean(metrics$bias),
    mc_sd_estimate = mc_sd_estimate,
    mc_var_estimate = mc_var_estimate,
    mc_se_estimate = mc_se_estimate,
    # Backward-compatible aliases: these remain Monte Carlo quantities.
    sd_estimate = mc_sd_estimate,
    var_estimate = mc_var_estimate,
    mean_bootstrap_se = mean(metrics$bootstrap_se),
    sd_bootstrap_se = stats::sd(metrics$bootstrap_se),
    mean_bootstrap_bias = mean(metrics$bootstrap_bias),
    coverage_95 = mean(metrics$covered_true_95),
    coverage_95_mc_mean = mean(metrics$covered_mc_mean_95),
    coverage_95_bc = mean(metrics$covered_true_95_bc),
    reject_h0_effect_prob = mean(metrics$reject_h0_effect),
    reject_h0_null_prob = mean(metrics$reject_h0_null),
    mean_ci_lower = mean(metrics$ci_lower),
    mean_ci_upper = mean(metrics$ci_upper),
    mean_bc_ci_lower = mean(metrics$bc_ci_lower),
    mean_bc_ci_upper = mean(metrics$bc_ci_upper),
    mean_bootstrap_time_sec = mean(metrics$bootstrap_time_sec),
    mean_replicate_time_sec = mean(metrics$replicate_time_sec),
    stringsAsFactors = FALSE
  )

  list(summary = summary, diagnostics = metrics)
}

run_two_stage_cf <- function(
  datasets,
  metadata,
  methods = NULL,
  use_parallel = TRUE,
  cores = NULL,
  cl = NULL,
  n_reps = REPLICATES_DEFAULT,
  bootstrap_reps = BOOTSTRAP_REPS_DEFAULT,
  bootstrap_seed = 20260417,
  true_effect = 50,
  null_effect = 0,
  ci_level = 0.95,
  replicate_ids = NULL,
  progress_dir = NULL,
  log_file = NULL
) {
  if (length(datasets) == 0) {
    stop("No datasets supplied.")
  }
  if (length(datasets) != nrow(metadata)) {
    stop("Mismatch between datasets and metadata.")
  }
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
    # Subset inputs so the caller can run fewer replicates than are stored on disk.
    datasets <- datasets[seq_len(n_keep)]
    metadata <- metadata[seq_len(n_keep), , drop = FALSE]
    replicate_ids <- seq_len(n_keep)
  } else {
    replicate_ids <- seq_len(length(datasets))
  }
  if (is.null(methods)) {
    methods <- get_two_stage_cf_registry()
  }
  if (is.null(cores)) {
    cores <- parallel::detectCores(logical = TRUE)
  }
  # Keep the CI critical value explicit so the nominal level is configurable.
  z_crit <- stats::qnorm((1 + ci_level) / 2)

  if (use_parallel) {
    doParallel::registerDoParallel(max(1, cores - 1))
    `%op%` <- foreach::`%dopar%`
  } else {
    foreach::registerDoSEQ()
    `%op%` <- foreach::`%do%`
  }

  # Store replicate-level diagnostics per method so downstream scripts can log/write them.
  diagnostics_by_method <- list()

  run_method <- function(method) {
    cat("Running CF method:", method$name, "\n")
    method_progress_dir <- NULL
    if (!is.null(progress_dir) && nzchar(progress_dir)) {
      method_progress_dir <- file.path(progress_dir, method$name)
      ensure_dir(method_progress_dir)
    }
    metrics <- foreach::foreach(
      i = seq_along(datasets),
      .combine = rbind,
      .packages = c("parsnip", "ranger", "dbarts", "SuperLearner", "xgboost", "magrittr", "purrr", "rsample")
    ) %op% {
      rep_id <- replicate_ids[i]
      # Emit per-replicate progress only in sequential mode to keep logs ordered and readable.
      if (!use_parallel) {
        cat("Running method:", method$name, "replicate", rep_id, "of", length(datasets), "\n")
      } else {
        log_progress_line(log_file, sprintf(
          "Running method: %s replicate %s of %s",
          method$name,
          rep_id,
          length(datasets)
        ))
      }
      # Per-replicate timing tracks full runtime (point estimate + bootstrap) for this method.
      rep_time_start <- Sys.time()

      df <- as.data.frame(datasets[[i]])
      df[] <- lapply(df, function(x) if (is.factor(x)) as.numeric(as.character(x)) else x)

      # Reusable estimator wrapper: each bootstrap draw calls this to rerun the full CF algorithm.
      estimate_from_df <- function(dat) {
        point_res <- method$fit(dat, cl = cl)
        point_m <- extract_two_stage_metrics_cf(point_res)
        as.numeric(point_m$estimate)
      }

      point_estimate <- estimate_from_df(df)
      point_bias <- point_estimate - true_effect

      # Bootstrap timing is tracked separately to quantify the overhead of model-based SE estimation.
      boot_time_start <- Sys.time()
      set.seed(bootstrap_seed + rep_id)
      boot_estimates <- vapply(seq_len(bootstrap_reps), function(b) {
        # Best practice: resample rows and rerun the entire estimator (including nuisance cross-fitting).
        idx <- sample.int(nrow(df), size = nrow(df), replace = TRUE)
        df_boot <- df[idx, , drop = FALSE]
        estimate_from_df(df_boot)
      }, numeric(1))
      bootstrap_time_sec <- as.numeric(difftime(Sys.time(), boot_time_start, units = "secs"))

      bootstrap_se <- stats::sd(boot_estimates)
      # Bias-correct the point estimate using the bootstrap mean.
      bootstrap_bias <- mean(boot_estimates) - point_estimate
      bc_estimate <- point_estimate - bootstrap_bias
      ci_lower <- point_estimate - z_crit * bootstrap_se
      ci_upper <- point_estimate + z_crit * bootstrap_se
      bc_ci_lower <- bc_estimate - z_crit * bootstrap_se
      bc_ci_upper <- bc_estimate + z_crit * bootstrap_se

      # Coverage evaluates calibration against the known true ATE in this simulation design.
      covered_true_95 <- as.integer(ci_lower <= true_effect && true_effect <= ci_upper)
      # Bias-corrected coverage uses the bootstrap bias-adjusted center.
      covered_true_95_bc <- as.integer(bc_ci_lower <= true_effect && true_effect <= bc_ci_upper)

      # Rejection indicator for H0: treatment effect equals true_effect (type I error if true_effect is the null).
      reject_h0_effect <- as.integer(ci_lower > true_effect || ci_upper < true_effect)
      # Optional power diagnostic for a user-specified null effect.
      reject_h0_null <- as.integer(ci_lower > null_effect || ci_upper < null_effect)

      total_rep_time_sec <- as.numeric(difftime(Sys.time(), rep_time_start, units = "secs"))

      rep_row <- data.frame(
        replicate = rep_id,
        Estimator = method$name,
        estimate = point_estimate,
        bias = point_bias,
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
        replicate_time_sec = total_rep_time_sec,
        stringsAsFactors = FALSE
      )

      write_replicate_cache(method_progress_dir, rep_row)

      if (!use_parallel) {
        cat("Running method:", method$name, "replicate", rep_id, "completed in", round(total_rep_time_sec, 3), "sec\n")
      } else {
        log_progress_line(log_file, sprintf(
          "Running method: %s replicate %s completed in %s sec",
          method$name,
          rep_id,
          round(total_rep_time_sec, 3)
        ))
      }

      rep_row
    }

    metrics$Estimator <- method$name
    summary_res <- summarize_two_stage_cf_metrics(metrics)
    diagnostics_by_method[[method$name]] <<- summary_res$diagnostics
    summary_res$summary
  }

  results <- do.call(rbind, lapply(methods, run_method))
  results$n <- metadata$n[1]
  results$target_r2 <- metadata$target_r2[1]
  results$achieved_r2_raw <- mean(metadata$achieved_r2_raw)
  results$ss <- mean(metadata$ss)

  # Relative efficiency remains Monte Carlo-based (not bootstrap-based).
  lm_var <- results$mc_var_estimate[results$Estimator == "lm"]
  results$relative_efficiency <- ifelse(results$mc_var_estimate > 0, lm_var / results$mc_var_estimate, NA)

  # Attach replicate-level diagnostics so scenario scripts can persist detailed logs downstream.
  attr(results, "replicate_diagnostics") <- do.call(rbind, diagnostics_by_method)

  results
}
