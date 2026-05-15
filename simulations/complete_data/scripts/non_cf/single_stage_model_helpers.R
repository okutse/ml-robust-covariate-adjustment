# Script for the implementation of the single-stage ML models on all datasets under complete data
# This script will be sourced in the single_stage_results.R script, which will run all the methods on all the datasets and return the results in a dataframe
# Author: A. Okutse
# Date Modified: 2026


# 1. Unadjusted
unadj <- function(df = NULL){
  ## since this is based on the full data set, then use the full data set
  full_unadjusted = mean(df$y[df$A == 1]) - mean(df$y[df$A == 0])
  full_bias_unadjusted = full_unadjusted - 50
  return(data.frame(full_unadjusted, full_bias_unadjusted))
}


# 2. Main effects Linear Regression (MLR)
lm_one <- function(df = NULL){
  # fit random forest model for all individuals
  lm_all <- linear_reg() %>% 
  set_mode("regression") %>% 
  set_engine("lm") %>% 
  fit(formula = y ~ A + x1 + x2 + x3 + x4, data = df)
## set A = 0 and generate predictions for everyone
  df_A0 <- df
  df_A0$A <- 0
  pred_A0 <- predict(lm_all, df_A0)
## set A = 1 and generate predictions for everyone
  df_A1 <- df
  df_A1$A <- 1
  pred_A1 <- predict(lm_all, df_A1)
## compute the ATE
  ATE_adjusted = mean(pred_A1$.pred - pred_A0$.pred)
## compute the biases in absolute values
  bias_adjusted = ATE_adjusted - 50
## return the results as a data frame
  rslt = data.frame("ATE_adjusted" = ATE_adjusted, "bias_adjusted" = bias_adjusted)
  return(rslt)
}



# 3. Interacted Linear Regression (ILR)
lm_interact_one <- function(df = NULL){
  # fit random forest model for all individuals
  lm_all <- linear_reg() %>% 
  set_mode("regression") %>% 
  set_engine("lm") %>% 
  fit(formula = y ~ A + x1 + x2 + x3 + x4 + A*x1 + A*x2 + A*x3 + A*x4, data = df)
## set A = 0 and generate predictions for everyone
  df_A0 <- df
  df_A0$A <- 0
  pred_A0 <- predict(lm_all, df_A0)
## set A = 1 and generate predictions for everyone
  df_A1 <- df
  df_A1$A <- 1
  pred_A1 <- predict(lm_all, df_A1)
## compute the ATE
  ATE_adjusted = mean(pred_A1$.pred - pred_A0$.pred)
## compute the biases in absolute values
  bias_adjusted = ATE_adjusted - 50
## return the results as a data frame
  rslt = data.frame("ATE_adjusted"=ATE_adjusted, "bias_adjusted"=bias_adjusted)
  return(rslt)
}




# 4. Random Forest (RF)
rf_one <- function(df = NULL){
  # fit random forest model for all individuals
  rf_all <- rand_forest(trees = 500) %>% 
    set_mode("regression") %>% 
    set_engine("ranger") %>% 
    fit(formula = y ~ A + x1 + x2 + x3 + x4, data = df)
  ## set A = 0 and generate predictions for everyone
  df_A0 <- df
  df_A0$A <- 0
  pred_A0 <- predict(rf_all, df_A0)
  ## set A = 1 and generate predictions for everyone
  df_A1 <- df
  df_A1$A <- 1
  pred_A1 <- predict(rf_all, df_A1)
  ## compute the ATE
  ATE_adjusted = mean(pred_A1$.pred - pred_A0$.pred)
  ## compute the bias
  bias_adjusted = ATE_adjusted - 50
  ## return the results as a data frame
  rslt = data.frame("ATE_adjusted" = ATE_adjusted, "bias_adjusted" = bias_adjusted)
  return(rslt)
}



# 5. Bayesian Additive Regression Trees (BART)
bart_one <- function(df = NULL){
  # fit random forest model for all individuals
  bart_all <- parsnip::bart(
    mode = "regression") %>%
    set_engine("dbarts") %>% 
  fit(formula = y ~ A + x1 + x2 + x3 + x4, data = df)
## set A = 0 and generate predictions for everyone
  df_A0 <- df
  df_A0$A <- 0
  pred_A0 <- predict(bart_all, df_A0)
## set A = 1 and generate predictions for everyone
  df_A1 <- df
  df_A1$A <- 1
  pred_A1 <- predict(bart_all, df_A1)
## compute the ATE
  ATE_adjusted = mean(pred_A1$.pred - pred_A0$.pred)
## compute the bias
  bias_adjusted = ATE_adjusted - 50
## return the results as a data frame
  rslt = data.frame("ATE_adjusted" = ATE_adjusted, "bias_adjusted" = bias_adjusted)
  return(rslt)
}


## ------------------------------------------------------------------
## Helpers for the custom dbarts learner
## ------------------------------------------------------------------

## Collapse posterior draws to one prediction per row of newdata.
## dbarts can return either:
##   - a vector
##   - a matrix with rows = observations
##   - a matrix with cols = observations
## so we handle all three cases deterministically.
.reduce_dbarts_pred <- function(pred_raw, n_obs) {
  if (is.null(dim(pred_raw))) {
    return(as.numeric(pred_raw))
  }
  
  if (nrow(pred_raw) == n_obs) {
    return(rowMeans(pred_raw))
  }
  
  if (ncol(pred_raw) == n_obs) {
    return(colMeans(pred_raw))
  }
  
  stop("SL.dbarts: could not determine orientation of dbarts predictions.")
}

## Standardize design matrices so training and prediction always use
## the same column names, same order, and numeric storage.
.prepare_dbarts_xy <- function(X, newX = NULL) {
  X <- as.data.frame(X)
  X[] <- lapply(X, as.numeric)
  
  if (!is.null(newX)) {
    newX <- as.data.frame(newX)
    newX <- newX[, colnames(X), drop = FALSE]
    newX[] <- lapply(newX, as.numeric)
  }
  
  list(
    X_df   = X,
    X_mat  = as.matrix(X),
    newX_df  = newX,
    newX_mat = if (!is.null(newX)) as.matrix(newX) else NULL
  )
}


## ------------------------------------------------------------------
## Custom SuperLearner wrapper for dbarts
## ------------------------------------------------------------------
SL.dbarts <- function(Y, X, newX, family, obsWeights, id, ...) {
  
  ## This wrapper is for regression only
  if (!identical(family$family, "gaussian")) {
    stop("SL.dbarts currently supports gaussian outcomes only.")
  }
  
  prep <- .prepare_dbarts_xy(X = X, newX = newX)
  Y <- as.numeric(Y)
  
  ## Capture optional dbarts tuning arguments safely.
  dbarts_args <- list(...)
  if (is.null(dbarts_args$nthread) && is.null(dbarts_args$n.threads)) {
    dbarts_args$nthread <- 1L
  }
  
  ## Keep trees so predict() on the fitted bart object works later.
  if (is.null(dbarts_args$keeptrees)) {
    dbarts_args$keeptrees <- TRUE
  }
  
  ## Fit dbarts using the matrix interface.
  fit_obj <- do.call(
    dbarts::bart,
    c(
      list(
        x.train = prep$X_mat,
        y.train = Y,
        x.test  = prep$newX_mat,
        verbose = FALSE
      ),
      dbarts_args
    )
  )
  
  ## Prefer the mean predictions if available; otherwise collapse draws.
  pred <- if (!is.null(fit_obj$yhat.test.mean)) {
    as.numeric(fit_obj$yhat.test.mean)
  } else {
    .reduce_dbarts_pred(fit_obj$yhat.test, n_obs = nrow(prep$newX_mat))
  }
  
  out <- list(
    pred = pred,
    fit  = list(
      object   = fit_obj,
      colnames = colnames(prep$X_df)
    )
  )
  
  class(out$fit) <- "SL.dbarts"
  out
}


## Prediction method used by predict.SuperLearner()
predict.SL.dbarts <- function(object, newdata, ...) {
  
  ## Rebuild newdata in the exact training-column order.
  newdata <- as.data.frame(newdata)
  newdata <- newdata[, object$colnames, drop = FALSE]
  newdata[] <- lapply(newdata, as.numeric)
  x_test <- as.matrix(newdata)
  
  ## Predict from the fitted dbarts model.
  pred_raw <- predict(object$object, newdata = x_test, ...)
  
  ## If predict() returns posterior draws, collapse to posterior mean.
  .reduce_dbarts_pred(pred_raw, n_obs = nrow(x_test))
}


## ------------------------------------------------------------------
# 6. Super Learner (SL)
## ------------------------------------------------------------------
super_one <- function(df = NULL, cl = NULL) {
  
  ## Build covariates and outcome.
  X <- as.data.frame(df[, c(2, 3, 4, 5, 6), drop = FALSE])
  X[] <- lapply(X, as.numeric)
  Y <- as.numeric(df[, 1])
  
  ## The intervention step below assumes a treatment column named A exists.
  if (!("A" %in% colnames(X))) {
    stop("super_one: X must contain a treatment column named 'A'.")
  }
  ## Candidate learner library (exclude BART to keep runtime manageable).
  sl_library <- c("SL.ranger", "SL.lm", "SL.xgboost")
  
  ## Initialize workers once.
  if (!is.null(cl)) {
    parallel::clusterEvalQ(cl, {
      library(SuperLearner)
      library(dbarts)
      library(ranger)
      library(xgboost)
      NULL
    })
    
    ## Export the custom learner, predict method, and helpers.
    parallel::clusterExport(
      cl,
      varlist = c(
        "SL.dbarts",
        "predict.SL.dbarts",
        ".reduce_dbarts_pred",
        ".prepare_dbarts_xy"
      ),
      envir = environment()
    )
  }
  
  ## Shared fitting helper.
  fit_super <- function(sl_library) {
    if (is.null(cl)) {
      SuperLearner::SuperLearner(
        Y          = Y,
        X          = X,
        family     = gaussian(),
        SL.library = sl_library,
        method     = "method.NNLS",
        cvControl  = list(V = 10)
      )
    } else {
      SuperLearner::snowSuperLearner(
        Y          = Y,
        X          = X,
        family     = gaussian(),
        SL.library = sl_library,
        method     = "method.NNLS",
        cvControl  = list(V = 10),
        cluster    = cl
      )
    }
  }
  
  all <- fit_super(sl_library)
  
  df_A0 <- X
  df_A0$A <- 0
  df_A0 <- df_A0[, colnames(X), drop = FALSE]
  
  pred_A0 <- predict.SuperLearner(object = all, newdata = df_A0, onlySL = TRUE)
  df_A1 <- X
  df_A1$A <- 1
  df_A1 <- df_A1[, colnames(X), drop = FALSE]
  
  pred_A1 <- predict.SuperLearner(object = all, newdata = df_A1, onlySL = TRUE)
  ATE_adjusted  <- mean(pred_A1$pred - pred_A0$pred)
  bias_adjusted <- ATE_adjusted - 50
  
  data.frame(
    ATE_adjusted  = ATE_adjusted,
    bias_adjusted = bias_adjusted
  )
}

# 7. XGBoost
xgboost_one <- function(df = NULL, min_n = 10, tree_depth = 9, learn_rate = 0.02, loss_reduction = 0, ...){ 
## fit the model for all individuals with predictions for everyone
  xg_all <- parsnip::boost_tree(
    mode = "regression",
    trees = 1000,
    min_n = min_n, ## then # of data points at a node before being split further
    tree_depth = tree_depth,  ## max depth of a tree
    learn_rate = learn_rate,  ## shrinkage parameter; step size
    loss_reduction = loss_reduction  ## ctrls model complexity
  ) %>%
    set_engine("xgboost", objective = "reg:squarederror") %>% 
    fit(formula = y ~ A + x1 + x2 + x3 + x4, data = df)
  df_A0 <- df
  df_A0$A <- 0
  pred_A0 <- predict(xg_all, df_A0)
  df_A1 <- df
  df_A1$A <- 1
  pred_A1 <- predict(xg_all, df_A1)
  ATE_adjusted = mean(pred_A1$.pred - pred_A0$.pred)
## compute the bias
  bias_adjusted = ATE_adjusted - 50
## return the results as a data frame
  rslt = data.frame("ATE_adjusted" = ATE_adjusted, "bias_adjusted" = bias_adjusted)
  return(rslt)}

# 8. Correct model (oracle)
correct_model <- function(df = NULL){
  lm_all <- linear_reg() %>% 
  set_mode("regression") %>% 
  set_engine("lm") %>% 
  fit(formula = y ~ A + z1 + z2 + z3 + z4, data = df)
  df_A0 <- df
  df_A0$A <- 0
  pred_A0 <- predict(lm_all, df_A0)
  df_A1 <- df
  df_A1$A <- 1
  pred_A1 <- predict(lm_all, df_A1)
  ATE_adjusted = mean(pred_A1$.pred - pred_A0$.pred)
  bias_adjusted = ATE_adjusted - 50
  rslt = data.frame("ATE_adjusted" = ATE_adjusted, "bias_adjusted" = bias_adjusted)
  return(rslt)
}



# create a registry of the methods to be used in the single-stage analyses
get_single_stage_registry <- function(){
  list(
    unadj = list(name = "unadj", fit = function(df, ...) unadj(df)),
    lm = list(name = "lm", fit = function(df, ...) lm_one(df)),
    lm_interact = list(name = "lm_interact", fit = function(df, ...) lm_interact_one(df)),
    rf = list(name = "rf", fit = function(df, ...) rf_one(df)),
    bart = list(name = "bart", fit = function(df, ...) bart_one(df)),
    super = list(name = "super", fit = function(df, ...) super_one(df, ...)),
    xgboost = list(name = "xgboost", fit = function(df, ...) xgboost_one(df, ...)),
    correct_model = list(name = "correct_model", fit = function(df, ...) correct_model(df))
  )
}




# helper function to extract the desired metrics from the results of the methods
extract_single_stage_metrics <- function(res){
  if (all(c("ATE_adjusted", "bias_adjusted") %in% names(res))) {
    return(list(estimate = res$ATE_adjusted, bias = res$bias_adjusted))
  }
  if (all(c("full_unadjusted", "full_bias_unadjusted") %in% names(res))) {
    return(list(estimate = res$full_unadjusted, bias = res$full_bias_unadjusted))
  }
  stop("Unknown result structure from estimator.")
}



summarize_single_stage_metrics <- function(metrics) {
  # Aggregate replicate-level diagnostics into a single method summary.
  metrics <- as.data.frame(metrics)
  if (nrow(metrics) == 0) {
    stop("No replicate diagnostics available to summarize.")
  }

  data.frame(
    Estimator = metrics$Estimator[1],
    mean_estimate = mean(metrics$estimate),
    mean_bias = mean(metrics$bias),
    sd_estimate = stats::sd(metrics$estimate),
    var_estimate = stats::var(metrics$estimate),
    stringsAsFactors = FALSE
  )
}

# runs all single-stage methods on the list of datasets and returns averaged results
run_single_stage <- function(
  datasets,
  metadata,
  methods = NULL,
  use_parallel = TRUE,
  cores = NULL,
  cl = NULL,
  replicate_ids = NULL
){
  if (length(datasets) == 0) {
    stop("No datasets supplied.")
  }
  if (length(datasets) != nrow(metadata)) {
    stop("Mismatch between datasets and metadata.")
  }
  if (is.null(methods)) {
    methods <- get_single_stage_registry()
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
  } else {
    replicate_ids <- seq_len(length(datasets))
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
  
# helper function to run a single method across all datasets and compute the average metrics
  # Capture replicate-level diagnostics for persistence by callers.
  diagnostics_by_method <- list()
  run_method <- function(method){
    cat("Running method:", method$name, "\n")
    metrics <- foreach::foreach(
      i = seq_along(datasets),
      .combine = rbind,
      .packages = c("parsnip", "ranger", "dbarts", "SuperLearner", "xgboost", "magrittr", "purrr"),
      .export = c("SL.dbarts", "predict.SL.dbarts")
    ) %op% {
      rep_id <- replicate_ids[i]
      # Emit per-replicate timing only in sequential mode to keep logs readable.
      rep_time_start <- Sys.time()
      df <- as.data.frame(datasets[[i]])
      df[] <- lapply(df, function(x) if (is.factor(x)) as.numeric(as.character(x)) else x)
      res <- method$fit(df, cl = cl)
      m <- extract_single_stage_metrics(res)
      if (!use_parallel) {
        rep_time_sec <- as.numeric(difftime(Sys.time(), rep_time_start, units = "secs"))
        cat("Running method:", method$name, "replicate", rep_id, "completed in", round(rep_time_sec, 3), "sec\n")
      }
      c(replicate = rep_id, estimate = m$estimate, bias = m$bias)
    }
    metrics <- as.data.frame(metrics)
    metrics$Estimator <- method$name
    diagnostics_by_method[[method$name]] <<- metrics
    summarize_single_stage_metrics(metrics)
  }

  results <- do.call(rbind, lapply(methods, run_method))

  results$n <- metadata$n[1]
  results$target_r2 <- metadata$target_r2[1]
  results$achieved_r2_raw <- mean(metadata$achieved_r2_raw)
  results$ss <- mean(metadata$ss)

  lm_var <- results$var_estimate[results$Estimator == "lm"]
  results$relative_efficiency <- ifelse(
    results$var_estimate > 0,
    lm_var / results$var_estimate,
    NA
  )

  attr(results, "replicate_diagnostics") <- do.call(rbind, diagnostics_by_method)
  results
}
