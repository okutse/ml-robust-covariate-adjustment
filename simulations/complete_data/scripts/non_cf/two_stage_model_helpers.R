# Script for the implementation of the two-stage ML models on all datasets under complete data
# This script will be sourced in the two_stage_results.R script, which will run all the methods on all the datasets and return the results in a dataframe
# Author: A. Okutse
# Date Modified: 2026


# Procedure:
# For a continuous outcome:
# (i)	For your selected ML model of choice, e.g., BART or SuperLearner, fit a model Y ~ A + X i.e. E(Y|A, X). Here, assume no missing data on covariates or the outcome.
# (ii)	For each individual in the data set, use the model in (i) to predict the potential outcome A = 1 and A = 0 to get y(1) and y(0), respectively. We now have a Y and predicted Y(1) and Y(0) for everyone in the dataset.
# (iii)	Now fit the model Y ~ Y(1) + Y(0) + A. The value of the treatment effect A from the model then is the treatment effect estimate.


# 1. Unadjusted (needs to be updated)
unadj <- function(df = NULL){
  ## since this is based on the full data set, then use the full data set
  full_unadjusted = mean(df$y[df$A == 1]) - mean(df$y[df$A == 0])
  full_bias_unadjusted = full_unadjusted - 50
  return(data.frame(full_unadjusted, full_bias_unadjusted))
}



# 2. Main effects Linear Regression (MLR)
lm_one_cv <- function(df) {
  # Define cross-validation folds
  cv_splits <- rsample::vfold_cv(df, v = 5)
  
  # Define a linear regression model
  lm_spec <- linear_reg() %>%
    set_mode("regression") %>%
    set_engine("lm")
  
  # Fit model using cross-validation and extract predictions
  preds <- purrr::map_dfr(cv_splits$splits, function(split) {
    train_data <- rsample::analysis(split)
    test_data <- rsample::assessment(split)
    
    # Fit the model on training data
    lm_fit <- fit(lm_spec, y ~ A + x1 + x2 + x3 + x4, data = train_data)
    
    # Predict Y(0) and Y(1) for test data
    test_data_A0 <- test_data
    test_data_A0$A <- 0
    test_data$y0 <- predict(lm_fit, test_data_A0)$.pred
    
    test_data_A1 <- test_data
    test_data_A1$A <- 1
    test_data$y1 <- predict(lm_fit, test_data_A1)$.pred
    
    return(test_data)
  })
  
  # Second-stage regression to estimate treatment effect
  outM <- glm(y ~ A + y0 + y1, data = preds)
  ATE_adjusted <- coef(outM)["A"]
  
  # Compute bias
  bias_adjusted <- ATE_adjusted - 50
  
  # Return result
  return(data.frame(ATE_adjusted = ATE_adjusted, bias_adjusted = bias_adjusted))
}



# 3. Interacted Linear Regression (ILR)
LMinteract_one_cv <- function(df) {
  # Define cross-validation folds
  cv_splits <- rsample::vfold_cv(df, v = 5)
  
  # Define a linear regression model
  lm_spec <- linear_reg() %>%
    set_mode("regression") %>%
    set_engine("lm")
  
  # Fit model using cross-validation and extract predictions
  preds <- map_dfr(cv_splits$splits, function(split) {
    train_data <- analysis(split)
    test_data <- assessment(split)
    
    # Fit the model on training data
    lm_fit <- fit(lm_spec, y ~ A + x1 + x2 + x3 + x4 + A*x1 + A*x2 + A*x3 + A*x4, data = train_data)
    
    # Predict Y(0) and Y(1) for test data
    test_data_A0 <- test_data
    test_data_A0$A <- 0
    test_data$y0 <- predict(lm_fit, test_data_A0)$.pred
    
    test_data_A1 <- test_data
    test_data_A1$A <- 1
    test_data$y1 <- predict(lm_fit, test_data_A1)$.pred
    
    return(test_data)
  })
  
  # Second-stage regression to estimate treatment effect
  outM <- glm(y ~ A + y0 + y1, data = preds)
  ATE_adjusted <- coef(outM)["A"]
  
  # Compute bias
  bias_adjusted <- ATE_adjusted - 50
  
  # Return result
  return(data.frame(ATE_adjusted = ATE_adjusted, bias_adjusted = bias_adjusted))
}




# 4. Random Forest (RF)
RF_one_cv <- function(df) {
  # Define cross-validation folds
  cv_splits <- rsample::vfold_cv(df, v = 5)
  
  # Define a linear regression model
  lm_spec <- rand_forest(trees = 500) %>% 
    set_mode("regression") %>% 
    set_engine("ranger")
  
  # Fit model using cross-validation and extract predictions
  preds <- map_dfr(cv_splits$splits, function(split) {
    train_data <- analysis(split)
    test_data <- assessment(split)
    
    # Fit the model on training data
    lm_fit <- fit(lm_spec, y ~ A + x1 + x2 + x3 + x4, data = train_data)
    
    # Predict Y(0) and Y(1) for test data
    test_data_A0 <- test_data
    test_data_A0$A <- 0
    test_data$y0 <- predict(lm_fit, test_data_A0)$.pred
    
    test_data_A1 <- test_data
    test_data_A1$A <- 1
    test_data$y1 <- predict(lm_fit, test_data_A1)$.pred
    
    return(test_data)
  })
  
  # Second-stage regression to estimate treatment effect
  outM <- glm(y ~ A + y0 + y1, data = preds)
  ATE_adjusted <- coef(outM)["A"]
  
  # Compute bias
  bias_adjusted <- ATE_adjusted - 50
  
  # Return result
  return(data.frame(ATE_adjusted = ATE_adjusted, bias_adjusted = bias_adjusted))
}





# 5. Bayesian Additive Regression Trees (BART)
bartone_cv <- function(df) {
  # Define cross-validation folds
  cv_splits <- rsample::vfold_cv(df, v = 5)
  
  # Define a linear regression model
  lm_spec <- parsnip::bart(
    mode = "regression") %>%
    set_engine("dbarts")
  
  # Fit model using cross-validation and extract predictions
  preds <- map_dfr(cv_splits$splits, function(split) {
    train_data <- analysis(split)
    test_data <- assessment(split)
    
    # Fit the model on training data
    lm_fit <- fit(lm_spec, y ~ A + x1 + x2 + x3 + x4, data = train_data)
    
    # Predict Y(0) and Y(1) for test data
    test_data_A0 <- test_data
    test_data_A0$A <- 0
    test_data$y0 <- predict(lm_fit, test_data_A0)$.pred
    
    test_data_A1 <- test_data
    test_data_A1$A <- 1
    test_data$y1 <- predict(lm_fit, test_data_A1)$.pred
    
    return(test_data)
  })
  
  # Second-stage regression to estimate treatment effect
  outM <- glm(y ~ A + y0 + y1, data = preds)
  ATE_adjusted <- coef(outM)["A"]
  
  # Compute bias
  bias_adjusted <- ATE_adjusted - 50
  
  # Return result
  return(data.frame(ATE_adjusted = ATE_adjusted, bias_adjusted = bias_adjusted))
}

## -----------------------------------------------------------------------------
## Helpers for the custom dbarts SuperLearner wrapper for two-stage estimation
## -----------------------------------------------------------------------------
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

## Standardize training and prediction covariates so that:
.prepare_dbarts_xy <- function(X, newX = NULL) {
  X <- as.data.frame(X)
  X[] <- lapply(X, as.numeric)
  
  if (!is.null(newX)) {
    newX <- as.data.frame(newX)
    newX <- newX[, colnames(X), drop = FALSE]
    newX[] <- lapply(newX, as.numeric)
  }
  
  list(
    X_df     = X,
    X_mat    = as.matrix(X),
    newX_df  = newX,
    newX_mat = if (!is.null(newX)) as.matrix(newX) else NULL
  )
}


## ------------------------------------------------------------------
## Custom SuperLearner wrapper for dbarts
## ------------------------------------------------------------------
SL.dbarts <- function(Y, X, newX, family, obsWeights, id, ...) {
  
  ## This implementation is for regression, matching family = gaussian().
  if (!identical(family$family, "gaussian")) {
    stop("SL.dbarts currently supports gaussian outcomes only.")
  }
  
  prep <- .prepare_dbarts_xy(X = X, newX = newX)
  Y <- as.numeric(Y)
  
  dbarts_args <- list(...)
  if (is.null(dbarts_args$nthread) && is.null(dbarts_args$n.threads)) {
    dbarts_args$nthread <- 1L
  }
  
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

## Prediction method for the fitted dbarts SuperLearner base learner
predict.SL.dbarts <- function(object, newdata, ...) {
  
  ## Rebuild prediction data using the same training column order.
  newdata <- as.data.frame(newdata)
  newdata <- newdata[, object$colnames, drop = FALSE]
  newdata[] <- lapply(newdata, as.numeric)
  x_test <- as.matrix(newdata)
  
  ## Predict from the fitted dbarts object.
  pred_raw <- predict(object$object, newdata = x_test, ...)
  
  ## Convert posterior draws to posterior mean predictions.
  .reduce_dbarts_pred(pred_raw, n_obs = nrow(x_test))
}


## ------------------------------------------------------------------
## 6. Two-stage SuperLearner estimator with cross-validation
## ------------------------------------------------------------------
## Stage 1:
##   Fit SuperLearner and obtain y0hat, y1hat.
##
## Stage 2:
##   Regress observed y on A, y0hat, y1hat and use the coefficient
##   of A as the adjusted treatment effect.
## ------------------------------------------------------------------
super_one_cv <- function(df, cl = NULL) {
  
  ## Use the same CV setting as in the original implementation.
  cv_control <- list(V = 10)
  
  ## Construct covariate matrix and outcome.
  X <- as.data.frame(df[, c(2, 3, 4, 5, 6), drop = FALSE])
  X[] <- lapply(X, as.numeric)
  Y <- as.numeric(df[, 1])
  
  ## The intervention step below assumes the treatment column is named A.
  if (!("A" %in% colnames(X))) {
    stop("super_one_cv: X must contain a treatment column named 'A'.")
  }
  
  ## The second-stage glm below assumes the data frame contains variables
  ## named y and A. Make this explicit rather than failing later.
  if (!("y" %in% colnames(df))) {
    stop("super_one_cv: df must contain an observed outcome column named 'y'.")
  }
  if (!("A" %in% colnames(df))) {
    stop("super_one_cv: df must contain a treatment column named 'A'.")
  }
  
  ## Candidate learner library.
  sl_library <- c("SL.ranger", "SL.lm", "SL.dbarts", "SL.xgboost")
  
  ## --------------------------------------------------------------
  ## If parallelization is requested, initialize workers once and
  ## export the custom learner plus helper functions once.
  ## --------------------------------------------------------------
  if (!is.null(cl)) {
    parallel::clusterEvalQ(cl, {
      library(SuperLearner)
      library(dbarts)
      library(ranger)
      library(xgboost)
      NULL
    })
    
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
  
  ## --------------------------------------------------------------
  ## Shared fitting helper so serial and parallel branches remain
  ## identical except for the SuperLearner backend.
  ## --------------------------------------------------------------
  fit_super <- function(sl_library) {
    if (is.null(cl)) {
      SuperLearner::SuperLearner(
        Y          = Y,
        X          = X,
        cvControl  = cv_control,
        family     = gaussian(),
        SL.library = sl_library,
        method     = "method.NNLS"
      )
    } else {
      SuperLearner::snowSuperLearner(
        Y          = Y,
        X          = X,
        cvControl  = cv_control,
        family     = gaussian(),
        SL.library = sl_library,
        method     = "method.NNLS",
        cluster    = cl
      )
    }
  }
  
  all <- tryCatch(
    fit_super(sl_library),
    error = function(e) {
      fit_super("SL.dbarts")
    }
  )
  
  ## Potential outcome predictions under A = 0
  df_A0 <- X
  df_A0$A <- 0
  df_A0 <- df_A0[, colnames(X), drop = FALSE]
  
  pred_A0 <- tryCatch(
    predict.SuperLearner(object = all, newdata = df_A0, onlySL = TRUE),
    error = function(e) {
      all_local <- fit_super("SL.dbarts")
      predict.SuperLearner(object = all_local, newdata = df_A0, onlySL = TRUE)
    }
  )
  
  ## Potential outcome predictions under A = 1
  df_A1 <- X
  df_A1$A <- 1
  df_A1 <- df_A1[, colnames(X), drop = FALSE]
  
  pred_A1 <- tryCatch(
    predict.SuperLearner(object = all, newdata = df_A1, onlySL = TRUE),
    error = function(e) {
      all_local <- fit_super("SL.dbarts")
      predict.SuperLearner(object = all_local, newdata = df_A1, onlySL = TRUE)
    }
  )
  
  ## Store first-stage nuisance estimates in the original data frame.
  df$y0 <- as.numeric(pred_A0$pred)
  df$y1 <- as.numeric(pred_A1$pred)
  
  ## Second-stage regression:
  ##   y ~ A + y0 + y1
  outM <- glm(y ~ A + y0 + y1, data = df)
  
  ## Extract the adjusted ATE from the coefficient on A.
  ATE_adjusted <- unname(coef(outM)["A"])
  
  ## Compute bias relative to the target value 50.
  bias_adjusted <- ATE_adjusted - 50
  
  ## Return results in the same downstream-compatible format.
  data.frame(
    ATE_adjusted  = ATE_adjusted,
    bias_adjusted = bias_adjusted
  )
}


# 7. XGBoost
XGB_one_cv <- function(df = NULL, min_n = 10, tree_depth = 9, learn_rate = 0.02, loss_reduction = 0, ...) {
  # Define cross-validation folds
  cv_splits <- rsample::vfold_cv(df, v = 5)
  
  # Define a linear regression model
  lm_spec <- parsnip::boost_tree(
    mode = "regression",
    trees = 1000,
    min_n = min_n, ## then # of data points at a node before being split further
    tree_depth = tree_depth,  ## max depth of a tree
    learn_rate = learn_rate,  ## shrinkage parameter; step size
    loss_reduction = loss_reduction  ## ctrls model complexity
  ) %>%
    set_engine("xgboost", objective = "reg:squarederror")
  
  # Fit model using cross-validation and extract predictions
  preds <- map_dfr(cv_splits$splits, function(split) {
    train_data <- analysis(split)
    test_data <- assessment(split)
    
    # Fit the model on training data
    lm_fit <- fit(lm_spec, y ~ A + x1 + x2 + x3 + x4, data = train_data)
    
    # Predict Y(0) and Y(1) for test data
    test_data_A0 <- test_data
    test_data_A0$A <- 0
    test_data$y0 <- predict(lm_fit, test_data_A0)$.pred
    
    test_data_A1 <- test_data
    test_data_A1$A <- 1
    test_data$y1 <- predict(lm_fit, test_data_A1)$.pred
    
    return(test_data)
  })
  
  # Second-stage regression to estimate treatment effect
  outM <- glm(y ~ A + y0 + y1, data = preds)
  ATE_adjusted <- coef(outM)["A"]
  
  # Compute bias
  bias_adjusted <- ATE_adjusted - 50
  
  # Return result
  return(data.frame(ATE_adjusted = ATE_adjusted, bias_adjusted = bias_adjusted))
}

# 8. Correct model specification (oracle)
correct_model_cv <- function(df) {
  # Define cross-validation folds
  cv_splits <- rsample::vfold_cv(df, v = 5)
  
  # Define a linear regression model
  lm_spec <- linear_reg() %>%
    set_mode("regression") %>%
    set_engine("lm")
  
  # Fit model using cross-validation and extract predictions
  preds <- map_dfr(cv_splits$splits, function(split) {
    train_data <- analysis(split)
    test_data <- assessment(split)
    
    # Fit the model on training data
    lm_fit <- fit(lm_spec, y ~ A + z1 + z2 + z3 + z4, data = train_data)
    
    # Predict Y(0) and Y(1) for test data
    test_data_A0 <- test_data
    test_data_A0$A <- 0
    test_data$y0 <- predict(lm_fit, test_data_A0)$.pred
    
    test_data_A1 <- test_data
    test_data_A1$A <- 1
    test_data$y1 <- predict(lm_fit, test_data_A1)$.pred
    
    return(test_data)
  })
  
  # Second-stage regression to estimate treatment effect
  outM <- glm(y ~ A + y0 + y1, data = preds)
  ATE_adjusted <- coef(outM)["A"]
  
  # Compute bias
  bias_adjusted <- ATE_adjusted - 50
  
  # Return result
  return(data.frame(ATE_adjusted = ATE_adjusted, bias_adjusted = bias_adjusted))
}



# create a registry of the methods to be used in the two-stage analyses
get_two_stage_registry <- function(){
  list(
    unadj = list(name = "unadj", fit = function(df, ...) unadj(df)),
    lm = list(name = "lm", fit = function(df, ...) lm_one_cv(df)),
    lm_interact = list(name = "lm_interact", fit = function(df, ...) LMinteract_one_cv(df)),
    rf = list(name = "rf", fit = function(df, ...) RF_one_cv(df)),
    bart = list(name = "bart", fit = function(df, ...) bartone_cv(df)),
    super = list(name = "super", fit = function(df, ...) super_one_cv(df, ...)),
    xgboost = list(name = "xgboost", fit = function(df, ...) XGB_one_cv(df, ...)),
    correct_model = list(name = "correct_model", fit = function(df, ...) correct_model_cv(df))
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



# runs all two-stage methods on the list of datasets and returns averaged results
run_two_stage <- function(datasets, metadata, methods = NULL, use_parallel = TRUE, cores = NULL, cl = NULL){
  if (length(datasets) == 0) {
    stop("No datasets supplied.")
  }
  if (length(datasets) != nrow(metadata)) {
    stop("Mismatch between datasets and metadata.")
  }
  if (is.null(methods)) {
    methods <- get_two_stage_registry()
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
  run_method <- function(method){
    cat("Running method:", method$name, "\n")
    metrics <- foreach::foreach(
      i = seq_along(datasets),
      .combine = rbind,
      .packages = c("parsnip", "ranger", "dbarts", "SuperLearner", "xgboost", "magrittr", "purrr", "tidymodels"),
      .export = c("SL.dbarts", "predict.SL.dbarts")
    ) %op% {
      res <- method$fit(datasets[[i]], cl = cl)
      m <- extract_single_stage_metrics(res)
      c(estimate = m$estimate, bias = m$bias)
    }

    data.frame(
      Estimator = method$name,
      mean_estimate = mean(metrics[, "estimate"]),
      mean_bias = mean(metrics[, "bias"]),
      sd_estimate = sd(metrics[, "estimate"]),
      var_estimate = var(metrics[, "estimate"]),
      stringsAsFactors = FALSE
    )
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

  results
}
