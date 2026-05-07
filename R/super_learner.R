
library(SuperLearner)
library(dbarts)
library(ranger)
library(xgboost)
library(parallel)
library(purrr)
library(magrittr)
library(dplyr)
library(tidymodels)

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
## IMPORTANT:
##   - We use dbarts::bart(), not bart2().
##   - bart() is the matrix interface documented with x.train/y.train/x.test.
##   - bart2() in your installed dbarts is formula/data/test based.
## ------------------------------------------------------------------
SL.dbarts <- function(Y, X, newX, family, obsWeights, id, ...) {
  
  ## This wrapper is for regression only, matching your current use.
  if (!identical(family$family, "gaussian")) {
    stop("SL.dbarts currently supports gaussian outcomes only.")
  }
  
  prep <- .prepare_dbarts_xy(X = X, newX = newX)
  Y <- as.numeric(Y)
  
  ## Capture optional dbarts tuning arguments safely.
  dbarts_args <- list(...)
  
  ## Avoid nested parallelism:
  ## snowSuperLearner parallelizes across workers already, so let each
  ## dbarts fit run single-threaded unless the caller explicitly overrides.
  if (is.null(dbarts_args$nthread) && is.null(dbarts_args$n.threads)) {
    dbarts_args$nthread <- 1L
  }
  
  ## Keep trees so predict() on the fitted bart object works later.
  ## This is important for downstream SuperLearner prediction.
  if (is.null(dbarts_args$keeptrees)) {
    dbarts_args$keeptrees <- TRUE
  }
  
  ## Fit dbarts using the matrix interface.
  ## This is the interface that matches SuperLearner's wrapper contract.
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


## ------------------------------------------------------------------
## Prediction method used by predict.SuperLearner()
## ------------------------------------------------------------------
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
## Main SuperLearner routine
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
  
  ## Candidate library.
  sl_library <- c("SL.ranger", "SL.lm", "SL.dbarts", "SL.xgboost")
  
  ## --------------------------------------------------------------
  ## Initialize workers once.
  ## --------------------------------------------------------------
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
  
  ## --------------------------------------------------------------
  ## Shared fitting helper.
  ## --------------------------------------------------------------
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
  
  ## --------------------------------------------------------------
  ## Fit the ensemble, with fallback to dbarts-only if another learner
  ## fails. The earlier "formula" failure came from the dbarts wrapper,
  ## so this should now disappear.
  ## --------------------------------------------------------------
  all <- tryCatch(
    fit_super(sl_library),
    error = function(e) {
      fit_super("SL.dbarts")
    }
  )
  
  ## --------------------------------------------------------------
  ## Counterfactual predictions under A = 0
  ## --------------------------------------------------------------
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
  
  ## --------------------------------------------------------------
  ## Counterfactual predictions under A = 1
  ## --------------------------------------------------------------
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
  
  ## --------------------------------------------------------------
  ## Estimate ATE and bias
  ## --------------------------------------------------------------
  ATE_adjusted  <- mean(pred_A1$pred - pred_A0$pred)
  bias_adjusted <- ATE_adjusted - 50
  
  data.frame(
    ATE_adjusted  = ATE_adjusted,
    bias_adjusted = bias_adjusted
  )
}


# test run SL implementation on a single dataset (a list of 1000 dataframes)
# load(file.path("simulations", "complete_data", "datasets", "complete_n200_r2_0p20.RData"))

# lapply over datasets with super_one, no parallelization
res_serial = lapply(datasets, super_one)

# run with parallelization
# res_parallel = lapply(datasets, super_one, cl = makeCluster(parallel::detectCores(logical = TRUE)-1))



