## ------------------------------------------------------------------
## Helpers for the custom dbarts SuperLearner wrapper
## ------------------------------------------------------------------

## Collapse posterior draws to one fitted value per observation.
## dbarts may return:
##   - a vector
##   - a matrix with rows = observations
##   - a matrix with columns = observations
## We handle all supported cases explicitly.
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
##   - all columns are numeric
##   - prediction columns are in the exact training order
##   - matrix conversion is deterministic on every worker
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
## IMPORTANT:
##   - use dbarts::bart(), not bart2()
##   - bart() is the matrix-style interface that matches the
##     SuperLearner wrapper signature Y, X, newX
##   - keep trees so predict() works on the fitted object later
## ------------------------------------------------------------------
SL.dbarts <- function(Y, X, newX, family, obsWeights, id, ...) {
  
  ## This implementation is for regression, matching family = gaussian().
  if (!identical(family$family, "gaussian")) {
    stop("SL.dbarts currently supports gaussian outcomes only.")
  }
  
  prep <- .prepare_dbarts_xy(X = X, newX = newX)
  Y <- as.numeric(Y)
  
  ## Collect optional dbarts tuning arguments.
  dbarts_args <- list(...)
  
  ## Avoid nested parallelism:
  ## snowSuperLearner is already parallelizing across workers, so keep
  ## each dbarts fit single-threaded unless the caller overrides it.
  if (is.null(dbarts_args$nthread) && is.null(dbarts_args$n.threads)) {
    dbarts_args$nthread <- 1L
  }
  
  ## Retain trees so predict() on the fitted bart object is available.
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
  
  ## Use posterior mean predictions if exposed directly; otherwise
  ## collapse posterior draws to a single prediction per row.
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
## Prediction method for the fitted dbarts SuperLearner base learner
## ------------------------------------------------------------------
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
## Two-stage SuperLearner estimator with cross-validation
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
  
  ## --------------------------------------------------------------
  ## Fit the Stage-1 outcome model. As in the refactored super_one(),
  ## keep a dbarts-only fallback if some library learner fails.
  ## --------------------------------------------------------------
  all <- tryCatch(
    fit_super(sl_library),
    error = function(e) {
      fit_super("SL.dbarts")
    }
  )
  
  ## --------------------------------------------------------------
  ## Potential outcome predictions under A = 0
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
  ## Potential outcome predictions under A = 1
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
  ## Store first-stage nuisance estimates in the original data frame.
  ## We keep the original two-stage structure unchanged.
  ## --------------------------------------------------------------
  df$y0 <- as.numeric(pred_A0$pred)
  df$y1 <- as.numeric(pred_A1$pred)
  
  ## --------------------------------------------------------------
  ## Second-stage regression:
  ##   y ~ A + y0 + y1
  ##
  ## This preserves the original implementation exactly.
  ## --------------------------------------------------------------
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


# Example run:
two_stg = super_one_cv(df = datasets[[1]], cl = NULL)
two_stg



