# Helpers for the custom dbarts SuperLearner wrapper

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

.prepare_dbarts_xy <- function(X, newX = NULL) {
  X <- as.data.frame(X)
  X[] <- lapply(X, as.numeric)

  if (!is.null(newX)) {
    newX <- as.data.frame(newX)
    newX <- newX[, colnames(X), drop = FALSE]
    newX[] <- lapply(newX, as.numeric)
  }

  list(
    X_df = X,
    X_mat = as.matrix(X),
    newX_df = newX,
    newX_mat = if (!is.null(newX)) as.matrix(newX) else NULL
  )
}

SL.dbarts <- function(Y, X, newX, family, obsWeights, id, ...) {
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

  fit_obj <- do.call(
    dbarts::bart,
    c(
      list(
        x.train = prep$X_mat,
        y.train = Y,
        x.test = prep$newX_mat,
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
    fit = list(
      object = fit_obj,
      colnames = colnames(prep$X_df)
    )
  )

  class(out$fit) <- "SL.dbarts"
  out
}

predict.SL.dbarts <- function(object, newdata, ...) {
  newdata <- as.data.frame(newdata)
  newdata <- newdata[, object$colnames, drop = FALSE]
  newdata[] <- lapply(newdata, as.numeric)
  x_test <- as.matrix(newdata)

  pred_raw <- predict(object$object, newdata = x_test, ...)
  .reduce_dbarts_pred(pred_raw, n_obs = nrow(x_test))
}
