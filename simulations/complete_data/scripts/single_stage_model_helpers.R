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




# 6. Super Learner (SL)
super_one <- function(df = NULL, cl = NULL){
  ## allow serial runs when no cluster is provided
  if (is.null(cl)) {
    all <- SuperLearner::SuperLearner(
      Y = as.numeric(df[, 1]),
      X = data.frame(df[, c(2, 3, 4, 5, 6)]),
      cvControl = list(V = 10), # number of folds for CV.SuperLearner
      family = gaussian(),
      SL.library = c("SL.ranger", "SL.lm", "SL.bartMachine", "SL.xgboost"),
      method = "method.NNLS")
  } else {
    all <- SuperLearner::snowSuperLearner(
      Y = as.numeric(df[, 1]),
      X = data.frame(df[, c(2, 3, 4, 5, 6)]),
      cvControl = list(V = 10), # number of folds for CV.SuperLearner
      family = gaussian(),
      SL.library = c("SL.ranger", "SL.lm", "SL.bartMachine", "SL.xgboost"),
      method = "method.NNLS",
      cluster = cl)
  }
  
  ## set A = 0 and generate predictions for everyone
  df_A0 <- data.frame(df[, c(2, 3, 4, 5, 6)])
  df_A0$A <- 0
  pred_A0 <- predict.SuperLearner(object = all, newdata = df_A0, onlySL = TRUE)
  
## set A = 1 and generate predictions for everyone
  df_A1 <- data.frame(df[, c(2, 3, 4, 5, 6)])
  df_A1$A <- 1
  pred_A1 <- predict.SuperLearner(object = all, newdata = df_A1, onlySL = TRUE)
## compute the ATE
  ATE_adjusted = mean(pred_A1$pred - pred_A0$pred)
## compute the bias
  bias_adjusted = ATE_adjusted - 50
## return the results as a data frame
  rslt = data.frame("ATE_adjusted" = ATE_adjusted, "bias_adjusted" = bias_adjusted)
  return(rslt)
}


# 7. XGBoost
xgboost_one <- function(df = NULL, min_n = 10, tree_depth = 9, learn_rate = 0.02, loss_reduction = 0){ 
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



# runs all single-stage methods on the list of datasets and returns averaged results
run_single_stage <- function(datasets, metadata, methods = NULL, use_parallel = TRUE, cores = NULL, cl = NULL){
  if (length(datasets) == 0) {
    stop("No datasets supplied.")
  }
  if (length(datasets) != nrow(metadata)) {
    stop("Mismatch between datasets and metadata.")
  }
  if (is.null(methods)) {
    methods <- get_single_stage_registry()
  }
  if (is.null(cores)) {
    cores <- parallel::detectCores(logical = TRUE)
  }

  if (use_parallel) {
    doParallel::registerDoParallel(max(1, cores - 1))
  } else {
    doParallel::registerDoParallel(1)
  }
  
# helper function to run a single method across all datasets and compute the average metrics
  run_method <- function(method){
    metrics <- foreach::foreach(i = seq_along(datasets), .combine = rbind) %dopar% {
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
