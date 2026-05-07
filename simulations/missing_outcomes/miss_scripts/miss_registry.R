# Registry and procedure helpers for missing-outcome analyses

source(file.path("simulations", "missing_outcomes", "miss_model_helpers", "sl_dbarts_helpers.R"))

# Default CF settings for missing-outcome procedures (can be overridden upstream).
if (!exists("CF_FOLDS")) {
  CF_FOLDS <- 2
}

# Ensure the pipe operator is available when helpers are sourced directly.
if (!exists("%>%")) {
  `%>%` <- magrittr::`%>%`
}

bound_prob <- function(x, eps = 1e-6) {
  pmin(pmax(x, eps), 1 - eps)
}

crossfit_outcome_predictions <- function(df, outcome, covariates, model, folds = CF_FOLDS, ...) {
  df_cf <- as.data.frame(df)
  df_cf$.row_id <- seq_len(nrow(df_cf))

  fold_obj <- rsample::vfold_cv(df_cf, v = folds)
  mu0 <- rep(NA_real_, nrow(df_cf))
  mu1 <- rep(NA_real_, nrow(df_cf))

  for (split in fold_obj$splits) {
    train_dat <- rsample::analysis(split)
    assess_dat <- rsample::assessment(split)

    train_obs <- train_dat[train_dat$R == 1, , drop = FALSE]
    if (nrow(train_obs) == 0) {
      stop("No observed outcomes (R == 1) in training fold.")
    }

    fit_outcome <- model$fit(train_obs, outcome, covariates, ...)

    assess_A0 <- assess_dat
    assess_A0$A <- 0
    assess_A1 <- assess_dat
    assess_A1$A <- 1

    assess_dat_fac <- assess_dat
    assess_dat_fac$A <- factor(assess_dat_fac$A, levels = levels(train_ps$A))
    assess_A0_fac <- assess_A0
    assess_A0_fac$A <- factor(assess_A0_fac$A, levels = levels(train_ps$A))
    assess_A1_fac <- assess_A1
    assess_A1_fac$A <- factor(assess_A1_fac$A, levels = levels(train_ps$A))

    idx <- assess_dat$.row_id
    mu0[idx] <- model$predict(fit_outcome, assess_A0, ...)
    mu1[idx] <- model$predict(fit_outcome, assess_A1, ...)
  }

  if (anyNA(mu0) || anyNA(mu1)) {
    stop("Cross-fitting failed: some out-of-fold predictions are missing.")
  }

  list(mu0 = mu0, mu1 = mu1, n_obs = sum(df$R == 1))
}

crossfit_dr_nuisance <- function(df, outcome, covariates, model, folds = CF_FOLDS, include_missingness = TRUE, ...) {
  df_cf <- as.data.frame(df)
  df_cf$.row_id <- seq_len(nrow(df_cf))
  x_covars <- setdiff(covariates, "A")

  fold_obj <- rsample::vfold_cv(df_cf, v = folds)
  mu0 <- rep(NA_real_, nrow(df_cf))
  mu1 <- rep(NA_real_, nrow(df_cf))
  e_hat <- rep(NA_real_, nrow(df_cf))
  pi1 <- rep(NA_real_, nrow(df_cf))
  pi0 <- rep(NA_real_, nrow(df_cf))

  for (split in fold_obj$splits) {
    train_dat <- rsample::analysis(split)
    assess_dat <- rsample::assessment(split)

    train_obs <- train_dat[train_dat$R == 1, , drop = FALSE]
    if (nrow(train_obs) == 0) {
      stop("No observed outcomes (R == 1) in training fold.")
    }

    fit_outcome <- model$fit(
      df_obs = train_obs,
      outcome = outcome,
      covariates = covariates,
      ...
    )

    train_ps <- train_dat
    train_ps$A <- as.factor(train_ps$A)
    ps_formula <- build_formula("A", x_covars)
    ps_mod <- parsnip::logistic_reg() %>%
      parsnip::set_mode("classification") %>%
      parsnip::set_engine("glm") %>%
      parsnip::fit(ps_formula, data = train_ps)

    if (include_missingness) {
      train_obs_mod <- train_ps
      train_obs_mod$R <- as.factor(train_obs_mod$R)
      obs_formula <- build_formula("R", c("A", x_covars))
      obs_mod <- parsnip::logistic_reg() %>%
        parsnip::set_mode("classification") %>%
        parsnip::set_engine("glm") %>%
        parsnip::fit(obs_formula, data = train_obs_mod)
    }

    assess_A0 <- assess_dat
    assess_A0$A <- 0
    assess_A1 <- assess_dat
    assess_A1$A <- 1

    idx <- assess_dat$.row_id
    mu0[idx] <- model$predict(fit_outcome, assess_A0, ...)
    mu1[idx] <- model$predict(fit_outcome, assess_A1, ...)
    e_hat[idx] <- as.numeric(predict(ps_mod, new_data = assess_dat, type = "prob")$.pred_1)

    if (include_missingness) {
      pi1[idx] <- as.numeric(predict(obs_mod, new_data = assess_A1_fac, type = "prob")$.pred_1)
      pi0[idx] <- as.numeric(predict(obs_mod, new_data = assess_A0_fac, type = "prob")$.pred_1)
    } else {
      pi1[idx] <- 1
      pi0[idx] <- 1
    }
  }

  e_hat <- bound_prob(e_hat)
  pi1 <- bound_prob(pi1)
  pi0 <- bound_prob(pi0)

  if (anyNA(mu0) || anyNA(mu1) || anyNA(e_hat) || anyNA(pi1) || anyNA(pi0)) {
    stop("Cross-fitting failed: some out-of-fold nuisance predictions are missing.")
  }

  list(
    mu0 = mu0,
    mu1 = mu1,
    e_hat = e_hat,
    pi1 = pi1,
    pi0 = pi0,
    n_obs = sum(df$R == 1)
  )
}

compute_dr_eif <- function(df, outcome, mu0, mu1, e_hat, pi1, pi0, tau_hat) {
  y_obs <- ifelse(df$R == 1, df[[outcome]], 0)
  term1 <- (df$R / pi1) * (df$A) * (y_obs - mu1) / e_hat
  term0 <- (df$R / pi0) * (1 - df$A) * (y_obs - mu0) / (1 - e_hat)
  mu1 - mu0 - tau_hat + term1 - term0
}

build_formula <- function(outcome, covariates, interactions = NULL) {
  rhs <- paste(covariates, collapse = " + ")
  if (!is.null(interactions) && length(interactions) > 0) {
    rhs <- paste(rhs, paste(interactions, collapse = " + "), sep = " + ")
  }
  as.formula(paste(outcome, "~", rhs))
}

get_covariate_registry <- function() {
  list(
    m1 = function(df) c("A", "x1", "x2", "x3", "x4"),
    m2 = function(df) c("A", "x2", "x3", "x4"),
    correct = function(df) c("A", "z1", "z2", "z3", "z4")
  )
}

resolve_covariates <- function(model_spec, covariate_registry, df) {
  cov_fn <- covariate_registry[[model_spec]]
  if (is.null(cov_fn)) {
    stop(paste("Unknown model_spec:", model_spec))
  }
  cov_fn(df)
}

get_model_registry <- function() {
  list(
    unadj = list(
      name = "unadj",
      fit = function(df_obs, outcome, covariates, ...) NULL,
      predict = function(fit, newdata, ...) NULL,
      direct = TRUE
    ),
    lm = list(
      name = "lm",
      fit = function(df_obs, outcome, covariates, ...) {
        formula <- build_formula(outcome, covariates)
        lm(formula, data = df_obs)
      },
      predict = function(fit, newdata, ...) {
        as.numeric(predict(fit, newdata))
      }
    ),
    lm_interact = list(
      name = "lm_interact",
      fit = function(df_obs, outcome, covariates, ...) {
        interactions <- paste0("A*", setdiff(covariates, "A"))
        formula <- build_formula(outcome, covariates, interactions)
        lm(formula, data = df_obs)
      },
      predict = function(fit, newdata, ...) {
        as.numeric(predict(fit, newdata))
      }
    ),
    rf = list(
      name = "rf",
      fit = function(df_obs, outcome, covariates, ...) {
        formula <- build_formula(outcome, covariates)
        rand_forest(trees = 500) %>%
          set_mode("regression") %>%
          set_engine("ranger") %>%
          fit(formula = formula, data = df_obs)
      },
      predict = function(fit, newdata, ...) {
        as.numeric(predict(fit, newdata)$.pred)
      }
    ),
    bart = list(
      name = "bart",
      fit = function(df_obs, outcome, covariates, ...) {
        formula <- build_formula(outcome, covariates)
        parsnip::bart(mode = "regression") %>%
          set_engine("dbarts") %>%
          fit(formula = formula, data = df_obs)
      },
      predict = function(fit, newdata, ...) {
        as.numeric(predict(fit, newdata)$.pred)
      }
    ),
    xgboost = list(
      name = "xgboost",
      fit = function(df_obs, outcome, covariates, ...) {
        formula <- build_formula(outcome, covariates)
        parsnip::boost_tree(
          mode = "regression",
          trees = 1000,
          min_n = 10,
          tree_depth = 9,
          learn_rate = 0.02,
          loss_reduction = 0
        ) %>%
          set_engine("xgboost", objective = "reg:squarederror") %>%
          fit(formula = formula, data = df_obs)
      },
      predict = function(fit, newdata, ...) {
        as.numeric(predict(fit, newdata)$.pred)
      }
    ),
    super = list(
      name = "super",
      fit = function(df_obs, outcome, covariates, ...) {
        X <- df_obs[, covariates, drop = FALSE]
        X[] <- lapply(X, as.numeric)
        Y <- as.numeric(df_obs[[outcome]])
        fit <- SuperLearner::SuperLearner(
          Y = Y,
          X = X,
          cvControl = list(V = 10),
          family = gaussian(),
          # Exclude SL.dbarts: setting_one_single_stage_logs shows ~5770.95s per scenario with BART in the stack.
          SL.library = c("SL.ranger", "SL.lm", "SL.xgboost"),
          method = "method.NNLS"
        )
        fit$covariates <- covariates
        fit
      },
      predict = function(fit, newdata, ...) {
        required_cols <- colnames(fit$X)
        if (is.null(required_cols) || length(required_cols) == 0) {
          required_cols <- fit$covariates
        }
        missing_cols <- setdiff(required_cols, colnames(newdata))
        if (length(missing_cols) > 0) {
          stop(sprintf(
            "super predictor missing required columns: %s",
            paste(missing_cols, collapse = ", ")
          ))
        }

        X_new <- newdata[, required_cols, drop = FALSE]
        X_new[] <- lapply(X_new, as.numeric)
        as.numeric(predict.SuperLearner(fit, newdata = X_new, onlySL = TRUE)$pred)
      }
    ),
    correct_model = list(
      name = "correct_model",
      fit = function(df_obs, outcome, covariates, ...) {
        formula <- build_formula(outcome, covariates)
        lm(formula, data = df_obs)
      },
      predict = function(fit, newdata, ...) {
        as.numeric(predict(fit, newdata))
      }
    )
  )
}

get_procedure_registry <- function() {
  list(
    single_stage = list(
      name = "single_stage",
      use_eif_var = FALSE,
      run = function(df, outcome, covariates, model, folds = CF_FOLDS, ...) {
        df_obs <- df[df$R == 1, , drop = FALSE]
        if (nrow(df_obs) == 0) {
          stop("No observed outcomes (R == 1) in dataset.")
        }

        if (!is.null(model$direct) && model$direct) {
          estimate <- mean(df_obs[[outcome]][df_obs$A == 1]) -
            mean(df_obs[[outcome]][df_obs$A == 0])
        } else {
          preds <- crossfit_outcome_predictions(
            df = df,
            outcome = outcome,
            covariates = covariates,
            model = model,
            folds = folds,
            ...
          )
          estimate <- mean(preds$mu1 - preds$mu0)
        }

        list(
          estimate = estimate,
          bias = estimate - 50,
          n_obs = sum(df$R == 1),
          eif_var = NA_real_,
          eif_se = NA_real_
        )
      }
    ),
    # two-stage procedure of Cohen and Fogarty (2024) with ML 
    two_stage = list(
      name = "two_stage",
      use_eif_var = FALSE,
      run = function(df, outcome, covariates, model, folds = CF_FOLDS, ...) {
        df_obs <- df[df$R == 1, , drop = FALSE]
        if (nrow(df_obs) == 0) {
          stop("No observed outcomes (R == 1) in dataset.")
        }

        preds <- crossfit_outcome_predictions(
          df = df,
          outcome = outcome,
          covariates = covariates,
          model = model,
          folds = folds,
          ...
        )

        stage2_df <- df
        stage2_df$y0 <- preds$mu0
        stage2_df$y1 <- preds$mu1
        stage2_obs <- stage2_df[stage2_df$R == 1, , drop = FALSE]
        outM <- glm(y ~ A + y0 + y1, data = stage2_obs)
        estimate <- unname(coef(outM)["A"])

        list(
          estimate = estimate,
          bias = estimate - 50,
          n_obs = nrow(df_obs),
          eif_var = NA_real_,
          eif_se = NA_real_
        )
      }
    ),
    # single-stage cross-fit AIPW with ML nuisance estimation (DR-ML)
    single_stage_drml = list(
      name = "single_stage_drml",
      use_eif_var = TRUE,
      run = function(df, outcome, covariates, model, folds = CF_FOLDS, ...) {
        nuis <- crossfit_dr_nuisance(
          df = df,
          outcome = outcome,
          covariates = covariates,
          model = model,
          folds = folds,
          include_missingness = FALSE,
          ...
        )

        mA <- ifelse(df$A == 1, nuis$mu1, nuis$mu0)
        y_obs <- ifelse(df$R == 1, df[[outcome]], 0)
        aug <- df$R / ifelse(df$A == 1, nuis$pi1, nuis$pi0) *
          (df$A - nuis$e_hat) / (nuis$e_hat * (1 - nuis$e_hat)) *
          (y_obs - mA)
        estimate <- mean(nuis$mu1 - nuis$mu0 + aug)

        eif <- compute_dr_eif(
          df = df,
          outcome = outcome,
          mu0 = nuis$mu0,
          mu1 = nuis$mu1,
          e_hat = nuis$e_hat,
          pi1 = nuis$pi1,
          pi0 = nuis$pi0,
          tau_hat = estimate
        )
        eif_var <- stats::var(eif)
        eif_se <- sqrt(eif_var / nrow(df))

        list(
          estimate = estimate,
          bias = estimate - 50,
          n_obs = nuis$n_obs,
          eif_var = eif_var,
          eif_se = eif_se
        )
      }
    ),
    # single-stage cross-fit AIPW with ML nuisance estimation and missing data bias correction (DR-ML-BC)
    single_stage_drml_bc = list(
      name = "single_stage_drml_bc",
      use_eif_var = TRUE,
      run = function(df, outcome, covariates, model, folds = CF_FOLDS, ...) {
        nuis <- crossfit_dr_nuisance(
          df = df,
          outcome = outcome,
          covariates = covariates,
          model = model,
          folds = folds,
          include_missingness = TRUE,
          ...
        )

        mA <- ifelse(df$A == 1, nuis$mu1, nuis$mu0)
        y_obs <- ifelse(df$R == 1, df[[outcome]], 0)
        aug <- df$R / ifelse(df$A == 1, nuis$pi1, nuis$pi0) *
          (df$A - nuis$e_hat) / (nuis$e_hat * (1 - nuis$e_hat)) *
          (y_obs - mA)
        estimate <- mean(nuis$mu1 - nuis$mu0 + aug)

        eif <- compute_dr_eif(
          df = df,
          outcome = outcome,
          mu0 = nuis$mu0,
          mu1 = nuis$mu1,
          e_hat = nuis$e_hat,
          pi1 = nuis$pi1,
          pi0 = nuis$pi0,
          tau_hat = estimate
        )
        eif_var <- stats::var(eif)
        eif_se <- sqrt(eif_var / nrow(df))

        list(
          estimate = estimate,
          bias = estimate - 50,
          n_obs = nuis$n_obs,
          eif_var = eif_var,
          eif_se = eif_se
        )
      }
    ),
    # Targeted maximum likelihood estimation (TMLE) with ML nuisance estimation and missing data bias correction (TMLE-ML-BC)
    tmle = list(
      name = "tmle",
      use_eif_var = TRUE,
      run = function(df, outcome, covariates, model, folds = CF_FOLDS, ...) {
        df_cf <- as.data.frame(df)
        df_cf$.row_id <- seq_len(nrow(df_cf))
        x_covars <- setdiff(covariates, "A")

        fold_obj <- rsample::vfold_cv(df_cf, v = folds)
        q1_star <- rep(NA_real_, nrow(df_cf))
        q0_star <- rep(NA_real_, nrow(df_cf))
        e_hat <- rep(NA_real_, nrow(df_cf))
        pi1 <- rep(NA_real_, nrow(df_cf))
        pi0 <- rep(NA_real_, nrow(df_cf))

        for (split in fold_obj$splits) {
          train_dat <- rsample::analysis(split)
          assess_dat <- rsample::assessment(split)

          train_obs <- train_dat[train_dat$R == 1, , drop = FALSE]
          if (nrow(train_obs) == 0) {
            stop("No observed outcomes (R == 1) in training fold.")
          }

          # 1) Initial outcome regression Q(A, X) on observed outcomes
          q_mod <- model$fit(
            df_obs = train_obs,
            outcome = outcome,
            covariates = covariates,
            ...
          )

          # 2) Treatment mechanism e(X)
          train_ps <- train_dat
          train_ps$A <- as.factor(train_ps$A)
          train_ps$R <- as.factor(train_ps$R)
          gA_formula <- build_formula("A", x_covars)
          gA_mod <- parsnip::logistic_reg() %>%
            parsnip::set_mode("classification") %>%
            parsnip::set_engine("glm") %>%
            parsnip::fit(gA_formula, data = train_ps)

          # 3) Outcome-observation mechanism pi(A, X)
          gR_formula <- build_formula("R", c("A", x_covars))
          gR_mod <- parsnip::logistic_reg() %>%
            parsnip::set_mode("classification") %>%
            parsnip::set_engine("glm") %>%
            parsnip::fit(gR_formula, data = train_ps)

          # 4) Initial predictions on held-out fold
          assess1 <- assess_dat
          assess1$A <- 1
          assess0 <- assess_dat
          assess0$A <- 0

          assess_dat_fac <- assess_dat
          assess_dat_fac$A <- factor(assess_dat_fac$A, levels = levels(train_ps$A))
          assess1_fac <- assess1
          assess1_fac$A <- factor(assess1_fac$A, levels = levels(train_ps$A))
          assess0_fac <- assess0
          assess0_fac$A <- factor(assess0_fac$A, levels = levels(train_ps$A))

          Q1_init <- as.numeric(model$predict(q_mod, assess1, ...))
          Q0_init <- as.numeric(model$predict(q_mod, assess0, ...))
          QA_init <- as.numeric(model$predict(q_mod, assess_dat, ...))

          e_assess <- as.numeric(predict(gA_mod, new_data = assess_dat, type = "prob")$.pred_1)
          piA_assess <- as.numeric(predict(gR_mod, new_data = assess_dat_fac, type = "prob")$.pred_1)
          pi1_assess <- as.numeric(predict(gR_mod, new_data = assess1_fac, type = "prob")$.pred_1)
          pi0_assess <- as.numeric(predict(gR_mod, new_data = assess0_fac, type = "prob")$.pred_1)


          # bound all probabilities away from 0 and 1 to avoid positivity issues in clever covariates and targeting step
          e_assess <- bound_prob(e_assess)
          piA_assess <- bound_prob(piA_assess)
          pi1_assess <- bound_prob(pi1_assess)
          pi0_assess <- bound_prob(pi0_assess)

          # 5) Clever covariates for treatment-specific means
          H1 <- assess_dat$R * (assess_dat$A == 1) / (piA_assess * e_assess)
          H0 <- assess_dat$R * (assess_dat$A == 0) / (piA_assess * (1 - e_assess))

          # 6) Targeting step (Gaussian fluctuation) using observed outcomes only
          obs_idx <- which(assess_dat$R == 1)
          fluct_df <- data.frame(
            y = assess_dat[[outcome]][obs_idx],
            QA = QA_init[obs_idx],
            H1 = H1[obs_idx],
            H0 = H0[obs_idx]
          )

          eps_fit <- stats::lm(y ~ -1 + offset(QA) + H1 + H0, data = fluct_df)
          eps_hat <- stats::coef(eps_fit)

          eps1 <- if ("H1" %in% names(eps_hat)) eps_hat["H1"] else 0
          eps0 <- if ("H0" %in% names(eps_hat)) eps_hat["H0"] else 0

          # 7) Update counterfactual regressions on held-out fold
          idx <- assess_dat$.row_id
          q1_star[idx] <- Q1_init + eps1 / pi1_assess
          q0_star[idx] <- Q0_init + eps0 / pi0_assess
          e_hat[idx] <- e_assess
          pi1[idx] <- pi1_assess
          pi0[idx] <- pi0_assess
        }

        if (anyNA(q1_star) || anyNA(q0_star) || anyNA(e_hat) || anyNA(pi1) || anyNA(pi0)) {
          stop("Cross-fitting failed: some out-of-fold TMLE predictions are missing.")
        }

        estimate <- mean(q1_star - q0_star)

        eif <- compute_dr_eif(
          df = df,
          outcome = outcome,
          mu0 = q0_star,
          mu1 = q1_star,
          e_hat = e_hat,
          pi1 = pi1,
          pi0 = pi0,
          tau_hat = estimate
        )
        eif_var <- stats::var(eif)
        eif_se <- sqrt(eif_var / nrow(df))

        list(
          estimate = estimate,
          bias = estimate - 50,
          n_obs = sum(df$R == 1),
          eif_var = eif_var,
          eif_se = eif_se
        )
      }
    )
  )
}
