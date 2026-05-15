# This script mirrors the missing-outcomes generator but adds heterogeneous
# treatment effects via theta * (A * z1) for setting_four.

# Load packages
ipak <- function(pkg) {
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) {
    install.packages(new.pkg, dependencies = TRUE, repos = "http://cran.rstudio.com/")
  }
  sapply(pkg, require, character.only = TRUE)
}

packages <- c("MASS", "foreach", "doParallel", "doRNG")
ipak(packages)

output_root <- file.path("simulations", "missing_outcomes", "miss_datasets")
if (!dir.exists(output_root)) {
  dir.create(output_root, recursive = TRUE)
}

expit <- function(x) {
  1 / (1 + exp(-x))
}

# Missing outcome data generating mechanism function.
# Adds a heterogeneous effect term theta * (A * z1).
generate_missing_data_hetero <- function(n, target_r2, max_attempts = 50, eta0, eta1, theta, setting) {
  A <- c(rep(1, floor(n / 2)), rep(0, n - floor(n / 2)))
  sigma <- diag(4)

  attempt <- 1
  repeat {
    mat <- MASS::mvrnorm(n = n, mu = c(0, 0, 0, 0), Sigma = sigma)
    colnames(mat) <- c("z1", "z2", "z3", "z4")

    y_signal <- 210 + 50 * A + theta * (A * mat[, 1]) +
      27.4 * mat[, 1] + 13.7 * mat[, 2] + 13.7 * mat[, 3] + 13.7 * mat[, 4]
    x1 <- exp(mat[, 1] / 2)
    x2 <- (mat[, 2] / (1 + exp(mat[, 1]))) + 10
    x3 <- (((mat[, 1] * mat[, 3]) / 25) + 0.6) ^ 3
    x4 <- (mat[, 2] + mat[, 4] + 20) ^ 2
    pi <- expit(-mat[, 1] + eta0 * A + eta1 * A * mat[, 1] + 0.5 * mat[, 2] - 0.25 * mat[, 3] - 0.1 * mat[, 4])
    R <- rbinom(n = n, size = 1, prob = pi)

    e0 <- rnorm(n = n, mean = 0, sd = 1)
    r2_for_sd <- function(sd) {
      y <- y_signal + sd * e0
      fit <- lm(y ~ A + x1 + x2 + x3 + x4)
      summary(fit)$r.squared
    }

    ## retry the draw if the target R2 is unattainable even at ss = 0
    r2_at0 <- r2_for_sd(0)
    if (target_r2 <= r2_at0) {
      break
    }
    attempt <- attempt + 1
    if (attempt > max_attempts) {
      stop("Target R2 exceeds the maximum achievable after max_attempts draws.")
    }
  }

  sd_hi <- 1
  r2_hi <- r2_for_sd(sd_hi)
  while (r2_hi > target_r2) {
    sd_hi <- sd_hi * 2
    r2_hi <- r2_for_sd(sd_hi)
    if (sd_hi > 1e6) {
      stop("Failed to bracket target R2.")
    }
  }

  sd <- uniroot(
    function(sd) r2_for_sd(sd) - target_r2,
    lower = 0,
    upper = sd_hi,
    tol = 1e-10
  )$root

  achieved_r2 <- r2_for_sd(sd)
  y <- y_signal + sd * e0
  e = sd * e0

  df <- data.frame(y, A, x1, x2, x3, x4, R, mat, e)
  meta <- data.frame(
    n = n,
    target_r2 = target_r2,
    achieved_r2 = round(achieved_r2, 2),
    achieved_r2_raw = achieved_r2,
    ss = sd,
    setting = setting,
    theta = theta,
    stringsAsFactors = FALSE
  )

  list(data = df, meta = meta)
}

format_r2 <- function(r2) {
  gsub("\\.", "p", sprintf("%.2f", r2))
}

# Generate datasets across n and R^2 values.
reps <- 1000
sample_sizes <- c(200, 500, 1000, 2000, 10000)
target_r2s <- c(0.20, 0.40, 0.60, 0.80) # achieved by varying the noise level
base_seed <- 12345
settings <- list(
  setting_four = list(eta0 = 1, eta1 = 1, theta = 1)
)

cores <- parallel::detectCores(logical = TRUE)
registerDoParallel(max(1, cores - 1))

for (setting_name in names(settings)) {
  # create a subfolder per setting
  setting_dir <- file.path(output_root, setting_name)
  if (!dir.exists(setting_dir)) {
    dir.create(setting_dir, recursive = TRUE)
  }

  for (n in sample_sizes) {
    for (target_r2 in target_r2s) {
      # seed per setting/n/R^2 for reproducibility
      scenario_seed <- base_seed + n + as.integer(round(target_r2 * 100)) + match(setting_name, names(settings)) * 10000
      doRNG::registerDoRNG(scenario_seed)

      params <- settings[[setting_name]]
      results <- foreach(i = seq_len(reps)) %dopar% {
        res <- generate_missing_data_hetero(
          n,
          target_r2,
          eta0 = params$eta0,
          eta1 = params$eta1,
          theta = params$theta,
          setting = setting_name
        )
        res$meta$replicate <- i
        res
      }

      datasets <- lapply(results, function(x) x$data)
      metadata <- do.call(rbind, lapply(results, function(x) x$meta))

      if (length(datasets) != nrow(metadata)) {
        stop(sprintf(
          "Metadata row count mismatch for %s n=%d r2=%s: %d datasets vs %d metadata rows.",
          setting_name,
          n,
          format_r2(target_r2),
          length(datasets),
          nrow(metadata)
        ))
      }

      r2_label <- format_r2(target_r2)
      file_base <- sprintf("%s_n%d_r2_%s", setting_name, n, r2_label)
      save(
        datasets,
        metadata,
        file = file.path(setting_dir, paste0(file_base, ".RData"))
      )
    }
  }
}
