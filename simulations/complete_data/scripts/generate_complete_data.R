ipak <- function(pkg) {
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) {
    install.packages(new.pkg, dependencies = TRUE, repos = "http://cran.rstudio.com/")
  }
  sapply(pkg, require, character.only = TRUE)
}

packages <- c("MASS", "foreach", "doParallel", "doRNG")
ipak(packages)

output_dir <- file.path("simulations", "complete_data", "datasets")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

expit <- function(x) {
  1 / (1 + exp(-x))
}

generate_complete_data <- function(n, target_r2, max_attempts = 50) {
  A <- c(rep(1, floor(n / 2)), rep(0, n - floor(n / 2)))
  sigma <- diag(4)

  attempt <- 1
  repeat {
    mat <- MASS::mvrnorm(n = n, mu = c(0, 0, 0, 0), Sigma = sigma)
    colnames(mat) <- c("z1", "z2", "z3", "z4")

    y_signal <- 210 + 50 * A + 27.4 * mat[, 1] + 13.7 * mat[, 2] + 13.7 * mat[, 3] + 13.7 * mat[, 4]
    x1 <- exp(mat[, 1] / 2)
    x2 <- (mat[, 2] / (1 + exp(mat[, 1]))) + 10
    x3 <- (((mat[, 1] * mat[, 3]) / 25) + 0.6) ^ 3
    x4 <- (mat[, 2] + mat[, 4] + 20) ^ 2
    pi <- expit(-mat[, 1] + 0.5 * mat[, 2] - 0.25 * mat[, 3] - 0.1 * mat[, 4])
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

  df <- data.frame(y, A, x1, x2, x3, x4, R, mat)
  meta <- data.frame(
    n = n,
    target_r2 = target_r2,
    achieved_r2 = round(achieved_r2, 2),
    achieved_r2_raw = achieved_r2,
    ss = sd,
    stringsAsFactors = FALSE
  )

  list(data = df, meta = meta)
}

format_r2 <- function(r2) {
  gsub("\\.", "p", sprintf("%.2f", r2))
}

# 1. Generate complete datasets across n and R^2 values for downstream analysis
reps <- 1000
sample_sizes <- c(200, 500, 1000, 2000, 10000)
target_r2s <- c(0.20, 0.40, 0.60, 0.80) # these are achieved by varying the noise level (\sigma_Z) in the data generating process
base_seed <- 12345

cores <- parallel::detectCores(logical = TRUE)
registerDoParallel(max(1, cores - 1))

for (n in sample_sizes) {
  for (target_r2 in target_r2s) {
    scenario_seed <- base_seed + n + as.integer(round(target_r2 * 100))
    doRNG::registerDoRNG(scenario_seed)

    results <- foreach(i = seq_len(reps)) %dopar% {
      res <- generate_complete_data(n, target_r2)
      res$meta$replicate <- i
      res
    }

    datasets <- lapply(results, function(x) x$data)
    metadata <- do.call(rbind, lapply(results, function(x) x$meta))

    r2_label <- format_r2(target_r2)
    file_base <- sprintf("complete_n%d_r2_%s", n, r2_label)
    save(
      datasets,
      metadata,
      file = file.path(output_dir, paste0(file_base, ".RData"))
    )
  }
}


# 2. Generate a large dataset for estimating model parameters with high precision: asymptotic performance check
large_n <- 5 * 10^7
large_seed_base <- 54321
run_large_data <- tolower(Sys.getenv("RUN_LARGE_DATA", "false")) == "true"

# Note: This may take a long time to run and produce a very large file, so it's gated behind an environment variable
run_large_data <- FALSE 
if (run_large_data) {
  for (target_r2 in target_r2s) {
    large_seed <- large_seed_base + as.integer(round(target_r2 * 100))
    set.seed(large_seed)

    res <- generate_complete_data(large_n, target_r2)
    res$meta$replicate <- 1
    res$meta$n <- large_n

    r2_label <- format_r2(target_r2)
    ss_label <- format_r2(res$meta$ss)
    file_base <- sprintf("complete_n%d_r2_%s_ss_%s", large_n, r2_label, ss_label)
    save(
      res$data,
      res$meta,
      file = file.path(output_dir, paste0(file_base, ".RData"))
    )
  }
}
