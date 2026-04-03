packages <- c("MASS", "foreach", "doParallel", "knitr", "kableExtra")

ipak <- function(pkg) {
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) {
    install.packages(new.pkg, dependencies = TRUE, repos = "http://cran.rstudio.com/")
  }
  sapply(pkg, require, character.only = TRUE)
}

ipak(packages)

input_dir <- file.path("simulations", "complete_data", "datasets")
output_dir <- file.path("simulations", "complete_data", "complete_data_results")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}


# 1. Estimate model params across n and R2 values to check how well we can 
# recover the true params using the analysts observed adjustment covariate vector X
fast_lm_stats <- function(df) {
  X <- cbind(1, df$A, df$x1, df$x2, df$x3, df$x4)
  y <- df$y
  fit <- lm.fit(X, y)
  n <- nrow(X)
  p <- ncol(X)
  rss <- sum(fit$residuals^2)
  tss <- sum((y - mean(y))^2)
  r2 <- 1 - (rss / tss)
  sigma <- sqrt(rss / (n - p))

  r <- qr.R(fit$qr)
  r_inv <- solve(r[seq_len(p), seq_len(p), drop = FALSE])
  xtx_inv <- tcrossprod(r_inv)
  se <- sqrt(diag(xtx_inv) * sigma^2)

  list(coef = fit$coefficients, se = se, r2 = r2, sigma = sigma)
}

collect_results <- function(file_path) {
  env <- new.env(parent = emptyenv())
  load(file_path, envir = env)
  if (!exists("datasets", envir = env) || !exists("metadata", envir = env)) {
    stop(paste("Missing datasets or metadata in", file_path))
  }

  datasets <- get("datasets", envir = env)
  metadata <- get("metadata", envir = env)

  if (length(datasets) != nrow(metadata)) {
    stop(paste("Mismatch between datasets and metadata in", file_path))
  }

  stats <- foreach(i = seq_along(datasets), .combine = rbind) %dopar% {
    s <- fast_lm_stats(datasets[[i]])
    c(s$coef, s$se, s$r2, s$sigma)
  }

  colnames(stats) <- c(
    "intercept_est", "A_est", "x1_est", "x2_est", "x3_est", "x4_est",
    "intercept_se", "A_se", "x1_se", "x2_se", "x3_se", "x4_se",
    "r2", "sigma_x"
  )

  avg_stats <- colMeans(stats)
  achieved_r2_raw_mean <- mean(metadata$achieved_r2_raw)
  data.frame(
    n = metadata$n[1],
    target_r2 = metadata$target_r2[1],
    achieved_r2 = round(achieved_r2_raw_mean, 2),
    achieved_r2_raw = achieved_r2_raw_mean,
    ss = mean(metadata$ss),
    intercept_est = avg_stats["intercept_est"],
    intercept_se = avg_stats["intercept_se"],
    A_est = avg_stats["A_est"],
    A_se = avg_stats["A_se"],
    x1_est = avg_stats["x1_est"],
    x1_se = avg_stats["x1_se"],
    x2_est = avg_stats["x2_est"],
    x2_se = avg_stats["x2_se"],
    x3_est = avg_stats["x3_est"],
    x3_se = avg_stats["x3_se"],
    x4_est = avg_stats["x4_est"],
    x4_se = avg_stats["x4_se"],
    r2 = avg_stats["r2"],
    sigma_x = avg_stats["sigma_x"],
    stringsAsFactors = FALSE
  )
}

input_files <- list.files(input_dir, pattern = "^complete_n\\d+_r2_\\d+p\\d+\\.RData$", full.names = TRUE)
if (length(input_files) == 0) {
  stop(paste("No scenario files found in", input_dir))
}

cores <- parallel::detectCores(logical = TRUE)
registerDoParallel(max(1, cores - 1))

avg_results <- do.call(rbind, lapply(input_files, collect_results))

format_est_se <- function(est, se) {
  sprintf("%.2f (%.2f)", est, se)
}

summary_by_r2 <- data.frame(
  n = avg_results$n,
  target_r2 = avg_results$target_r2,
  sigma_z = avg_results$ss,
  intercept = format_est_se(avg_results$intercept_est, avg_results$intercept_se),
  A = format_est_se(avg_results$A_est, avg_results$A_se),
  x1 = format_est_se(avg_results$x1_est, avg_results$x1_se),
  x2 = format_est_se(avg_results$x2_est, avg_results$x2_se),
  x3 = format_est_se(avg_results$x3_est, avg_results$x3_se),
  x4 = format_est_se(avg_results$x4_est, avg_results$x4_se),
  r2 = round(avg_results$r2, 2),
  sigma_x = round(avg_results$sigma_x, 2),
  stringsAsFactors = FALSE
)

# reorder columns for better presentation in the paper: arrange by n from smallest to largest and then by target_r2 from smallest to largest
summary_by_r2 <- summary_by_r2[order(summary_by_r2$n, summary_by_r2$target_r2), ]
rownames(summary_by_r2) <- NULL

write.csv(
  summary_by_r2,
  file = file.path(output_dir, "complete_data_model_params_using_X.csv"),
  row.names = FALSE
)

# create table1.tex and save in output_dir as table1.tex
model_parameters = summary_by_r2 %>% 
  dplyr::select(
    n, intercept, A, x1, x2, x3, x4,
    r2, sigma_x
  )

table1 <- kableExtra::kbl(
  model_parameters,
  format = "latex",
  booktabs = TRUE,
  linesep = '',
  escape = FALSE,
  caption = "Model parameter estimates and their standard errors (SEs) across different sample sizes for the complete data generating mechanism averaged across 1000 simulated datasets.$R^2$ corresponds to the proportion of variance explained based on a linear regression adjustment model that uses the analysts observed covariate vector $X$. $\\sigma_X$ is the corresponding average standard deviation of the error term in the analysts outcome model. Results provide a baseline on the recoverability of the true outcome model parameter estimates without the additional missing outcome data complication.",
  align = c("lcccccccc"),
  col.names = c(
    "$n$","Intercept (SE)", "$A$ (SE)", "$X_1$ (SE)", "$X_2$ (SE)", "$X_3$ (SE)", "$X_4$ (SE)",
    "$R^2$", "$\\sigma_X$")
) %>% 
  kableExtra::collapse_rows(columns = 1, valign = "middle", latex_hline = "none") %>%
  kableExtra::row_spec(c(4, 8, 12, 16, 20), hline_after = TRUE) %>% 
  kableExtra::kable_styling(latex_options = c("hold_position", "scale_down"))

writeLines(table1, con = file.path(output_dir, "table1.tex"))








# 3. Estimate model parameters under large sample sizes and varying R to check asymptotic performance
# get_env_object <- function(env, candidates) {
#   for (name in candidates) {
#     if (exists(name, envir = env)) {
#       return(get(name, envir = env))
#     }
#   }
#   NULL
# }

# parse_large_filename <- function(file_path) {
#   base <- basename(file_path)
#   m <- regexec("complete_n(\\d+)_r2_(\\d+p\\d+)_ss_(\\d+p\\d+)", base)
#   parts <- regmatches(base, m)[[1]]
#   if (length(parts) < 4) {
#     return(NULL)
#   }
#   r2 <- as.numeric(gsub("p", ".", parts[3]))
#   ss <- as.numeric(gsub("p", ".", parts[4]))
#   list(n = as.numeric(parts[2]), target_r2 = r2, ss = ss)
# }

# collect_large_result <- function(file_path) {
#   env <- new.env(parent = emptyenv())
#   load(file_path, envir = env)

#   df <- get_env_object(env, c("data", "dataset", "res$data", "res.dataset"))
#   if (is.null(df)) {
#     stop(paste("Missing data object in", file_path))
#   }

#   meta <- get_env_object(env, c("meta", "metadata", "res$meta", "res.metadata"))
#   if (is.null(meta)) {
#     parsed <- parse_large_filename(file_path)
#     if (is.null(parsed)) {
#       stop(paste("Missing metadata and unable to parse", file_path))
#     }
#     meta <- data.frame(
#       n = parsed$n,
#       target_r2 = parsed$target_r2,
#       ss = parsed$ss,
#       stringsAsFactors = FALSE
#     )
#   }

#   stats <- fast_lm_stats(df)
#   data.frame(
#     n = meta$n[1],
#     target_r2 = meta$target_r2[1],
#     ss = meta$ss[1],
#     intercept_est = stats$coef[1],
#     intercept_se = stats$se[1],
#     A_est = stats$coef[2],
#     A_se = stats$se[2],
#     x1_est = stats$coef[3],
#     x1_se = stats$se[3],
#     x2_est = stats$coef[4],
#     x2_se = stats$se[4],
#     x3_est = stats$coef[5],
#     x3_se = stats$se[5],
#     x4_est = stats$coef[6],
#     x4_se = stats$se[6],
#     r2 = stats$r2,
#     sigma_x = stats$sigma,
#     stringsAsFactors = FALSE
#   )
# }

# large_files <- list.files(
#   input_dir,
#   pattern = "^complete_n50000000_r2_\\d+p\\d+_ss_\\d+p\\d+\\.RData$",
#   full.names = TRUE
# )

# if (length(large_files) > 0) {
#   large_results <- foreach(file_path = large_files, .combine = rbind) %dopar% {
#     collect_large_result(file_path)
#   }

#   write.csv(
#     large_results,
#     file = file.path(output_dir, "complete_data_model_params_large_n.csv"),
#     row.names = FALSE
#   )

#   summary_large <- data.frame(
#     n = large_results$n,
#     sigma_z = large_results$ss,
#     target_r2 = large_results$target_r2,
#     intercept = format_est_se(large_results$intercept_est, large_results$intercept_se),
#     A = format_est_se(large_results$A_est, large_results$A_se),
#     x1 = format_est_se(large_results$x1_est, large_results$x1_se),
#     x2 = format_est_se(large_results$x2_est, large_results$x2_se),
#     x3 = format_est_se(large_results$x3_est, large_results$x3_se),
#     x4 = format_est_se(large_results$x4_est, large_results$x4_se),
#     r2 = round(large_results$r2, 2),
#     sigma_x = round(large_results$sigma_x, 2),
#     stringsAsFactors = FALSE
#   )

#   make_latex_table(
#     summary_large,
#     panel_col = "sigma_z",
#     file_path = file.path(output_dir, "table1_sup.tex"),
#     caption = "Large-n model parameters by sigma_z"
#   )
# }












