if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "http://cran.rstudio.com/")
}

if (file.exists("renv.lock")) {
  message("renv.lock already exists; run renv::snapshot() to update it.")
} else {
  renv::init(bare = TRUE)
  renv::snapshot(prompt = FALSE)
}
