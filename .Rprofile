source("renv/activate.R")
## Auto-activate renv when a lockfile is present
if (file.exists("renv.lock")) {
  tryCatch(
    renv::activate(),
    error = function(e) message("renv activation failed: ", e$message)
  )
}
