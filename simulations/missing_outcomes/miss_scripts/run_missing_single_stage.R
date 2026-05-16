# Run missing-outcome single-stage procedure for one setting
# Inputs:
# - .RData files under simulations/missing_outcomes/miss_datasets/<setting>
# Outputs:
# - Per-scenario CSVs under simulations/missing_outcomes/miss_intermediate_results/<setting>/single_stage/<model_spec>
# - Aggregated CSVs per model_spec and procedure
# - Checkpoints per model_spec
# - Logs under logs/missing_outcomes/<setting>_single_stage_logs.txt
# Naming:
# - <setting>_<scenario>_<model_spec>.csv per scenario
# - <setting>_single_stage_<model_spec>_aggregated.csv per model spec
# - <setting>_single_stage_aggregated.csv across model specs

get_script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- args[grepl("^--file=", args)]
  if (length(file_arg) == 0) {
    return(NA_character_)
  }
  sub("^--file=", "", file_arg[1])
}

script_path <- get_script_path()
if (is.na(script_path) || !nzchar(script_path)) {
  stop("Unable to determine the path to run_missing_single_stage.R.")
}

project_root <- normalizePath(file.path(dirname(script_path), "..", "..", ".."), winslash = "/", mustWork = TRUE)
setwd(project_root)

renv_activate <- file.path(project_root, "renv", "activate.R")
if (file.exists(renv_activate)) {
  source(renv_activate)
}

# Fail fast in batch jobs if the project library is not ready before any analysis starts.
required_packages <- c(
  "parsnip",
  "ranger",
  "dbarts",
  "SuperLearner",
  "xgboost",
  "magrittr",
  "purrr",
  "rsample",
  "dplyr",
  "foreach",
  "doParallel",
  "nnls",
  "gam",
  "jsonlite"
)
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    "Missing required packages before processing begins: ",
    paste(missing_packages, collapse = ", "),
    ". Run `Rscript -e \"renv::restore()\"` from the repository root and resubmit the job."
  )
}

source(file.path(project_root, "simulations", "missing_outcomes", "miss_scripts", "miss_runner.R"))

setting_name <- Sys.getenv("SETTING", "setting_three") # can be changed via environment variable, e.g. when running on Slurm or locally with different settings
model_specs <- c("m1", "m2")

run_procedure_for_setting(
  setting_name = setting_name,
  procedure_name = "single_stage",
  model_specs = model_specs,
  use_parallel = TRUE,
  cores = parallel::detectCores(logical = TRUE)
)

# Example run:
# Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage.R
#
# Use the Zenodo DOI if the project-local miss_datasets folder is not present:
# zenodo_data_doi="https://doi.org/10.5281/zenodo.19393092" \
# Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage.R
#
# Force a local archive folder if you already unpacked Data.zip:
# SETTING=setting_three DATA_SOURCE=archive ARCHIVE_DATASETS_DIR=simulations/missing_outcomes/miss_datasets RESET_CHECKPOINT=true \
# Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage.R

# Slurm one-liner example (adjust partition/resources as needed):
# sbatch --job-name=miss_single_stage --partition=gpu --gres=gpu:0 --nodes=1 \
#   --ntasks=1 --cpus-per-task=10 --mem=64G --time=48:00:00 \
#   --output=logs/missing_outcomes/single_stage_%A.out \
#   --error=logs/missing_outcomes/single_stage_%A.err \
#   --export=SETTING=setting_three,DATA_SOURCE=local,RESET_CHECKPOINT=true \
#   --wrap="module load r/4.5.1 && Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage.R"
