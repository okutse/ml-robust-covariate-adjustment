# Run missing-outcome single-stage DRML-BC procedure for one setting
# Inputs:
# - .RData files under simulations/missing_outcomes/miss_datasets/<setting>
# Outputs:
# - Per-scenario CSVs under simulations/missing_outcomes/miss_intermediate_results/<setting>/single_stage_drml_bc/<model_spec>
# - Aggregated CSVs per model_spec and procedure
# - Checkpoints per model_spec
# - Logs under logs/missing_outcomes/<setting>_single_stage_drml_bc_logs.txt
# Naming:
# - <setting>_<scenario>_<model_spec>.csv per scenario
# - <setting>_single_stage_drml_bc_<model_spec>_aggregated.csv per model spec
# - <setting>_single_stage_drml_bc_aggregated.csv across model specs

source(file.path("simulations", "missing_outcomes", "miss_scripts", "miss_runner.R"))

setting_name <- Sys.getenv("SETTING", "setting1")
model_specs <- c("m1", "m2")

run_procedure_for_setting(
  setting_name = setting_name,
  procedure_name = "single_stage_drml_bc",
  model_specs = model_specs,
  use_parallel = TRUE,
  cores = parallel::detectCores(logical = TRUE)
)

# Example run:
# SETTING=setting1 DATA_SOURCE=local RESET_CHECKPOINT=true \
# Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage_drml_bc.R
