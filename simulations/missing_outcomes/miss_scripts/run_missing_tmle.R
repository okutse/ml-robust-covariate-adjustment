# Run missing-outcome TMLE procedure for one setting
# Inputs:
# - .RData files under simulations/missing_outcomes/miss_datasets/<setting>
# Outputs:
# - Per-scenario CSVs under simulations/missing_outcomes/miss_intermediate_results/<setting>/tmle/<model_spec>
# - Aggregated CSVs per model_spec and procedure
# - Checkpoints per model_spec
# - Logs under logs/missing_outcomes/<setting>_tmle_logs.txt
# Naming:
# - <setting>_<scenario>_<model_spec>.csv per scenario
# - <setting>_tmle_<model_spec>_aggregated.csv per model spec
# - <setting>_tmle_aggregated.csv across model specs

source(file.path("simulations", "missing_outcomes", "miss_scripts", "miss_runner.R"))

setting_name <- Sys.getenv("SETTING", "setting1")
model_specs <- c("m1", "m2")

run_procedure_for_setting(
  setting_name = setting_name,
  procedure_name = "tmle",
  model_specs = model_specs,
  use_parallel = TRUE,
  cores = parallel::detectCores(logical = TRUE)
)

# Example run:
# SETTING=setting1 DATA_SOURCE=local RESET_CHECKPOINT=true \
# Rscript simulations/missing_outcomes/miss_scripts/run_missing_tmle.R

# Slurm one-liner example (adjust partition/resources as needed):
# sbatch --job-name=miss_tmle --partition=gpu --gres=gpu:0 --nodes=1 \
#   --ntasks=1 --cpus-per-task=10 --mem=64G --time=48:00:00 \
#   --output=logs/missing_outcomes/tmle_%A.out \
#   --error=logs/missing_outcomes/tmle_%A.err \
#   --export=SETTING=setting1,DATA_SOURCE=local,RESET_CHECKPOINT=true \
#   --wrap="module load r/4.5.1 && Rscript simulations/missing_outcomes/miss_scripts/run_missing_tmle.R"
