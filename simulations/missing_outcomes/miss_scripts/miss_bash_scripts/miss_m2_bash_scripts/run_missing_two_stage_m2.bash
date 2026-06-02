#!/bin/bash
#SBATCH --job-name=miss_setting_four_two_stage_m2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amos_okutse@brown.edu

set -euo pipefail
module load r/4.5.1

setting="${SETTING:-setting_four}"
procedure="two_stage"
model_spec="${MODEL_SPEC:-m2}"
job_name="${SLURM_JOB_NAME:-miss_${setting}_${procedure}_${model_spec}}"

# Resolve repository root robustly across path aliases (/users vs /oscar/home).
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script_root_guess="$(cd "$script_dir/../../../../.." && pwd)"
submit_root="${SLURM_SUBMIT_DIR:-$PWD}"

project_root=""
for candidate in "$script_root_guess" "$submit_root"; do
	if [[ -f "$candidate/renv/activate.R" && -f "$candidate/simulations/missing_outcomes/miss_scripts/miss_runner.R" ]]; then
		project_root="$candidate"
		break
	fi
done

if [[ -z "$project_root" ]]; then
	echo "Unable to locate project root with renv/activate.R and miss_runner.R" >&2
	echo "Checked: $script_root_guess and $submit_root" >&2
	exit 1
fi

cd "$project_root"
echo "Using project_root=$project_root"

export SETTING="$setting"
export MODEL_SPEC="$model_spec"
export DATA_SOURCE="${DATA_SOURCE:-archive}"
export ARCHIVE_DATASETS_DIR="${ARCHIVE_DATASETS_DIR:-$project_root/simulations/missing_outcomes/archives/zenodo/zenodo_datasets.zip}"

log_dir="$project_root/logs/missing_outcomes/$setting"
mkdir -p "$log_dir"
run_log="$log_dir/${job_name}_${SLURM_JOB_ID:-local}.log"

echo "Running procedure=$procedure setting=$setting model=$model_spec"
echo "Run log: $run_log"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
	echo "DRY_RUN=true; skipping renv restore and analysis execution."
	exit 0
fi

# Activate renv first, then restore the project library before any analysis code runs.
Rscript -e 'source("renv/activate.R"); renv::restore(prompt = FALSE)'

Rscript - <<'RSCRIPT' 2>&1 | tee -a "$run_log"
source("renv/activate.R")
source("simulations/missing_outcomes/miss_scripts/miss_runner.R")

setting_name <- Sys.getenv("SETTING", "setting_four")
model_spec <- Sys.getenv("MODEL_SPEC", "m2")
model_specs <- trimws(strsplit(model_spec, ",", fixed = TRUE)[[1]])
model_specs <- model_specs[nzchar(model_specs)]
if (length(model_specs) == 0) {
  stop("MODEL_SPEC must contain at least one non-empty value.")
}

run_procedure_for_setting(
  setting_name = setting_name,
  procedure_name = "two_stage",
  model_specs = model_specs,
  use_parallel = TRUE
)
RSCRIPT

# Example submission:
# setting=setting_four
# repo=/users/aokutse/ml-robust-covariate-adjustment
# sbatch --chdir="$repo" \
#   --job-name="miss_${setting}_two_stage_m2" \
#   --output="$repo/logs/missing_outcomes/$setting/miss_${setting}_two_stage_m2_%j.out" \
#   --error="$repo/logs/missing_outcomes/$setting/miss_${setting}_two_stage_m2_%j.err" \
#   --export=SETTING=$setting,MODEL_SPEC=m2,DATA_SOURCE=archive,ARCHIVE_DATASETS_DIR=$repo/simulations/missing_outcomes/archives/zenodo/zenodo_datasets.zip \
#   simulations/missing_outcomes/miss_scripts/miss_bash_scripts/miss_m2_bash_scripts/run_missing_two_stage_m2.bash