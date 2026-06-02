#!/bin/bash
#SBATCH --job-name=miss_single_stage_drml_bc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=80G
#SBATCH --time=72:00:00 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amos_okutse@brown.edu

set -euo pipefail
module load r/4.5.1

# Resolve repository root robustly across path aliases (/users vs /oscar/home).
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script_root_guess="$(cd "$script_dir/../../../.." && pwd)"
submit_root="${SLURM_SUBMIT_DIR:-$PWD}"

project_root=""
for candidate in "$script_root_guess" "$submit_root"; do
	if [[ -f "$candidate/renv/activate.R" && -f "$candidate/simulations/missing_outcomes/miss_scripts/run_missing_single_stage_drml_bc.R" ]]; then
		project_root="$candidate"
		break
	fi
done

if [[ -z "$project_root" ]]; then
	echo "Unable to locate project root with renv/activate.R and run_missing_single_stage_drml_bc.R" >&2
	echo "Checked: $script_root_guess and $submit_root" >&2
	exit 1
fi

cd "$project_root"
echo "Using project_root=$project_root"

# Activate renv first, then restore the project library before any analysis code runs.
Rscript -e 'source("renv/activate.R"); renv::restore(prompt = FALSE)'

# Example:
# sbatch simulations/missing_outcomes/miss_scripts/miss_bash_scripts/run_missing_single_stage_drml_bc.bash
# setting=setting_four
# mkdir -p "$repo/logs/missing_outcomes/$setting"
# sbatch --chdir="$repo" \
#   --job-name="miss_${setting}_single_stage_drml_bc" \
#   --output="$repo/logs/missing_outcomes/$setting/single_stage_drml_bc_%j.out" \
#   --error="$repo/logs/missing_outcomes/$setting/single_stage_drml_bc_%j.err" \
#   --export=SETTING=$setting,DATA_SOURCE=archive,ARCHIVE_DATASETS_DIR=$repo/simulations/missing_outcomes/archives/zenodo/zenodo_datasets.zip \
#   simulations/missing_outcomes/miss_scripts/miss_bash_scripts/run_missing_single_stage_drml_bc.bash

Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage_drml_bc.R
