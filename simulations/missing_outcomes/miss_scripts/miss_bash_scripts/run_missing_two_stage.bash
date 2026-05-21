#!/bin/bash
#SBATCH --job-name=miss_two_stage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=115G
#SBATCH --time=72:00:00
#SBATCH --output=logs/missing_outcomes/two_stage_%A.out
#SBATCH --error=logs/missing_outcomes/two_stage_%A.err
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
	if [[ -f "$candidate/renv/activate.R" && -f "$candidate/simulations/missing_outcomes/miss_scripts/run_missing_two_stage.R" ]]; then
		project_root="$candidate"
		break
	fi
done

if [[ -z "$project_root" ]]; then
	echo "Unable to locate project root with renv/activate.R and run_missing_two_stage.R" >&2
	echo "Checked: $script_root_guess and $submit_root" >&2
	exit 1
fi

cd "$project_root"
echo "Using project_root=$project_root"

# Activate renv first, then restore the project library before any analysis code runs.
Rscript -e 'source("renv/activate.R"); renv::restore(prompt = FALSE)'

# Example:
# sbatch simulations/missing_outcomes/miss_scripts/miss_bash_scripts/run_missing_two_stage.bash
# SETTING=setting_one DATA_SOURCE=local RESET_CHECKPOINT=true \
# sbatch --export=SETTING,DATA_SOURCE,RESET_CHECKPOINT simulations/missing_outcomes/miss_scripts/miss_bash_scripts/run_missing_two_stage.bash

Rscript simulations/missing_outcomes/miss_scripts/run_missing_two_stage.R
