#!/bin/bash
#SBATCH --job-name=complete_data_two_stage_cf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=115G
#SBATCH --time=72:00:00
#SBATCH --output=logs/complete_data/two_stage_cf_%A.out
#SBATCH --error=logs/complete_data/two_stage_cf_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amos_okutse@brown.edu

set -euo pipefail
module load r/4.5.1

# Resolve the repository root robustly across submission directories and mounted path aliases.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script_root_guess="$(cd "$script_dir/../../../.." && pwd)"
submit_root="${SLURM_SUBMIT_DIR:-$PWD}"

project_root=""
for candidate in "$script_root_guess" "$submit_root"; do
	if [[ -f "$candidate/renv/activate.R" && -f "$candidate/simulations/complete_data/scripts/cf/two_stage_cf_results.R" ]]; then
		project_root="$candidate"
		break
	fi
done

if [[ -z "$project_root" ]]; then
	echo "Unable to locate project root with renv/activate.R and two_stage_cf_results.R" >&2
	echo "Checked: $script_root_guess and $submit_root" >&2
	exit 1
fi

cd "$project_root"
echo "Using project_root=$project_root"

# Activate renv first, then restore the project library before any analysis code runs.
Rscript -e 'source("renv/activate.R"); renv::restore(prompt = FALSE)'

# Example:
# sbatch simulations/complete_data/scripts/cf_bash_scripts/run_two_stage_cf.bash
# DATA_SOURCE=local RESET_CHECKPOINT=true REPLICATE_RETRIES=3 \
# sbatch --export=DATA_SOURCE,RESET_CHECKPOINT,REPLICATE_RETRIES simulations/complete_data/scripts/cf_bash_scripts/run_two_stage_cf.bash

Rscript simulations/complete_data/scripts/cf/two_stage_cf_results.R
