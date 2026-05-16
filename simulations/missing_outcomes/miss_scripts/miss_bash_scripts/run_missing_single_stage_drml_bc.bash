#!/bin/bash
#SBATCH --job-name=miss_single_stage_drml_bc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/missing_outcomes/single_stage_drml_bc_%A.out
#SBATCH --error=logs/missing_outcomes/single_stage_drml_bc_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amos_okutse@brown.edu

set -euo pipefail
module load r/4.5.1

# Run the job from the repository root so renv and project-relative paths resolve correctly.
# Run the job from the Slurm submission directory so renv and project-relative paths resolve correctly.
project_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$project_root"

# Activate renv first, then restore the project library before any analysis code runs.
Rscript -e 'source("renv/activate.R"); renv::restore(prompt = FALSE)'

# Example:
# sbatch simulations/missing_outcomes/miss_scripts/miss_bash_scripts/run_missing_single_stage_drml_bc.bash
# SETTING=setting_one DATA_SOURCE=local RESET_CHECKPOINT=true \
# sbatch --export=SETTING,DATA_SOURCE,RESET_CHECKPOINT simulations/missing_outcomes/miss_scripts/miss_bash_scripts/run_missing_single_stage_drml_bc.bash

Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage_drml_bc.R
