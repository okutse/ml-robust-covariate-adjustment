#!/bin/bash
#SBATCH --job-name=two_stage_cf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=115G
#SBATCH --time=48:00:00
#SBATCH --output=logs/complete_data/two_stage_cf_%A.out
#SBATCH --error=logs/complete_data/two_stage_cf_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amos_okutse@brown.edu

set -euo pipefail
module load r/4.5.1

# Run the job from the Slurm submission directory so renv and project-relative paths resolve correctly.
project_root="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$project_root"

# Activate renv first, then restore the project library before any analysis code runs.
Rscript -e 'source("renv/activate.R"); renv::restore(prompt = FALSE)'

# Example:
# sbatch simulations/complete_data/scripts/cf_bash_scripts/run_two_stage_cf.bash
# DATA_SOURCE=local RESET_CHECKPOINT=true REPLICATE_RETRIES=3 \
# sbatch --export=DATA_SOURCE,RESET_CHECKPOINT,REPLICATE_RETRIES simulations/complete_data/scripts/cf_bash_scripts/run_two_stage_cf.bash

Rscript simulations/complete_data/scripts/cf/two_stage_cf_results.R
