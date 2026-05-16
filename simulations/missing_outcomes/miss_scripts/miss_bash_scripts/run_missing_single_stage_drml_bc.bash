#!/bin/bash
#SBATCH --job-name=miss_single_stage_drml_bc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/missing_outcomes/single_stage_drml_bc_%A.out
#SBATCH --error=logs/missing_outcomes/single_stage_drml_bc_%A.err

set -euo pipefail
module load r/4.5.1

# Example:
# sbatch simulations/missing_outcomes/miss_scripts/miss_bash_scripts/run_missing_single_stage_drml_bc.bash
# SETTING=setting_one DATA_SOURCE=local RESET_CHECKPOINT=true \
# sbatch --export=SETTING,DATA_SOURCE,RESET_CHECKPOINT simulations/missing_outcomes/miss_scripts/miss_bash_scripts/run_missing_single_stage_drml_bc.bash

Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage_drml_bc.R
