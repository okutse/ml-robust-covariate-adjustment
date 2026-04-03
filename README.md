# ml-robust-covariate-adjustment
This repository holds reproducibility files for the paper "Machine learning for robust covariate adjustment: enhancing efficiency and mitigating bias in randomized trials with missing outcomes."

## Reproducible R environment (renv)
This repo uses renv to freeze package versions for reproducibility.

Setup (first time only):
1. Run `Rscript scripts/setup_renv.R` to initialize renv and create `renv.lock`.
2. Commit the generated `renv.lock` to record the exact package versions.

Restore on another machine:
1. Open R in the repo root.
2. Run `renv::restore()` to install the exact locked versions.
